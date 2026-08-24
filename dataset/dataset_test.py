"""Scaffold-split molecular datasets used for Uni-MRL fine-tuning."""

import csv
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem.rdchem import BondType as BT
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from dataset.llm4sd_vector import gen_smile_feature, load_llm4sd_features


RDLogger.DisableLog("rdApp.*")

# Preserve the encoding used by the archived paper fine-tuning runs. Atomic
# number N is mapped to index N because that code used range(0, 119).
ATOM_LIST = list(range(0, 119))
CHIRALITY_TO_INDEX = {
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED: 0,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW: 1,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW: 2,
    Chem.rdchem.ChiralType.CHI_OTHER: 3,
}
BOND_TO_INDEX = {
    BT.SINGLE: 0,
    BT.DOUBLE: 1,
    BT.TRIPLE: 2,
    BT.AROMATIC: 3,
}
BOND_DIRECTION_TO_INDEX = {
    Chem.rdchem.BondDir.NONE: 0,
    Chem.rdchem.BondDir.ENDUPRIGHT: 1,
    Chem.rdchem.BondDir.ENDDOWNRIGHT: 2,
}


def read_smiles(data_path, target, task):
    """Read molecules with the row selection used by the paper experiment code."""
    data_path = Path(data_path)
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset split not found: {data_path}")

    smiles_data = []
    labels = []
    with data_path.open(newline="", encoding="utf-8") as csv_file:
        csv_reader = csv.DictReader(csv_file)
        required = {"smiles", target}
        missing = required.difference(csv_reader.fieldnames or ())
        if missing:
            raise KeyError(
                f"Missing columns {sorted(missing)} in {data_path}; "
                f"available columns are {csv_reader.fieldnames}"
            )
        for row_index, row in enumerate(csv_reader):
            # The archived code skipped row zero after DictReader consumed the
            # header. Retain this unusual behavior for numerical parity.
            if row_index == 0:
                continue
            smiles = row["smiles"]
            label = row[target]
            if not smiles or not label or Chem.MolFromSmiles(smiles) is None:
                continue
            smiles_data.append(smiles)
            if task == "classification":
                labels.append(int(float(label)))
            elif task == "regression":
                labels.append(float(label))
            else:
                raise ValueError("task must be classification or regression")

    if not smiles_data:
        raise ValueError(f"No valid labeled molecules found in {data_path}")
    print(f"Loaded {len(smiles_data)} molecules from {data_path}")
    return smiles_data, labels


def molecule_to_graph(molecule):
    """Convert a molecule using the archived paper fine-tuning encoding."""
    molecule = Chem.AddHs(molecule)
    atom_features = []
    for atom in molecule.GetAtoms():
        atomic_number = atom.GetAtomicNum()
        if atomic_number not in ATOM_LIST:
            raise ValueError(f"Unsupported atomic number: {atomic_number}")
        chirality = CHIRALITY_TO_INDEX[atom.GetChiralTag()]
        atom_features.append([ATOM_LIST.index(atomic_number), chirality])

    rows = []
    columns = []
    edge_features = []
    for bond in molecule.GetBonds():
        bond_type = BOND_TO_INDEX.get(bond.GetBondType())
        if bond_type is None:
            raise ValueError(f"Unsupported bond type: {bond.GetBondType()}")
        bond_direction = BOND_DIRECTION_TO_INDEX[bond.GetBondDir()]
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        rows.extend((begin, end))
        columns.extend((end, begin))
        edge_features.extend(((bond_type, bond_direction), (bond_type, bond_direction)))

    x = torch.tensor(atom_features, dtype=torch.long)
    edge_index = torch.tensor([rows, columns], dtype=torch.long)
    edge_attr = torch.tensor(edge_features, dtype=torch.long).reshape(-1, 2)
    return x, edge_index, edge_attr


class MolTestDataset(Dataset):
    def __init__(
        self,
        data_path,
        target,
        task,
        dataset_name,
        subtask,
        model,
        knowledge_type,
        num_samples,
        scaler=None,
        rule_root=None,
        use_llm_features=True,
    ):
        self.smiles_data, self.labels = read_smiles(data_path, target, task)
        self.task = task
        self.llm4sd_book = {}

        if use_llm_features:
            feature_code, function_names = load_llm4sd_features(
                dataset_name,
                subtask,
                model,
                knowledge_type,
                num_samples,
                rule_root=rule_root,
            )
            for smiles in self.smiles_data:
                self.llm4sd_book[smiles] = gen_smile_feature(
                    feature_code, smiles, function_names
                )
        else:
            self.llm4sd_book = {smiles: [0.0] for smiles in self.smiles_data}

        feature_lengths = {len(features) for features in self.llm4sd_book.values()}
        if len(feature_lengths) != 1 or next(iter(feature_lengths)) == 0:
            raise ValueError(f"Inconsistent or empty LLM feature vectors: {feature_lengths}")
        self.feature_dim = next(iter(feature_lengths))
        if scaler is not None:
            self.apply_normalization(scaler)

    def feature_matrix(self):
        return np.asarray(
            [self.llm4sd_book[smiles] for smiles in self.smiles_data],
            dtype=np.float64,
        )

    def apply_normalization(self, scaler):
        normalized = scaler.transform(self.feature_matrix())
        for index, smiles in enumerate(self.smiles_data):
            self.llm4sd_book[smiles] = normalized[index].astype(np.float32)

    def __getitem__(self, index):
        smiles = self.smiles_data[index]
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError(f"Invalid SMILES at index {index}: {smiles}")
        x, edge_index, edge_attr = molecule_to_graph(molecule)
        llm4sd_x = torch.as_tensor(
            self.llm4sd_book[smiles], dtype=torch.float32
        ).reshape(1, -1)
        if self.task == "classification":
            y = torch.tensor(self.labels[index], dtype=torch.long).reshape(1)
        else:
            y = torch.tensor(self.labels[index], dtype=torch.float32).reshape(1)
        return Data(
            x=x,
            y=y,
            llm4sd_x=llm4sd_x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            smiles=smiles,
        )

    def __len__(self):
        return len(self.smiles_data)


class MolTestDatasetWrapper:
    """Build splits and fit the rule-feature scaler on training data only."""

    def __init__(
        self,
        batch_size,
        num_workers,
        dataset_name,
        subtask,
        model,
        knowledge_type,
        num_samples,
        train_data_path,
        valid_data_path,
        test_data_path,
        task,
        target,
        rule_root=None,
        use_llm_features=True,
    ):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_args = {
            "target": target,
            "task": task,
            "dataset_name": dataset_name,
            "subtask": subtask,
            "model": model,
            "knowledge_type": knowledge_type,
            "num_samples": num_samples,
            "rule_root": rule_root,
            "use_llm_features": use_llm_features,
        }
        self.paths = {
            "train": train_data_path,
            "valid": valid_data_path,
            "test": test_data_path,
        }
        self.scaler = StandardScaler()
        self.feature_dim = None

    def get_data_loaders(self):
        train_dataset = MolTestDataset(
            self.paths["train"], scaler=None, **self.dataset_args
        )
        self.scaler.fit(train_dataset.feature_matrix())
        train_dataset.apply_normalization(self.scaler)
        self.feature_dim = train_dataset.feature_dim

        valid_dataset = MolTestDataset(
            self.paths["valid"], scaler=self.scaler, **self.dataset_args
        )
        test_dataset = MolTestDataset(
            self.paths["test"], scaler=self.scaler, **self.dataset_args
        )
        if not (train_dataset.feature_dim == valid_dataset.feature_dim == test_dataset.feature_dim):
            raise ValueError("Rule feature dimensions differ across data splits")

        common = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "drop_last": False,
        }
        return (
            DataLoader(train_dataset, shuffle=True, **common),
            # These shuffles consume the global RNG between epochs in the old
            # workflow and therefore affect later training-batch order.
            DataLoader(valid_dataset, shuffle=True, **common),
            DataLoader(test_dataset, shuffle=True, **common),
        )
