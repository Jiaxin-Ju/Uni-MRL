import os
import csv
import math
import time
import random
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler

from torch_scatter import scatter
from torch_geometric.data import Data, Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')
from dataset.llm4sd_vector import *

ATOM_LIST = list(range(0, 119))
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER
]
BOND_LIST = [BT.SINGLE, BT.DOUBLE, BT.TRIPLE, BT.AROMATIC]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT
]


def read_smiles(data_path, target, task):
    smiles_data, labels = [], []
    with open(data_path) as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=',')
        for i, row in enumerate(csv_reader):
            if i != 0:
                smiles = row['smiles']
                label = row[target]
                mol = Chem.MolFromSmiles(smiles)
                if mol != None and label != '':
                    smiles_data.append(smiles)
                    if task == 'classification':
                        labels.append(int(label))
                    elif task == 'regression':
                        labels.append(float(label))
                    else:
                        ValueError('task must be either regression or classification')
    print(len(smiles_data))
    return smiles_data, labels


class MolTestDataset(Dataset):
    def __init__(self, data_path, target, task, dataset_name,
                 subtask, model, knowledge_type, num_samples, scaler=None):
        super(Dataset, self).__init__()
        self.smiles_data, self.labels = read_smiles(data_path, target, task)
        self.task = task

        self.conversion = 1
        if 'qm9' in data_path and target in ['homo', 'lumo', 'gap', 'zpve', 'u0']:
            self.conversion = 27.211386246
            print(target, 'Unit conversion needed!')

        self.llm4sd_book = {}
        all_features = []
        llm4sd_features_code, function_names = load_llm4sd_features(dataset_name, subtask, model,
                                                                    knowledge_type, num_samples)
        for smiles_str in self.smiles_data:
            smile_feat = gen_smile_feature(llm4sd_features_code, smiles_str, function_names)
            self.llm4sd_book[smiles_str] = smile_feat
            all_features.append(smile_feat)

        self.scaler = scaler  # Store scaler
        if self.scaler is not None:
            self.apply_normalization(self.scaler)

    def apply_normalization(self, scaler):
        all_features = np.array([self.llm4sd_book[smiles] for smiles in self.smiles_data])
        normalized_features = scaler.transform(all_features)
        for i, smiles_str in enumerate(self.smiles_data):
            self.llm4sd_book[smiles_str] = normalized_features[i]

        assert len(set([len(i) for i in self.llm4sd_book.values()])) == 1

    def __getitem__(self, index):
        smiles_str = self.smiles_data[index]
        llm4sd_x = torch.tensor(self.llm4sd_book[smiles_str]).float()
        emb_dim = llm4sd_x.size(0)
        llm4sd_x = llm4sd_x.view(1, emb_dim)

        mol = Chem.MolFromSmiles(self.smiles_data[index])
        mol = Chem.AddHs(mol)

        N = mol.GetNumAtoms()
        M = mol.GetNumBonds()

        type_idx = []
        chirality_idx = []
        atomic_number = []
        for atom in mol.GetAtoms():
            type_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))
            chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
            atomic_number.append(atom.GetAtomicNum())

        x1 = torch.tensor(type_idx, dtype=torch.long).view(-1, 1)
        x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1, 1)
        x = torch.cat([x1, x2], dim=-1)

        row, col, edge_feat = [], [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]
            edge_feat.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])
            edge_feat.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])

        edge_index = torch.tensor([row, col], dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.long)
        if self.task == 'classification':
            y = torch.tensor(self.labels[index], dtype=torch.long).view(1, -1)
        elif self.task == 'regression':
            y = torch.tensor(self.labels[index] * self.conversion, dtype=torch.float).view(1, -1)
        data = Data(x=x, y=y, llm4sd_x=llm4sd_x, edge_index=edge_index, edge_attr=edge_attr)
        return data

    def __len__(self):
        return len(self.smiles_data)


class MolTestDatasetWrapper(object):

    def __init__(self,
                 batch_size, num_workers, dataset_name, subtask, model,
                 knowledge_type, num_samples, train_data_path, valid_data_path,
                 test_data_path, task, target):
        super(object, self).__init__()
        self.train_data_path = train_data_path
        self.valid_data_path = valid_data_path
        self.test_data_path = test_data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.target = target
        self.task = task
        self.dataset_name = dataset_name
        self.subtask = subtask
        self.model = model
        self.knowledge_type = knowledge_type
        self.num_samples = num_samples
        self.scaler = StandardScaler()

    def get_data_loaders(self):
        train_dataset = MolTestDataset(self.train_data_path, self.target, self.task, self.dataset_name,
                                       self.subtask, self.model, self.knowledge_type, self.num_samples, scaler=None)
        train_features = np.array([train_dataset.llm4sd_book[smiles] for smiles in train_dataset.smiles_data])
        self.scaler.fit(train_features)  # Fit scaler on training features

        train_dataset.apply_normalization(self.scaler)
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, drop_last=False
        )

        valid_dataset = MolTestDataset(self.valid_data_path, self.target, self.task, self.dataset_name,
                                       self.subtask, self.model, self.knowledge_type, self.num_samples,
                                       scaler=self.scaler)
        valid_loader = DataLoader(
            valid_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, drop_last=False
        )

        test_dataset = MolTestDataset(self.test_data_path, self.target, self.task, self.dataset_name,
                                      self.subtask, self.model, self.knowledge_type, self.num_samples,
                                      scaler=self.scaler)
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, drop_last=False
        )
        return train_loader, valid_loader, test_loader