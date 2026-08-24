"""Validate the environment, bundled assets, rule code, and a model forward pass."""

import argparse
import ast
import csv
import importlib
import sys
from pathlib import Path

from dataset.rule_repository import resolve_rule_file
from utils.tasks import (
    QM9_TARGETS,
    SIMPLE_DATASETS,
    split_paths,
)


PROJECT_ROOT = Path(__file__).resolve().parent
REQUIRED_MODULES = (
    "torch",
    "torchvision",
    "torch_geometric",
    "torch_scatter",
    "torch_sparse",
    "numpy",
    "scipy",
    "pandas",
    "sklearn",
    "networkx",
    "rdkit",
    "mordred",
    "yaml",
    "tensorboard",
)


def check_python_sources():
    files = list(PROJECT_ROOT.rglob("*.py"))
    for path in files:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    print(f"[ok] parsed {len(files)} Python source files")


def check_rule_sources():
    root = PROJECT_ROOT / "dataset" / "eval_code_generation_repo"
    files = list(root.rglob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No bundled rule files below {root}")
    function_count = 0
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        function_count += sum(isinstance(node, ast.FunctionDef) for node in tree.body)
    print(f"[ok] parsed {len(files)} rule files containing {function_count} functions")


def check_assets():
    required = (
        PROJECT_ROOT / "ckpt" / "pretrained_gin" / "checkpoints" / "model.pth",
        PROJECT_ROOT / "ckpt" / "pretrained_gcn" / "checkpoints" / "model.pth",
        PROJECT_ROOT / "dataset" / "scaffold_datasets" / "esol" / "esol_train.csv",
        PROJECT_ROOT / "dataset" / "scaffold_datasets" / "freesolv" / "freesolv_train.csv",
        PROJECT_ROOT / "dataset" / "scaffold_datasets" / "lipophilicity" / "lipophilicity_train.csv",
    )
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Required asset is missing or empty: {path}")

    expected_targets = {
        dataset: (target,) for dataset, (_task, target) in SIMPLE_DATASETS.items()
    }
    expected_targets["qm9"] = tuple(QM9_TARGETS.values())
    split_count = 0
    for dataset, targets in expected_targets.items():
        for target in targets:
            paths = split_paths(PROJECT_ROOT / "dataset" / "scaffold_datasets", dataset, target)
            for split, path in paths.items():
                if not path.is_file() or path.stat().st_size == 0:
                    raise FileNotFoundError(f"Missing or empty {split} split: {path}")
                with path.open(newline="", encoding="utf-8") as handle:
                    reader = csv.DictReader(handle)
                    fields = reader.fieldnames or []
                    if "smiles" not in fields or target not in fields:
                        raise KeyError(f"Unexpected columns in {path}: {fields}")
                    if next(reader, None) is None:
                        raise ValueError(f"Dataset split contains no records: {path}")
                split_count += 1

    print(
        "[ok] pretrained GIN/GCN checkpoints and "
        f"{split_count} labeled scaffold splits are present"
    )


def check_paper_rule_matrix():
    """Check the rule sets used by the paper's physical/QM9 experiments."""
    physical_models = (
        "galactica-6.7b",
        "galactica-30b",
        "chemdfm",
        "falcon-7b",
        "falcon-40b",
    )
    resolved = []
    for model in physical_models:
        for dataset in ("esol", "freesolv", "lipophilicity"):
            for kind in ("synthesize", "inference"):
                resolved.append(
                    resolve_rule_file(
                        dataset,
                        "",
                        model,
                        kind,
                        num_samples=30,
                    )
                )

    for target in QM9_TARGETS.values():
        for kind in ("synthesize", "inference"):
            resolved.append(
                resolve_rule_file(
                    "qm9",
                    target,
                    "galactica-30b",
                    kind,
                    num_samples=50,
                )
            )

    if any(not path.is_file() for path in resolved):
        raise FileNotFoundError("One or more paper rule files could not be resolved")
    print(f"[ok] resolved {len(resolved)} paper experiment rule files")


def check_imports():
    failures = []
    for name in REQUIRED_MODULES:
        try:
            module = importlib.import_module(name)
            version = getattr(module, "__version__", "installed")
            print(f"[ok] import {name} ({version})")
        except Exception as exc:
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
    if failures:
        raise RuntimeError("Environment import failures:\n  " + "\n  ".join(failures))


def check_runtime_smoke():
    import torch
    from rdkit import Chem
    from torch_geometric.data import Batch, Data

    from dataset.dataset_test import molecule_to_graph, read_smiles
    from dataset.llm4sd_vector import gen_smile_feature, load_llm4sd_features
    from models.ginet_finetune_ws_contrastive import GINet
    from train import load_pretrained_encoder
    from utils.nt_xent import NTXentLoss

    code, names = load_llm4sd_features(
        dataset="esol",
        subtask="",
        model="galactica-6.7b",
        knowledge_type="all",
        num_samples=30,
    )
    rule_features = gen_smile_feature(code, "CCO", names)
    if len(rule_features) != 41:
        raise RuntimeError(
            f"Expected the archived 41 ESOL rule positions, got {len(rule_features)}"
        )

    carbon_x, _, _ = molecule_to_graph(Chem.MolFromSmiles("C"))
    if carbon_x[0, 0].item() != 6:
        raise RuntimeError("Paper-compatible atomic-number indexing is not active")

    esol_target = "ESOL predicted log solubility in mols per litre"
    esol_counts = []
    for split in ("train", "valid", "test"):
        path = PROJECT_ROOT / "dataset" / "scaffold_datasets" / "esol" / f"esol_{split}.csv"
        smiles, _ = read_smiles(path, esol_target, "regression")
        esol_counts.append(len(smiles))
    if esol_counts != [901, 112, 112]:
        raise RuntimeError(
            f"Paper-compatible ESOL row selection changed: {esol_counts}"
        )

    def graph(feature_scale):
        return Data(
            # Archived fine-tuning encoding maps C/O atomic numbers directly
            # to embedding indices 6/8.
            x=torch.tensor([[6, 0], [6, 0], [8, 0]], dtype=torch.long),
            edge_index=torch.tensor(
                [[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long
            ),
            edge_attr=torch.zeros((4, 2), dtype=torch.long),
            llm4sd_x=torch.tensor(rule_features, dtype=torch.float32).reshape(1, -1)
            * feature_scale,
            y=torch.tensor([0.0]),
        )

    batch = Batch.from_data_list((graph(1.0), graph(0.9)))
    model = GINet(
        task="regression",
        num_layer=5,
        emb_dim=300,
        feat_dim=512,
        drop_ratio=0.0,
        pool="mean",
        llm4sd_x_dim=len(rule_features),
    )
    load_pretrained_encoder(
        model,
        PROJECT_ROOT / "ckpt" / "pretrained_gin" / "checkpoints" / "model.pth",
        torch.device("cpu"),
    )
    model.eval()
    with torch.no_grad():
        graph_h, rule_h, prediction = model(batch)
        loss = NTXentLoss(temperature=0.5, use_cosine_similarity=True)(graph_h, rule_h)
    if graph_h.shape != (2, 512) or rule_h.shape != (2, 512) or prediction.shape != (2, 1):
        raise RuntimeError(
            f"Unexpected smoke-test shapes: {graph_h.shape}, {rule_h.shape}, {prediction.shape}"
        )
    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite contrastive loss: {loss}")
    print(
        "[ok] paper-compatible row/atom/rule semantics, GIN Uni-MRL forward "
        f"pass, and NT-Xent loss ({loss.item():.6f})"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--static-only",
        action="store_true",
        help="Check source and bundled assets without importing optional dependencies",
    )
    args = parser.parse_args()
    check_python_sources()
    check_rule_sources()
    check_assets()
    check_paper_rule_matrix()
    if not args.static_only:
        check_imports()
        check_runtime_smoke()
    print("All requested checks passed.")


if __name__ == "__main__":
    main()
