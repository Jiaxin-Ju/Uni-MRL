"""Dataset metadata shared by the training CLI and setup checks."""

from pathlib import Path


SIMPLE_DATASETS = {
    "esol": ("regression", "ESOL predicted log solubility in mols per litre"),
    "freesolv": ("regression", "expt"),
    "lipophilicity": ("regression", "exp"),
}

QM9_TARGETS = {
    "mu": "mu",
    "alpha": "alpha",
    "r2": "R^2",
    "zpve": "ZPVE",
    "cv": "c_v",
    "deltaepsilon": "Delta_epsilon",
    "epsilonhomo": "epsilon_HOMO",
    "epsilonlumo": "epsilon_LUMO",
    "u0": "U_0",
    "u": "U",
    "h": "H",
    "g": "G",
}


def normalize_name(value):
    """Normalize case and punctuation for user-facing task aliases."""
    return "".join(character for character in value.lower() if character.isalnum())


def canonical_qm9_target(subtask):
    key = normalize_name(subtask)
    if key not in QM9_TARGETS:
        supported = ", ".join(QM9_TARGETS.values())
        raise ValueError(f"Unsupported QM9 subtask {subtask!r}. Choose one of: {supported}")
    return QM9_TARGETS[key]


def dataset_task(dataset_name):
    dataset_name = dataset_name.lower()
    if dataset_name in SIMPLE_DATASETS:
        return SIMPLE_DATASETS[dataset_name][0]
    if dataset_name == "qm9":
        return "regression"
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def resolve_targets(dataset_name, subtask=None, all_subtasks=False):
    """Return exact CSV column names for a dataset invocation."""
    dataset_name = dataset_name.lower()
    if dataset_name in SIMPLE_DATASETS:
        return [SIMPLE_DATASETS[dataset_name][1]]

    if dataset_name == "qm9":
        if all_subtasks:
            return list(QM9_TARGETS.values())
        if not subtask:
            raise ValueError("--subtask is required for QM9")
        return [canonical_qm9_target(subtask)]

    raise ValueError(
        f"Unsupported dataset: {dataset_name}. The paper uses ESOL, FreeSolv, "
        "Lipophilicity, and QM9."
    )


def split_paths(dataset_root, dataset_name, target):
    """Resolve the bundled train/valid/test scaffold split files."""
    dataset_name = dataset_name.lower()
    if dataset_name == "qm9":
        stem = target.lower()
    else:
        stem = dataset_name
    folder = Path(dataset_root) / dataset_name
    return {
        split: folder / f"{stem}_{split}.csv"
        for split in ("train", "valid", "test")
    }


def rule_subtask(dataset_name, target):
    """Return the task token used in the LLM rule repository."""
    dataset_name = dataset_name.lower()
    return target if dataset_name == "qm9" else ""
