"""Filesystem resolution for the heterogeneous historical rule-file names."""

import re
from pathlib import Path


DEFAULT_RULE_ROOT = Path(__file__).resolve().parent / "eval_code_generation_repo"


def _normalized(value):
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _case_insensitive_child(parent, name):
    wanted = name.lower()
    matches = [path for path in parent.iterdir() if path.is_dir() and path.name.lower() == wanted]
    if len(matches) != 1:
        available = ", ".join(sorted(path.name for path in parent.iterdir() if path.is_dir()))
        raise FileNotFoundError(f"Cannot resolve {name!r} below {parent}. Available: {available}")
    return matches[0]


def _rule_task_name(dataset, subtask):
    if dataset == "qm9":
        if not subtask:
            raise ValueError("A subtask is required for QM9 rules")
        return subtask
    return dataset


def resolve_rule_file(
    dataset,
    subtask,
    model,
    rule_kind,
    num_samples=50,
    rule_root=None,
):
    """Resolve original pk/dk names as well as descriptive later names."""
    dataset = dataset.lower()
    model = model.lower()
    rule_root = Path(rule_root) if rule_root else DEFAULT_RULE_ROOT
    model_folder = _case_insensitive_child(rule_root, model)
    dataset_folder = _case_insensitive_child(model_folder, dataset)
    task_name = _rule_task_name(dataset, subtask)

    if rule_kind == "synthesize":
        folder = dataset_folder / "synthesize"
        suffixes = ("pk_rules", "synthesize_rules")
    elif rule_kind == "inference":
        if num_samples not in {30, 50}:
            raise ValueError("num_samples must be 30 or 50")
        folder = dataset_folder / "inference" / f"sample_{num_samples}"
        suffixes = ("dk_rules", "inference_rules")
    else:
        raise ValueError("rule_kind must be 'synthesize' or 'inference'")

    if not folder.is_dir():
        raise FileNotFoundError(f"Rule folder does not exist: {folder}")

    exact_names = [folder / f"{model}_{task_name}_{suffix}.txt" for suffix in suffixes]
    for path in exact_names:
        if path.is_file():
            return path

    task_token = _normalized(f"{model}_{task_name}")
    suffix_tokens = tuple(_normalized(suffix) for suffix in suffixes)
    matches = []
    for path in folder.glob("*.txt"):
        stem = _normalized(path.stem)
        if task_token in stem and any(stem.endswith(suffix) for suffix in suffix_tokens):
            matches.append(path)

    if len(matches) == 1:
        return matches[0]
    if not matches:
        available = ", ".join(sorted(path.name for path in folder.glob("*.txt")))
        raise FileNotFoundError(
            f"No {rule_kind} rule file for model={model}, dataset={dataset}, "
            f"subtask={subtask!r}, num_samples={num_samples}. Available: {available}"
        )
    raise RuntimeError(f"Ambiguous rule files for {task_name}: {matches}")
