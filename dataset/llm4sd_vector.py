"""Load and evaluate the trusted LLM-generated molecular rule functions."""

import ast
import math
import numbers
from math import sqrt

import numpy as np
import rdkit
from mordred import EccentricConnectivityIndex, RotatableBond, Weight, WienerIndex
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem, Crippen, Descriptors, Fragments, Lipinski, MolSurf
from rdkit.Chem import rdchem, rdmolops, rdMolDescriptors
from rdkit.Chem.rdMolDescriptors import (
    CalcNumAliphaticCarbocycles,
    CalcNumAromaticCarbocycles,
)

from dataset.rule_repository import resolve_rule_file


def _read_rule_code(path):
    code = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(code, filename=str(path))
    except SyntaxError as exc:
        raise SyntaxError(f"Invalid generated rule code in {path}: {exc}") from exc
    function_names = [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]
    if not function_names:
        raise ValueError(f"No rule functions found in {path}")
    return code, function_names


def load_llm4sd_features(
    dataset="esol",
    subtask="",
    model="galactica-6.7b",
    knowledge_type="all",
    num_samples=50,
    rule_root=None,
):
    """Load trusted bundled rule code and return it with function names."""
    if knowledge_type not in {"synthesize", "inference", "all"}:
        raise ValueError("knowledge_type must be synthesize, inference, or all")

    kinds = (
        ("synthesize", "inference")
        if knowledge_type == "all"
        else (knowledge_type,)
    )
    code_blocks = []
    function_names = []
    loaded_paths = []
    for kind in kinds:
        path = resolve_rule_file(
            dataset,
            subtask,
            model,
            kind,
            num_samples=num_samples,
            rule_root=rule_root,
        )
        code, names = _read_rule_code(path)
        code_blocks.append(code)
        function_names.extend(names)
        loaded_paths.append(path)

    generated_code = "\n\n".join(code_blocks)
    # Match the archived paper code: both files share one global namespace.
    # Later inference definitions replace same-named synthesize definitions,
    # while both positions stay in the feature-name list.
    exec(compile(generated_code, "<bundled Uni-MRL rules>", "exec"), globals())
    print("Loaded LLM rule features from:")
    for path in loaded_paths:
        print(f"  {path}")
    return generated_code, function_names


def gen_smile_feature(_generated_code, smiles, valid_function_names):
    """Evaluate rule functions and coerce invalid/non-finite outputs to zero."""
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    features = []
    for function_name in valid_function_names:
        function = globals().get(function_name)
        if not callable(function):
            raise NameError(f"Generated rule function is unavailable: {function_name}")
        try:
            value = function(molecule)
            if value is not None and isinstance(value, (int, float)):
                features.append(value)
            else:
                features.append(0.0)
        except Exception as exc:
            print(f"Rule {function_name} failed for {smiles}: {exc}")
            features.append(0.0)
    return features


if __name__ == "__main__":
    sample_smiles = "C[C@H](N)Cc1ccccc1"
    feature_code, names = load_llm4sd_features()
    print(gen_smile_feature(feature_code, sample_smiles, names))
