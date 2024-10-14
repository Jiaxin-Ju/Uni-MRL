import numpy as np
import math
from math import sqrt
import pandas as pd
import os
import json
import warnings

from rdkit import Chem, rdBase
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Descriptors, Crippen, Lipinski, MolSurf, Fragments
from rdkit.Chem.rdMolDescriptors import CalcNumAliphaticCarbocycles, CalcNumAromaticCarbocycles
from mordred import Weight, WienerIndex, RotatableBond, EccentricConnectivityIndex
from rdkit.Chem import rdchem
from rdkit.Chem import rdmolops
from rdkit.Chem import AllChem
import rdkit


def gen_smile_feature(generated_code, smiles, valid_function_names):
    smiles_feat = []
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            for function_name in valid_function_names:  # Loop over the valid function names
                try:
                    feature = globals()[function_name](mol)
                    if feature is not None and isinstance(feature, (int, float)):
                        smiles_feat.append(feature)
                    else:
                        smiles_feat.append(0)
                except Exception as e:
                    print(f"Unexpected error in function {function_name}: {str(e)}")
                    smiles_feat.append(0)
        else:
            print(f"Error in generating features for SMILES: {smiles}")
            smiles_feat = [0] * len(valid_function_names)
    except:
        print(f"Error in generating features for SMILES: {smiles}")
        smiles_feat = [0] * len(valid_function_names)
    return smiles_feat


def load_llm4sd_features(dataset="bbbp", subtask="", model="falcon-40b",knowledge_type="all",num_samples=50):
    if subtask != '':
        if dataset in ['tox21', 'sider']:
            subtask_name = f'{dataset}_{subtask}'
        elif dataset == 'qm9' and num_samples == 50:
            subtask_name = subtask
        else:
            raise NotImplementedError(f"Folder Name error")
    else:
        subtask_name = dataset

    llm4sd_code_folder = "dataset/eval_code_generation_repo"

    feature_file_folder = os.path.join(llm4sd_code_folder, model, dataset)
    synthesize_folder = os.path.join(feature_file_folder, 'synthesize')
    synthesize_file_name = f'{model}_{subtask_name}_pk_rules.txt'
    synthesize_file_path = os.path.join(synthesize_folder, synthesize_file_name)

    if knowledge_type != 'synthesize' and num_samples not in [30, 50]:
        raise NotImplementedError(f"num_samples should be 30 or 50")

    inference_folder = os.path.join(feature_file_folder, 'inference', f"sample_{num_samples}")
    inference_file_name = f'{model}_{subtask_name}_dk_rules.txt'
    inference_file_path = os.path.join(inference_folder, inference_file_name)

    if knowledge_type == 'synthesize':
        with open(synthesize_file_path, 'r') as f:
            generated_code = f.read()
        print(f"Loading llm4sd features from {synthesize_file_path}")
    elif knowledge_type == 'inference':
        with open(inference_file_path, 'r') as f:
            generated_code = f.read()
        print(f"Loading llm4sd features from {inference_file_path}")
    elif knowledge_type == 'all':
        with open(synthesize_file_path, 'r') as f:
            synthesize_code = f.read()
        with open(inference_file_path, 'r') as f:
            inference_code = f.read()
        generated_code = synthesize_code + '\n' + inference_code  # combine synthesize_code and inference_code
        print(f"Loading llm4sd features from: \n{synthesize_file_path}, \n{inference_file_path}")
    else:
        raise NotImplementedError(f"Knowledge_type is wrong.(synthesize/inference/all)")

    exec(generated_code, globals())
    gen_function_names = [line.split()[1].split('(')[0] for line in generated_code.split('\n') if line.startswith('def ')]

    return generated_code, gen_function_names


if __name__ == '__main__':
    smiles_string = "C[C@H](N)Cc1ccccc1"
    llm4sd_features_code, function_names = load_llm4sd_features()
    smile_feat = gen_smile_feature(llm4sd_features_code, smiles_string, function_names)
    print(smile_feat)

