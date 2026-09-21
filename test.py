import pickle
import json
from rdkit import Chem

from decompose.molConn import gen2mol

if "__main__" == __name__:
    with open('stru_data_ZINC_250K_pamf_train.json', 'r') as f:
        stru_data = json.load(f)
        stru_data = list(stru_data['pamf'].keys())
        stru_smiles = [Chem.MolToSmiles(Chem.MolFromSmiles(i), isomericSmiles=False, canonical=True) for i in stru_data]

    file_path = 'gen/pR_1_1'
    with open(file_path, 'r') as file:
        smiles = [i.strip('\n') for i in file.readlines()]
        smiles = [Chem.MolToSmiles(Chem.MolFromSmiles(i), canonical=True) for i in smiles]
        pass

    new = []
    for i in smiles:
        if not i in stru_smiles:
            new.append(i)

    print(len(new))
    print(new)