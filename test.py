import pickle
import json
from rdkit import Chem

from decompose.molConn import gen2mol

def patest(file):
    with open(file, 'rb') as f:
        data = pickle.load(f)
        frags = [i['frag'].split(' ') for k, i in data['mol'].items()]

        error = []

        for frag in frags:
            count = 0
            for token in frag:
                if token=='(':
                    count += 1
                if token==')':
                    count -= 1
            if not count == 0:
                error.append(frag)

        print(error)

def test1():
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

if "__main__" == __name__:
    patest('gpt/frag_file/frag_decom_ZINC_250K_pamf_train.pkl')