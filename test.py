import pickle
import json
from rdkit import Chem

from decompose.molConn import gen2mol

if "__main__" == __name__:
    file = open('gpt/frag_file/frag_decom_ZINC_250K_pamf_train.pkl', 'rb')
    data = pickle.load(file)

    frags = [i['frag'] for k, i in data['mol'].items()]
    smiles = [i['oring'] for k, i in data['mol'].items()]

    frags2smiles = gen2mol(frags)
    frags2smiles = [Chem.MolToSmiles(Chem.MolFromSmiles(smi[1]), canonical=True) for smi in frags2smiles]
    json.dump(frags2smiles, open('zinc_250_train_recon.json', 'w'), indent=4)

    smiles = [Chem.MolToSmiles(Chem.MolFromSmiles(smi), canonical=True, isomericSmiles=False) for smi in smiles]
    json.dump(smiles, open('zinc_250_train_smiles.json', 'w'), indent=4)

    for i in range(len(smiles)):
        if not frags2smiles[i] == smiles[i]:
            print(f"{i}: original smiles: {smiles[i]}------recon smiles: {frags2smiles[i]}")