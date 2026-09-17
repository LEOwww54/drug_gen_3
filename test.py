import pickle
import json
from rdkit import Chem

from decompose.molConn import gen2mol

if "__main__" == __name__:
    file1 = open('gpt/frag_file/frag_decom_ZINC_250K_pamf_train.pkl', 'rb')
    data1 = pickle.load(file1)

    frags1 = [i['frag'] for k, i in data1['mol'].items()]
    smiles1 = [i['oring'] for k, i in data1['mol'].items()]

    file2 = open('gpt/frag_file/frag_decom_ZINC_refined_pamf_train.pkl', 'rb')
    data2 = pickle.load(file2)

    frags2= [i['frag'] for k, i in data2['mol'].items()]
    smiles2 = [i['oring'] for k, i in data2['mol'].items()]

    frags2smiles1 = gen2mol(frags1)
    frags2smiles1 = [Chem.MolToSmiles(Chem.MolFromSmiles(smi[1]), canonical=True) for smi in frags2smiles1]
    json.dump(frags2smiles1, open('zinc_250_train_recon.json', 'w'), indent=4)

    smiles1 = [Chem.MolToSmiles(Chem.MolFromSmiles(smi), canonical=True, isomericSmiles=False) for smi in smiles1]
    json.dump(smiles1, open('zinc_250_train_smiles.json', 'w'), indent=4)

    for i in range(len(smiles)):
        if not frags2smiles1[i] == smiles[i]:
            print(f"{i}: original smiles: {smiles[i]}------recon smiles: {frags2smiles[i]}")