import moses

with open('generated_smiles.txt', 'r') as f:
    smiles = f.readlines()
    smiles = [x.strip() for x in smiles]
    metrics = moses.get_all_metrics(smiles)