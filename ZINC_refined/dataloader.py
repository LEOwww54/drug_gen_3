def data_from_ZINC_refined(n, path=r'data\ZINC_refined.txt'):
    smiles = {}
    smiles_all = []
    with open(path, 'r') as file_:
        lines = file_.readlines()
        count= 0
        for line in lines[1:]:
            if n > 0 and count >= n:
                break
            count = count + 1
            s = line.split(',')
            smile = s[0]
            type = s[1]
            type = type.replace('\n', '')
            if type not in smiles:
                smiles[type] = [smile]
            else:
                smiles[type].append(smile)
            smiles_all.append(smile)

    return smiles, smiles_all