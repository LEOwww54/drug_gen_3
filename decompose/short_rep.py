from rdkit import Chem
import periodictable

def short_ring(smiles, Aromatic = True):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        mol = Chem.MolFromSmarts(smiles)
        if mol is None:
            return short_rep_smiles(smiles, ring=True, Aromatic=Aromatic)

    ringInfo = mol.GetRingInfo()
    ringNum = ringInfo.NumRings()

    atom_rings = list(ringInfo.AtomRings())
    atoms = []
    for i in atom_rings:
        for j in i:
            if not Aromatic:
                atoms.append((mol.GetAtomWithIdx(j).GetSymbol().upper(), mol.GetAtomWithIdx(j).GetAtomicNum()))
            else:
                atoms.append((mol.GetAtomWithIdx(j).GetSymbol(), mol.GetAtomWithIdx(j).GetAtomicNum()))
    atoms = set(atoms)
    atoms = sorted(atoms, key=lambda x: x[1])
    atoms_str = ''
    for i in atoms:
        atoms_str += str(i[0])

    connections = smiles.count('*')
    conn_str = ''
    if connections == 1:
        conn_str = '*'
    elif connections >= 2:
        conn_str = '*'
        for i in range(connections-1):
            conn_str += '(*)'

    ring_smiles = f'{ringNum}ring_{atoms_str}{conn_str}'
    pass
    return ring_smiles

def short_rep_smiles(smiles, Aromatic = True):
    from constant import elements, ele_num, num_ele
    elements = elements
    symbols = {}
    result = ''

    connections = smiles.count('*')
    conn_str = ''
    if connections == 1:
        conn_str = '*'
    elif connections >= 2:
        conn_str = '*'
        for i in range(connections - 1):
            conn_str += '(*)'

    is_aromatic = False
    i = 0
    for i in range(len(smiles)):
        s1 = smiles[i]
        if i < len(smiles) - 1:
            s2 = smiles[i+1]
        else:
            s2 = ''
        s_two = s1 + s2

        if s_two in elements:
            if s_two in symbols:
                symbols[s_two] += 1
            else:
                symbols[s_two] = 1
            i += 2
            continue

        if s1 in elements:
            if s1 in symbols:
                symbols[s1] += 1
            else:
                symbols[s1] = 1
            i += 1
            continue

        s_upper = s1.upper()
        if s_upper in elements:
            is_aromatic = True

            if s_upper in symbols:
                symbols[s_upper] += 1
            else:
                symbols[s_upper] = 1
            i += 1
            continue

        i += 1

    symbols_sorted = {k: symbols[k] for k in sorted(symbols.keys(), key=lambda x: ele_num[x])}

    for k, v in symbols_sorted.items():
        result += f'{k}{v}'

    return result + conn_str, is_aromatic

def short_rep_1(smiles, Aromatic = True):
    isring = 0
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        mol = Chem.MolFromSmarts(smiles)

    if mol is not None:
        try:
            ring_info = mol.GetRingInfo()
            isring = ring_info.NumRings()
        except:
            pass

    result, is_aromatic = short_rep_smiles(smiles)

    if isring > 0:
        result = f'{isring}ring_' + result

    if is_aromatic:
        result = f'aromatic_' + result

    return result

def short_rep(smiles, Aromatic = True):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        mol = Chem.MolFromSmarts(smiles)
        if mol is None:
            return short_rep_smiles(smiles, ring=False, Aromatic=Aromatic)

    ring_info = mol.GetRingInfo()
    isring = False
    if len(ring_info.AtomRings()) > 0 and ring_info.NumRings() > 0:
        # return short_ring(smiles, Aromatic)
        isring = True
        pass

    symbols = {}
    for atom in mol.GetAtoms():
        symbol = atom.GetAtomicNum()
        if symbol in symbols:
            symbols[symbol] += 1
        else:
            symbols[symbol] = 1

    result = ''
    for i in range(117):
        if i+1 in symbols:
            result += f'{periodictable.elements[i+1]}{symbols[i+1]}'

    connections = smiles.count('*')
    conn_str = ''
    if connections == 1:
        conn_str = '*'
    elif connections >= 2:
        conn_str = '*'
        for i in range(connections - 1):
            conn_str += '(*)'

    result += conn_str
    if isring:
        result = 'ring_' + result

    return result