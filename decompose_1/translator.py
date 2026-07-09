from rdkit import Chem

def tree(mol : Chem.Mol):
    visited = []
    queue = []
    ring_count = 1
    pairs = []
    edge = {}
    ring_pairs = {}

    queue.append(0)
    while True:
        if len(queue) <= 0:
            break

        atomid = queue.pop()
        atom = mol.GetAtomWithIdx(atomid)

        neighbors = atom.GetNeighbors()


        for neighbor in neighbors:
            idx = neighbor.GetIdx()
            if idx in visited:
                continue
            bond = mol.GetBondBetweenAtoms(atomid, idx).GetBondType()

            if idx in queue:
                pairs.append(sorted((atomid, idx)))

                if idx in ring_pairs:
                    ring_pairs[idx].add((ring_count, bond))
                else:
                    ring_pairs[idx] = {(ring_count, bond)}

                if atomid in ring_pairs:
                    ring_pairs[atomid].add((ring_count, bond))
                else:
                    ring_pairs[atomid] = {(ring_count, bond)}

                if not atomid in edge:
                    edge[atomid] = {idx : (atomid, idx, ring_count, bond)}
                else:
                    edge[atomid][idx] = (atomid, idx, ring_count, bond)
                ring_count += 1
                continue

            queue.append(idx)

            if not sorted((atomid, idx)) in pairs:
                pairs.append((atomid, idx))
                if not atomid in edge:
                    edge[atomid] = {idx : (atomid, idx, 0, bond)}
                else:
                    edge[atomid][idx]  = (atomid, idx, 0, bond)

        visited.append(atomid)

    return  edge, ring_pairs

def getR(connections : dict[int, dict[int, tuple]], idx : int, text : dict[int, list], ring_pairs : dict[int, set]):
    s = text[idx]
    if idx in connections:
        l = len(connections[idx])
    else:
        l = 0

    count = 0

    if idx in ring_pairs:
        for i in ring_pairs[idx]:
            bond = bond_type_to_str(i[1])
            if i[0] < 10:
                s.append(f'{bond} <r{i[0]}>')
            else:
                s.append(f'{bond} <r%{i[0]}>')

    if idx in connections:
        for i, content in connections[idx].items():
            bond = content[3]
            ring_count = content[2]
            st = []

            if bond is not None:
                if (l > 1):
                    st.append('(')
                if ring_count > 0:
                    pass
                else:
                    st += bond_type_to_str(bond)
                    st += getR(connections, i, text, ring_pairs)
                if (l > 1):
                    st .append(')')

                s.extend(st)
            count += 1

    return s

def bond_type_to_str(bond_type) -> str:
    """将键类型转换为字符串"""
    from rdkit.Chem.rdchem import BondType

    mapping = {
        BondType.SINGLE: "-",
        BondType.DOUBLE: "=",
        BondType.TRIPLE: "#",
        BondType.AROMATIC: ":",
    }
    return mapping.get(bond_type, "-")

def smiles_test(smiles = 'C1CCC2C[1*]CC12'):
    mol = Chem.MolFromSmiles(smiles)
    Chem.Kekulize(mol,True)
    text = {}
    for atom in mol.GetAtoms():
        text[atom.GetIdx()] = [f"{atom.GetSymbol()}"]

    connections, ring_pairs = tree(mol)

    return getR(connections, 0, text, ring_pairs)

def smiles2token(mol, text):
    Chem.Kekulize(mol, True)
    connections, ring_pairs = tree(mol)

    return getR(connections, 0, text, ring_pairs)

if __name__ == '__main__':
    e = smiles_test('C=1=C=C=C=C=1')
    x = smiles_test('[*]C1=CC([*])=C2C(C([*])=CC2=C1)=[*]')
    mol = Chem.MolFromSmiles('C1CC4CC(C(O)CC(N)CC4)C12CCC3CC2CC(C1CCCC1)5CCCC35')
