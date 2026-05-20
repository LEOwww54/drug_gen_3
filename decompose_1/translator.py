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

            if idx in queue:
                pairs.append(sorted((atomid, idx)))

                if idx in ring_pairs:
                    ring_pairs[idx].add(ring_count)
                else:
                    ring_pairs[idx] = {ring_count}

                if atomid in ring_pairs:
                    ring_pairs[atomid].add(ring_count)
                else:
                    ring_pairs[atomid] = {ring_count}

                if not atomid in edge:
                    edge[atomid] = {idx : (atomid, idx, ring_count, None)}
                else:
                    edge[atomid][idx] = (atomid, idx, ring_count, None)
                ring_count += 1
                continue

            queue.append(idx)

            if not sorted((atomid, idx)) in pairs:
                pairs.append((atomid, idx))
                bond = mol.GetBondBetweenAtoms(atomid, idx).GetBondType()
                if not atomid in edge:
                    edge[atomid] = {idx : (atomid, idx, 0, bond)}
                else:
                    edge[atomid][idx]  = (atomid, idx, 0, bond)

        visited.append(atomid)

    return  edge, ring_pairs

def getR(connections : dict[int, dict[int, tuple]], idx : int, text : dict[int, list], ring_pairs : dict[int, set]):
    s = text[idx]
    l = len(connections)

    if idx in ring_pairs:
        for i in ring_pairs[idx]:
            if i < 10:
                s.append(f'<r{i}>')
            else:
                s.append(f'<r%{i}>')

    count = 0

    if idx in connections:
        for i, content in connections[idx].items():
            bond = content[3]
            ring_count = content[2]
            st = []

            if bond is not None:
                if (l > 1) and count < l - 1:
                    st.append('(')
                st += bond_type_to_str(bond)
                st += getR(connections, i, text, ring_pairs)
                if (l > 1) and count < l - 1:
                    st .append(')')

                s.extend(st)
            count += 1
    return s

def bond_type_to_str(bond_type) -> str:
    """将键类型转换为字符串"""
    from rdkit.Chem.rdchem import BondType

    mapping = {
        BondType.SINGLE: "",
        BondType.DOUBLE: "=",
        BondType.TRIPLE: "#",
        BondType.AROMATIC: ":",
    }
    return mapping.get(bond_type, "-")

def smiles_test(smiles = 'C1CCC2C[1*]CC12'):
    mol = Chem.MolFromSmiles(smiles)
    Chem.Kekulize(mol, True)
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
    x = smiles_test('[H][C@@]12C[C@H](O)[C@@]3(C)C(=O)[C@H](OC(C)=O)C4=C(C)[C@H](C[C@@](O)([C@@H](OC(=O)c5cc([1*])ccc5)[C@]3([H])[C@@]1(CO2)OC(C)=O)C4(C)C)OC(=O)[C@H](O)[C@@H](NC(=O)c6ccccc6)c7ccccc7')
    mol = Chem.MolFromSmiles('C1CC4CC(C(O)CC(N)CC4)C12CCC3CC2CC(C1CCCC1)5CCCC35')
