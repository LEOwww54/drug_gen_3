"""Longest-tree-path SISR serialization with minimal brackets for that tree.

Acyclic components use their exact diameter. Cyclic components compare DFS
spanning trees: their best diameter is not a proof of the original graph's
longest simple path. Every non-tree bond becomes a paired ring label.
"""
from collections import deque
from rdkit import Chem


def _diameter(adjacency):
    """Return endpoints and atom count of a tree's longest path in O(V)."""
    def farthest(start):
        distance = {start: 0}
        queue = deque([start])
        while queue:
            atom = queue.popleft()
            for neighbor in adjacency[atom]:
                if neighbor not in distance:
                    distance[neighbor] = distance[atom] + 1
                    queue.append(neighbor)
        end = max(distance, key=lambda i: (distance[i], -i))
        return end, distance[end]
    first, _ = farthest(min(adjacency))
    last, length = farthest(first)
    return first, last, length + 1


def _spanning_tree(component, neighbors):
    adjacency = {i: neighbors[i] for i in component}
    if sum(map(len, adjacency.values())) == 2 * (len(component) - 1):
        return adjacency
    best, best_score = None, None
    # At most two degree-one atoms can belong to any simple path.
    mandatory_leaves = sum(len(neighbors[i]) == 1 for i in component)
    path_bound = len(component) - max(0, mandatory_leaves - 2)
    for descending in (False, True):
        ordered = {i: sorted(neighbors[i], key=lambda j:
                             ((-1 if descending else 1) * len(neighbors[j]), j))
                   for i in component}
        for start in sorted(component, key=lambda i: (len(neighbors[i]), i)):
            candidate = {i: [] for i in component}
            visited = {start}
            stack = [(start, iter(ordered[start]))]
            while stack:
                atom, iterator = stack[-1]
                neighbor = next(iterator, None)
                if neighbor is None:
                    stack.pop()
                elif neighbor not in visited:
                    visited.add(neighbor)
                    candidate[atom].append(neighbor)
                    candidate[neighbor].append(atom)
                    stack.append((neighbor, iter(ordered[neighbor])))
            _, _, length = _diameter(candidate)
            leaves = sum(len(v) == 1 for v in candidate.values())
            score = (length, -leaves)
            if best_score is None or score > best_score:
                best, best_score = candidate, score
            if length == path_bound and leaves == max(2, mandatory_leaves):
                return best
    return best


def tree(mol: Chem.Mol):
    """Return directed tree edges and ring endpoints for every component.

    Roots are diameter endpoints, not necessarily atom 0. Every atom is a key
    in edges, including leaves; roots are keys without incoming tree edges.
    Ring bonds live only in ring_pairs, never among the child edges.
    """
    neighbors = {a.GetIdx(): sorted(n.GetIdx() for n in a.GetNeighbors())
                 for a in mol.GetAtoms()}
    bonds = {tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))): b.GetBondType()
             for b in mol.GetBonds()}
    edges, ring_pairs, tree_bonds = {}, {}, set()
    for component in Chem.GetMolFrags(mol):
        adjacency = _spanning_tree(component, neighbors)
        first, last, _ = _diameter(adjacency)
        root = min(first, last)
        stack = [(root, None)]
        while stack:
            atom, parent = stack.pop()
            children = sorted(i for i in adjacency[atom] if i != parent)
            edges[atom] = {}
            for neighbor in children:
                pair = tuple(sorted((atom, neighbor)))
                tree_bonds.add(pair)
                edges[atom][neighbor] = (atom, neighbor, 0, bonds[pair])
            stack.extend((i, atom) for i in reversed(children))
    for number, pair in enumerate(sorted(bonds.keys() - tree_bonds), 1):
        for atom in pair:
            ring_pairs.setdefault(atom, set()).add((number, bonds[pair]))
    return edges, ring_pairs


def getR(connections: dict, idx: int, text: dict, ring_pairs: dict):
    """Emit side branches first, then the longest child path without brackets.

    Iterative traversal never mutates atom payloads. A node with k children
    needs exactly max(0, k-1) pairs of parentheses.
    """
    children, order, seen = {}, [], set()
    pending = [idx]
    while pending:
        atom = pending.pop()
        if atom in seen:
            raise ValueError('Child edges must form a tree; use ring_pairs for cycles')
        seen.add(atom)
        order.append(atom)
        children[atom] = [(i, content) for i, content in connections.get(atom, {}).items()
                          if content[2] == 0 and content[3] is not None]
        pending.extend(i for i, _ in children[atom])
    heights, sizes, main_child = {}, {}, {}
    for atom in reversed(order):
        descendants = children[atom]
        heights[atom] = 1 + max((heights[i] for i, _ in descendants), default=0)
        sizes[atom] = 1 + sum(sizes[i] for i, _ in descendants)
        if descendants:
            main_child[atom] = max(descendants, key=lambda item:
                                   (heights[item[0]], sizes[item[0]], -item[0]))[0]
    tokens = []
    pending = [('atom', idx)]
    while pending:
        item = pending.pop()
        if isinstance(item, str):
            tokens.append(item)
            continue
        _, atom = item
        tokens.extend(text[atom])
        for number, bond in sorted(ring_pairs.get(atom, ()), key=lambda pair: pair[0]):
            label = f'<r{number}>' if number < 10 else f'<r%{number}>'
            tokens.extend([bond_type_to_str(bond), label])
        descendants = children[atom]
        if not descendants:
            continue
        actions = []
        main = main_child[atom]
        for neighbor, content in descendants:
            if neighbor != main:
                actions.extend(['(', bond_type_to_str(content[3]), ('atom', neighbor), ')'])
        content = next(content for neighbor, content in descendants if neighbor == main)
        actions.extend([bond_type_to_str(content[3]), ('atom', main)])
        pending.extend(reversed(actions))
    return tokens


def bond_type_to_str(bond_type) -> str:
    mapping = {Chem.BondType.SINGLE: '-', Chem.BondType.DOUBLE: '=',
               Chem.BondType.TRIPLE: '#', Chem.BondType.AROMATIC: ':'}
    return mapping.get(bond_type, '-')


def smiles2token(mol, text):
    mol = Chem.Mol(mol)
    Chem.Kekulize(mol, True)
    connections, ring_pairs = tree(mol)
    child_atoms = {i for children in connections.values() for i in children}
    tokens = []
    for root in connections:
        if root not in child_atoms:
            if tokens:
                tokens.append('.')
            tokens.extend(getR(connections, root, text, ring_pairs))
    return tokens


def smiles_test(smiles='C1CCC2C[1*]CC12'):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f'Invalid SMILES: {smiles!r}')
    return smiles2token(mol, {a.GetIdx(): [a.GetSymbol()] for a in mol.GetAtoms()})
