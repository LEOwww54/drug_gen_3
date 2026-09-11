"""Conservative candidate boundaries and loss-checked attachment labels."""

from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import BRICS


# Protect query *bonds*, not all bonds incident to a matched atom. Substituents
# outside a protected functional unit remain eligible as boundary proposals.
PROTECTED_GROUPS = {
    'amide_urea_carbamate': '[C](=[O,S])-[N]',
    'ester_carboxylate': '[C](=[O,S])-[O,S]',
    'sulfonamide_sulfonate': '[S](=[O])(=[O])-[N,O]',
    'amidine_guanidine': '[N]-[C]=[N]',
    'nitro': '[O]=[N+](-[O-])',
    'enone': '[C]=[C]-[C]=[O]',
    'azo': '[N]=[N]',
    'nitrile': '[C]#[N]',
}


def parse_smiles(smiles):
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError('Expected a non-empty SMILES string')
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None or not mol.GetNumAtoms():
        raise ValueError(f'Invalid SMILES: {smiles!r}')
    if any(a.GetAtomicNum() == 0 for a in mol.GetAtoms()):
        raise ValueError('Input must be a complete molecule without dummy atoms')
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    return mol


def bond_class(bond):
    return '-'.join(sorted(
        f'{a.GetSymbol()}({a.GetHybridization()})'
        for a in (bond.GetBeginAtom(), bond.GetEndAtom())))


def analyze_bonds(mol, extra_protected_smarts=()):
    protected = defaultdict(set)
    groups = []
    patterns = dict(PROTECTED_GROUPS)
    patterns.update({f'custom_{i}': s for i, s in enumerate(extra_protected_smarts)})
    for name, smarts in patterns.items():
        query = Chem.MolFromSmarts(smarts)
        if query is None:
            raise ValueError(f'Invalid protected SMARTS: {smarts}')
        for match in mol.GetSubstructMatches(query):
            groups.append(set(match))
            for b in query.GetBonds():
                idx = mol.GetBondBetweenAtoms(match[b.GetBeginAtomIdx()],
                                             match[b.GetEndAtomIdx()]).GetIdx()
                protected[idx].add(name)

    brics = {frozenset(pair) for pair, _ in BRICS.FindBRICSBonds(mol)}
    stereo_atoms = set()
    for a in mol.GetAtoms():
        if a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED:
            stereo_atoms.add(a.GetIdx())
    for b in mol.GetBonds():
        if b.GetStereo() != Chem.BondStereo.STEREONONE:
            stereo_atoms.update([b.GetBeginAtomIdx(), b.GetEndAtomIdx(), *b.GetStereoAtoms()])

    candidates = []
    for bond in mol.GetBonds():
        idx = bond.GetIdx()
        a, b = bond.GetBeginAtom(), bond.GetEndAtom()
        i, j = a.GetIdx(), b.GetIdx()
        reasons = protected[idx]
        if bond.GetBondType() != Chem.BondType.SINGLE:
            reasons.add('not_single')
        if bond.IsInRing():
            reasons.add('ring_internal')
        if bond.GetIsAromatic():
            reasons.add('aromatic')
        if bond.GetIsConjugated():
            reasons.add('conjugated')
        if min(a.GetAtomicNum(), b.GetAtomicNum()) <= 1:
            reasons.add('hydrogen_bond')
        if i in stereo_atoms or j in stereo_atoms:
            reasons.add('stereochemistry')
        if ((a.GetIsAromatic() and b.GetAtomicNum() in (7, 8, 16)) or
                (b.GetIsAromatic() and a.GetAtomicNum() in (7, 8, 16))):
            reasons.add('aromatic_heteroatom_resonance')
        if reasons:
            continue
        sources = ['acyclic_single']
        if frozenset((i, j)) in brics:
            sources.append('BRICS_boundary')
        if a.IsInRing() != b.IsInRing():
            sources.append('ring_boundary')
        if any((i in group) != (j in group) for group in groups):
            sources.append('functional_boundary')
        rotatable = (sum(n.GetAtomicNum() > 1 for n in a.GetNeighbors()) > 1 and
                     sum(n.GetAtomicNum() > 1 for n in b.GetNeighbors()) > 1)
        candidates.append(dict(bond_idx=idx, atom_i=i, atom_j=j,
                               bond_type='SINGLE', chemical_class=bond_class(bond),
                               candidate_source=sources, rotatable=rotatable,
                               ring=False, conjugated=False))
    return candidates, {str(k): sorted(v) for k, v in protected.items() if v}


def components(mol, cuts):
    blocked = set(cuts)
    remaining = set(range(mol.GetNumAtoms()))
    result = []
    while remaining:
        root = min(remaining)
        seen, stack = {root}, [root]
        remaining.remove(root)
        while stack:
            current = stack.pop()
            for b in mol.GetAtomWithIdx(current).GetBonds():
                other = b.GetOtherAtomIdx(current)
                if b.GetIdx() not in blocked and other in remaining:
                    remaining.remove(other)
                    seen.add(other)
                    stack.append(other)
        result.append(seen)
    return result


def reassemble(fragments):
    """Rejoin isotope-labelled single-bond stubs; reject malformed labels."""
    if not fragments or isinstance(fragments, str):
        raise ValueError('Expected a non-empty list of fragment SMILES')
    mols, labels = [], defaultdict(list)
    for smiles in fragments:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or not mol.GetNumAtoms():
            raise ValueError(f'Invalid fragment: {smiles!r}')
        for a in mol.GetAtoms():
            if a.GetAtomicNum() == 0:
                if (a.GetIsotope() < 1 or a.GetDegree() != 1 or
                        a.GetNeighbors()[0].GetAtomicNum() == 0 or
                        a.GetBonds()[0].GetBondType() != Chem.BondType.SINGLE):
                    raise ValueError('Attachment must be a numbered, terminal single-bond dummy')
                labels[a.GetIsotope()].append(a.GetIdx())
        mols.append(mol)
    if any(len(v) != 2 for v in labels.values()):
        raise ValueError('Each connection label must occur exactly twice')
    merged = mols[0]
    for mol in mols[1:]:
        merged = Chem.CombineMols(merged, mol)
    if labels:
        params = Chem.MolzipParams()
        params.label = Chem.MolzipLabel.Isotope
        merged = Chem.molzip(merged, params)
    Chem.SanitizeMol(merged)
    if any(a.GetAtomicNum() == 0 for a in merged.GetAtoms()):
        raise ValueError('Unresolved attachment after reassembly')
    return Chem.MolToSmiles(merged, isomericSmiles=True)


def make_fragments(mol, cuts):
    cuts = sorted(cuts)
    connections = {}
    for label, idx in enumerate(cuts, 1):
        b = mol.GetBondWithIdx(idx)
        connections[str(label)] = dict(bond_idx=idx, atom_i=b.GetBeginAtomIdx(),
                                       atom_j=b.GetEndAtomIdx(), bond_type=str(b.GetBondType()))
    fragmented = (Chem.FragmentOnBonds(mol, cuts, dummyLabels=[(i, i) for i in range(1, len(cuts)+1)])
                  if cuts else Chem.Mol(mol))
    fragments = sorted(Chem.MolToSmiles(f, isomericSmiles=True)
                       for f in Chem.GetMolFrags(fragmented, asMols=True))
    rebuilt = reassemble(fragments)
    if rebuilt != Chem.MolToSmiles(mol, isomericSmiles=True):
        raise ValueError('Fragment serialization/reassembly changed molecular identity or stereochemistry')
    return fragments, connections, rebuilt
