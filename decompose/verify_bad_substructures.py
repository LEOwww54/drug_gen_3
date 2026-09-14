"""Trace bad statistics SMILES to PKL fragments and parent atom mappings."""
import json
import pickle
from functools import lru_cache
from pathlib import Path

from rdkit import Chem, rdBase
from tqdm import tqdm


def key(mol):
    mol = Chem.Mol(mol)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    mol.UpdatePropertyCache(strict=False)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


@lru_cache(None)
def variants(smiles):
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return []
    direct = key(mol)
    clean = Chem.RWMol(mol)
    for i in reversed([a.GetIdx() for a in clean.GetAtoms() if a.GetAtomicNum() == 0]):
        clean.RemoveAtom(i)
    parts = Chem.GetMolFrags(clean.GetMol(), asMols=True, sanitizeFrags=False)
    return [(direct, 'raw')] + [(key(part), 'dummy_removed') for part in parts]


def topology_query(smiles):
    # Deliberately do not kekulize an invalid standalone aromatic fragment.
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    query = Chem.RWMol()
    for atom in mol.GetAtoms():
        charge = atom.GetFormalCharge()
        query.AddAtom(Chem.AtomFromSmarts(
            f'[#{atom.GetAtomicNum()};{"a" if atom.GetIsAromatic() else "A"};{charge:+d}]'))
    for bond in mol.GetBonds():
        query.AddBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond.GetBondType())
    return query.GetMol()


def main():
    root = Path(__file__).resolve().parents[1]
    bad = json.loads((root / 'bad_sub.json').read_text(encoding='utf-8'))
    entries, lookup, queries = {}, {}, {}
    for smiles, count in bad.items():
        with rdBase.BlockLogs():
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            try:
                Chem.SanitizeMol(Chem.Mol(mol))
                error = None
            except Exception as exc:
                error = str(exc)
        entries[smiles] = dict(expected_count=count, standalone_error=error, occurrences=[])
        lookup.setdefault(key(mol), []).append(smiles)
        queries[smiles] = topology_query(smiles)
    source = root / 'gpt/frag_file/frag_decom_ZINC_250K_pamf_train.pkl'
    with source.open('rb') as handle:
        rows = pickle.load(handle)['mol']
    for index, row in tqdm(rows.items(), desc='Tracing bad fragments'):
        parent = None
        for fi, fragment in enumerate(row['fragment_smiles']):
            matches = {}
            for canonical, mode in variants(fragment):
                for bad_smiles in lookup.get(canonical, []):
                    matches.setdefault(bad_smiles, mode)
            for bad_smiles, mode in matches.items():
                if parent is None:
                    parent = Chem.MolFromSmiles(row['oring'])
                mapping = parent.GetSubstructMatch(queries[bad_smiles]) if parent is not None else ()
                entries[bad_smiles]['occurrences'].append(dict(
                    record_index=index, smiles=row['oring'], fragment_index=fi,
                    matched_fragment=fragment, fragment_smiles=row['fragment_smiles'],
                    match_mode=mode, is_parent_substructure=bool(mapping),
                    parent_atom_indices=list(mapping)))
    for entry in entries.values():
        entry['observed_count'] = len(entry['occurrences'])
        entry['count_matches'] = entry['observed_count'] == entry['expected_count']
    summary = dict(bad_fragment_types=len(entries),
                   found_types=sum(e['observed_count'] > 0 for e in entries.values()),
                   expected_occurrences=sum(bad.values()),
                   observed_occurrences=sum(e['observed_count'] for e in entries.values()),
                   count_mismatch_types=sum(not e['count_matches'] for e in entries.values()),
                   non_substructure_occurrences=sum(not o['is_parent_substructure']
                       for e in entries.values() for o in e['occurrences']),
                   raw_match_occurrences=sum(o['match_mode'] == 'raw'
                       for e in entries.values() for o in e['occurrences']))
    report = dict(summary=summary,
                  method='Canonical raw and dummy-deleted SMILES matching without sanitization; '
                         'parent substructure checks atom element, aromaticity, formal charge and bond type; '
                         'ignores hydrogen counts and stereochemistry. Atom indices are zero-based.',
                  fragments=entries)
    destination = root / 'bad_sub_verification.json'
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary))
    print(destination)


if __name__ == '__main__':
    main()
