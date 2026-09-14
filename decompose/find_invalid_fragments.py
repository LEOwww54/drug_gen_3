"""Audit a trusted local decomposition PKL without modifying it."""
import argparse
import json
import pickle
from functools import lru_cache
from pathlib import Path

from rdkit import Chem, rdBase
from tqdm import tqdm


@lru_cache(maxsize=None)
def fragment_error(smiles):
    if not smiles.strip():
        return 'Empty SMILES'
    with rdBase.BlockLogs():
        try:
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            if mol is None:
                return 'SMILES parse failed'
            if mol.GetNumAtoms() == 0:
                return 'No atoms'
            Chem.SanitizeMol(mol)
        except Exception as exc:
            return f'{type(exc).__name__}: {exc}'
    return None


def audit(input_path, output_path):
    input_path, output_path = Path(input_path), Path(output_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError('Output must not overwrite the input PKL')
    with input_path.open('rb') as handle:
        data = pickle.load(handle)
    records = data['mol']
    items = records.items() if isinstance(records, dict) else enumerate(records)
    failures = []
    checked_fragments = invalid_fragments = 0
    for index, row in tqdm(items, total=len(records), desc='Checking fragments'):
        fragments = row.get('fragment_smiles')
        errors = []
        if not isinstance(fragments, list) or not fragments:
            errors.append(dict(fragment_index=None, error='Missing, empty or non-list fragment_smiles'))
        else:
            for fragment_index, smiles in enumerate(fragments):
                checked_fragments += 1
                error = fragment_error(smiles) if isinstance(smiles, str) else 'Fragment is not a string'
                if error:
                    invalid_fragments += 1
                    errors.append(dict(fragment_index=fragment_index, smiles=smiles, error=error))
        if errors:
            failures.append(dict(record_index=index, smiles=row.get('oring'),
                                 fragment_smiles=fragments, errors=errors))
    report = dict(source=str(input_path.resolve()),
                  criterion='RDKit SMILES parsing and sanitization; dummy atoms (*) are allowed',
                  total_molecules=len(records), checked_fragments=checked_fragments,
                  invalid_molecules=len(failures), invalid_fragments=invalid_fragments,
                  molecules=failures)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != 'molecules'}, ensure_ascii=False))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('input')
    parser.add_argument('output')
    args = parser.parse_args()
    audit(args.input, args.output)
