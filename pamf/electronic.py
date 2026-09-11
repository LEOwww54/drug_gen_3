"""Isolated GFN2-xTB singlepoints and validated, atom-indexed output parsing."""

import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from rdkit import Chem


def parse_wbo(text, atom_count):
    """xTB sparse `wbo` file: 1-based atom i, atom j, Wiberg order."""
    values = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 3:
            raise ValueError(f'Malformed xTB WBO row: {line!r}')
        i, j = int(fields[0])-1, int(fields[1])-1
        value = float(fields[2].replace('D', 'E').replace('d', 'e'))
        if not (0 <= i < atom_count and 0 <= j < atom_count) or i == j or not math.isfinite(value):
            raise ValueError('Invalid xTB WBO atom index/value')
        key = tuple(sorted((i, j)))
        if key in values and abs(values[key]-value) > 1e-6:
            raise ValueError('Inconsistent duplicate xTB WBO pair')
        values[key] = value
    return values


def parse_charges(text, atom_count):
    charges = [float(v.replace('D', 'E').replace('d', 'e')) for v in text.split()]
    if len(charges) != atom_count or not all(math.isfinite(v) for v in charges):
        raise ValueError('xTB charges must contain one finite value per XYZ atom')
    return charges


def parse_polarizabilities(stdout, atom_count):
    """Optional GFN2 atomic table: # Z symbol covCN q C6AA alpha(0)."""
    values = {}
    active = False
    for line in stdout.splitlines():
        if 'covCN' in line and 'C6AA' in line:
            active = True
            continue
        fields = line.split()
        if active and len(fields) == 7 and fields[0].isdigit() and fields[1].isdigit():
            try:
                value = float(fields[-1])
                idx = int(fields[0])-1
                if 0 <= idx < atom_count and math.isfinite(value):
                    values[idx] = value
            except ValueError:
                pass
        elif active and values:
            break
    return [values.get(i) for i in range(atom_count)]


def run_xtb(mol_h, conf_id, executable='xtb', timeout=300, threads=1, unpaired=None):
    resolved = shutil.which(executable)
    if resolved is None:
        raise FileNotFoundError(f'xTB executable not found: {executable}. Install xTB or explicitly use mode="rules".')
    charge = sum(a.GetFormalCharge() for a in mol_h.GetAtoms())
    radicals = sum(a.GetNumRadicalElectrons() for a in mol_h.GetAtoms())
    uhf = radicals if unpaired is None else unpaired
    electrons = sum(a.GetAtomicNum() for a in mol_h.GetAtoms()) - charge
    if uhf < 0 or uhf > electrons or (electrons-uhf) % 2:
        raise ValueError('Unpaired electron count is incompatible with total electron count')
    command = [str(Path(resolved).resolve()), 'molecule.xyz', '--gfn', '2', '--sp', '--wbo',
               '--chrg', str(charge), '--uhf', str(uhf), '--parallel', str(threads)]
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads))
    with tempfile.TemporaryDirectory(prefix='pamf-xtb-') as folder:
        work = Path(folder)
        (work / 'molecule.xyz').write_text(Chem.MolToXYZBlock(mol_h, confId=conf_id), encoding='utf-8')
        try:
            process = subprocess.run(command, cwd=work, env=env, capture_output=True, text=True,
                                     errors='replace', timeout=timeout, check=False,
                                     creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f'xTB timed out after {timeout} seconds') from exc
        if process.returncode != 0 or (work / '.sccnotconverged').exists():
            raise RuntimeError(f'xTB failed (exit {process.returncode}):\n{process.stderr[-2000:]}\n{process.stdout[-2000:]}')
        if not (work / 'wbo').is_file() or not (work / 'charges').is_file():
            raise RuntimeError('xTB did not write required wbo/charges output files')
        n = mol_h.GetNumAtoms()
        wbo = parse_wbo((work / 'wbo').read_text(), n)
        charges = parse_charges((work / 'charges').read_text(), n)
        if abs(sum(charges)-charge) > 0.05:
            raise ValueError('xTB charges do not sum to the input formal charge')
        return dict(wbo=wbo, charges=charges,
                    polarizabilities=parse_polarizabilities(process.stdout, n),
                    metadata=dict(method='GFN2-xTB', calculation='singlepoint', command=command,
                                  charge=charge, unpaired_electrons=uhf, conformer_id=conf_id,
                                  xyz_atomic_numbers=[a.GetAtomicNum() for a in mol_h.GetAtoms()],
                                  version_lines=[s.strip() for s in process.stdout.splitlines()
                                                 if 'xtb version' in s.lower()],
                                  wbo_representation='sparse; unreported nonbonded pairs treated as zero'))


def neighborhood(mol, root, blocked_bond, radius):
    seen, frontier = {root}, {root}
    for _ in range(radius):
        new = set()
        for i in frontier:
            for bond in mol.GetAtomWithIdx(i).GetBonds():
                other = bond.GetOtherAtomIdx(i)
                if bond.GetIdx() != blocked_bond and mol.GetAtomWithIdx(other).GetAtomicNum() > 1:
                    new.add(other)
        frontier = new - seen
        seen.update(frontier)
    return seen


def add_electronic_features(mol, candidates, data, radius=2):
    for row in candidates:
        i, j, idx = row['atom_i'], row['atom_j'], row['bond_idx']
        pair = tuple(sorted((i, j)))
        if pair not in data['wbo']:
            raise ValueError(f'xTB WBO missing for candidate bond {idx}; refusing a zero bond-order assumption')
        left = neighborhood(mol, i, idx, radius)
        right = neighborhood(mol, j, idx, radius)
        if left & right:
            raise ValueError('Candidate bond does not separate its local neighborhoods')
        cross = sum(data['wbo'].get(tuple(sorted((a, b))), 0.0) for a in left for b in right)
        qi, qj = data['charges'][i], data['charges'][j]
        row.update(wbo=data['wbo'][pair], cross_wbo=cross,
                   charge_i=qi, charge_j=qj, charge_difference=abs(qi-qj),
                   polarizability_i=data['polarizabilities'][i],
                   polarizability_j=data['polarizabilities'][j])
