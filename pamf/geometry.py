"""Reproducible ETKDG/MMFF ensembles; no inferred protonation changes."""

import math
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms


def conformers(mol, count=10, seed=42, energy_window=10.0):
    hydrogenated = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.numThreads = 1
    params.pruneRmsThresh = 0.25
    ids = list(AllChem.EmbedMultipleConfs(hydrogenated, numConfs=count, params=params))
    if not ids:
        raise RuntimeError('ETKDG failed to generate a conformer')
    if AllChem.MMFFHasAllMoleculeParams(hydrogenated):
        results = AllChem.MMFFOptimizeMoleculeConfs(hydrogenated, numThreads=1, maxIters=1000)
        method = 'MMFF94'
    elif AllChem.UFFHasAllMoleculeParams(hydrogenated):
        results = AllChem.UFFOptimizeMoleculeConfs(hydrogenated, numThreads=1, maxIters=1000)
        method = 'UFF'
    else:
        raise RuntimeError('Neither MMFF nor UFF has complete parameters for this molecule')
    converged = [(cid, float(energy)) for cid, (status, energy) in zip(ids, results)
                 if status == 0 and math.isfinite(energy)]
    if not converged:
        raise RuntimeError('No converged force-field conformer; refusing unoptimized geometry')
    converged.sort(key=lambda row: (row[1], row[0]))
    kept = [(cid, e) for cid, e in converged if e <= converged[0][1] + energy_window]
    return hydrogenated, [cid for cid, _ in kept], dict(
        embedding='ETKDGv3', force_field=method, seed=seed, requested=count,
        embedded=len(ids), converged=len(converged), retained=len(kept),
        energy_window_kcal_mol=energy_window, conformer_energies_kcal_mol=kept)


def torsion_variability(mol, candidates, conformer_ids):
    result = {}
    for candidate in candidates:
        i, j = candidate['atom_i'], candidate['atom_j']
        left = sorted(a.GetIdx() for a in mol.GetAtomWithIdx(i).GetNeighbors()
                      if a.GetIdx() != j and a.GetAtomicNum() > 1)
        right = sorted(a.GetIdx() for a in mol.GetAtomWithIdx(j).GetNeighbors()
                       if a.GetIdx() != i and a.GetAtomicNum() > 1)
        value = None
        if left and right and len(conformer_ids) >= 2:
            angles = [rdMolTransforms.GetDihedralRad(mol.GetConformer(cid), left[0], i, j, right[0])
                      for cid in conformer_ids]
            if all(math.isfinite(a) for a in angles):
                value = max(0.0, min(1.0, 1.0 - abs(sum(complex(math.cos(a), math.sin(a))
                                                       for a in angles) / len(angles))))
        result[candidate['bond_idx']] = value
    return result
