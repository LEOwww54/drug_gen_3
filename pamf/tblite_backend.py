"""In-process GFN2-xTB, with calculator state owned by one invocation."""

from importlib.metadata import version
import math
from threading import RLock

# threadpoolctl changes native thread settings process-wide. Serialize callers
# within a process; multiprocessing still runs independent calculations.
_native_lock = RLock()
BOHR_IN_ANGSTROM = 0.529177210903


def run_tblite(mol_h, conf_id, threads=1, unpaired=None, max_iterations=250):
    try:
        import numpy as np
        from tblite.interface import Calculator
        from threadpoolctl import threadpool_limits
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            'tblite backend is unavailable. Install tblite-python and threadpoolctl '
            'from conda-forge, or choose PAMFConfig(xtb_backend="cli"). '
            f'Import error: {exc}') from exc
    charge = sum(a.GetFormalCharge() for a in mol_h.GetAtoms())
    uhf = sum(a.GetNumRadicalElectrons() for a in mol_h.GetAtoms()) if unpaired is None else unpaired
    numbers = np.array([a.GetAtomicNum() for a in mol_h.GetAtoms()], dtype=np.int32)
    electrons = int(numbers.sum()) - charge
    if type(uhf) is not int or uhf < 0 or uhf > electrons or (electrons-uhf) % 2:
        raise ValueError('Unpaired electron count is incompatible with total electron count')
    if type(threads) is not int or threads < 1 or type(max_iterations) is not int or max_iterations < 1:
        raise ValueError('threads and max_iterations must be positive integers')
    positions = np.array(mol_h.GetConformer(conf_id).GetPositions(), dtype=float) / BOHR_IN_ANGSTROM
    if not np.isfinite(positions).all():
        raise ValueError('Nonfinite input coordinates')
    with _native_lock:
        # Import above loads the native libraries before threadpoolctl scans them.
        with threadpool_limits(limits=threads):
            calc = Calculator('GFN2-xTB', numbers, positions, charge=charge, uhf=uhf)
            calc.set('verbosity', 0)
            calc.set('max-iter', max_iterations)
            try:
                result = calc.singlepoint()
                charges = np.array(result.get('charges'), dtype=float, copy=True)
                orders = np.array(result.get('bond-orders'), dtype=float, copy=True)
                energy = float(result.get('energy'))
            except RuntimeError as exc:
                raise RuntimeError(f'tblite GFN2-xTB singlepoint failed: {exc}') from exc
    n = len(numbers)
    package_version = version('tblite')
    # tblite 0.7 retains a singleton spin channel for restricted GFN2-xTB.
    if orders.shape == (n, n, 1):
        orders = orders[:, :, 0]
    elif orders.shape == (n, n, 2) and package_version == '0.7.0':
        # 0.7.0's post-processing dictionary wraps Fortran data as C-order.
        # Channels occupy contiguous nat*nat blocks: total then magnetization
        # (tblite wavefunction/mulliken.f90 calls updown_to_magnet).
        orders = orders.reshape(2, n, n)[0]
    if charges.shape != (n,) or not np.isfinite(charges).all() or abs(float(charges.sum())-charge) > 0.05:
        raise ValueError('Invalid tblite charges or total charge mismatch')
    if orders.shape != (n, n) or not np.isfinite(orders).all() or not np.allclose(orders, orders.T, atol=1e-8):
        raise ValueError('Invalid tblite bond-order matrix')
    if not math.isfinite(energy):
        raise ValueError('Nonfinite tblite energy')
    return dict(
        wbo={(i, j): float(orders[i, j]) for i in range(n) for j in range(i+1, n)},
        charges=charges.tolist(), polarizabilities=[None]*n,
        metadata=dict(method='GFN2-xTB', backend='tblite', calculation='singlepoint',
                      version=package_version, charge=charge, unpaired_electrons=uhf,
                      conformer_id=conf_id, xyz_atomic_numbers=numbers.tolist(),
                      coordinate_unit='bohr', energy_hartree=energy,
                      threads=threads, max_iterations=max_iterations,
                      timeout_policy='SCC iteration limit; no wall-clock timeout',
                      wbo_representation='dense; all off-diagonal pairs, no cutoff',
                      polarizabilities_available=False))
