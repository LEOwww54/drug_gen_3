"""Ordered batch fragmentation with isolated spawn workers."""

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from copy import deepcopy
import multiprocessing
import os
from tqdm import tqdm

from .pipeline import PAMFConfig, fragment_smiles


def _fragment_job(smiles, index, config, reference):
    # Only serializable inputs cross process boundaries. RDKit molecules and
    # xTB scratch files are created and owned by the worker executing this job.
    try:
        return fragment_smiles(smiles, config, reference=reference)
    except Exception as exc:
        raise RuntimeError(f'Batch input[{index}] {smiles!r} failed: {exc}') from exc


def fragment_smiles_batch(smiles_list: list[str], config: PAMFConfig | None = None,
                          *, workers: int | None = None, return_format: str = 'list',
                          reference=None, on_error='skip', return_indices=False):
    """Fragment a list using independent processes; default output matches input order.

    Identical input strings are computed once per call. List output preserves
    duplicates as independent lists; dict output contains one key per exact
    original string, in first-occurrence order. No canonicalization of keys.
    workers=1 runs serially. Otherwise use a guarded __main__ entry point.
    Failed molecules are logged and omitted by default; on_error='raise' aborts.
    return_indices=True returns (list_output, original_zero_based_indices), so
    callers can filter labels/properties without guessing from SMILES strings.
    Native process crashes remain fatal (the shared process pool is broken).
    Progress counts attempted unique SMILES, including failures.
    """
    if not isinstance(smiles_list, list):
        raise TypeError('smiles_list must be a list of SMILES strings')
    inputs = list(smiles_list)
    for index, smiles in enumerate(inputs):
        if not isinstance(smiles, str):
            raise ValueError(f'Batch input[{index}] must be a SMILES string')
    if on_error not in ('skip', 'raise'):
        raise ValueError('on_error must be skip or raise')
    if return_indices and return_format != 'list':
        raise ValueError('return_indices requires return_format=list')
    if return_format not in ('list', 'dict'):
        raise ValueError('return_format must be list or dict')
    if workers is not None and (type(workers) is not int or workers < 1):
        raise ValueError('workers must be a positive integer or None')
    config = deepcopy(config) if config is not None else PAMFConfig()
    reference = deepcopy(reference)
    first_indices = {}
    for index, smiles in enumerate(inputs):
        first_indices.setdefault(smiles, index)
    if not inputs:
        if return_indices:
            return [], []
        return [] if return_format == 'list' else {}
    if workers is None:
        cpu_count = getattr(os, 'process_cpu_count', os.cpu_count)() or 1
        workers = max(1, cpu_count // config.xtb_threads)
        if os.name == 'nt':
            workers = min(workers, 61)
    workers = min(workers, len(first_indices))
    results = {}
    failures = 0

    def failed(smiles, exc):
        nonlocal failures
        if on_error == 'raise' or isinstance(exc, BrokenProcessPool):
            raise exc
        failures += 1
        tqdm.write(f'PAMF skipped input[{first_indices[smiles]}] {smiles!r}: {exc}')

    if workers == 1:
        for smiles, index in tqdm(first_indices.items(), total=len(first_indices),
                                  desc='PAMF decomposition', unit='mol'):
            try:
                results[smiles] = _fragment_job(smiles, index, config, reference)
            except Exception as exc:
                failed(smiles, exc)
    else:
        # Bound submitted work to avoid a Future and serialized config for
        # every molecule in large datasets. Only the parent assembles output.
        with ProcessPoolExecutor(max_workers=workers,
                                 mp_context=multiprocessing.get_context('spawn')) as pool, \
                tqdm(total=len(first_indices), desc='PAMF decomposition', unit='mol') as progress:
            jobs = iter(first_indices.items())
            pending = {}

            def submit_next():
                item = next(jobs, None)
                if item is not None:
                    smiles, index = item
                    pending[pool.submit(_fragment_job, smiles, index, config, reference)] = smiles

            try:
                for _ in range(2 * workers):
                    submit_next()
                while pending:
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        smiles = pending.pop(future)
                        try:
                            results[smiles] = future.result()
                        except Exception as exc:
                            failed(smiles, exc)
                        progress.update(1)
                        progress.set_postfix(failed=failures, refresh=False)
                    for _ in done:
                        submit_next()
            finally:
                for future in pending:
                    future.cancel()
    if return_format == 'dict':
        return {smiles: list(results[smiles]) for smiles in first_indices if smiles in results}
    indices = [i for i, smiles in enumerate(inputs) if smiles in results]
    output = [list(results[inputs[i]]) for i in indices]
    return (output, indices) if return_indices else output
