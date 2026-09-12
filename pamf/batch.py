"""Ordered batch fragmentation with isolated spawn workers."""

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from copy import deepcopy
import multiprocessing
import os

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
                          reference=None) -> list[list[str]] | dict[str, list[str]]:
    """Fragment a list using independent processes; default output matches input order.

    Identical input strings are computed once per call. List output preserves
    duplicates as independent lists; dict output contains one key per exact
    original string, in first-occurrence order. No canonicalization of keys.
    workers=1 runs serially. Otherwise use a guarded __main__ entry point.
    Failure raises RuntimeError with the first occurrence's zero-based index;
    no partial output is returned. Running jobs finish/timeout before cleanup.
    """
    if not isinstance(smiles_list, list):
        raise TypeError('smiles_list must be a list of SMILES strings')
    inputs = list(smiles_list)
    for index, smiles in enumerate(inputs):
        if not isinstance(smiles, str) or not smiles.strip():
            raise ValueError(f'Batch input[{index}] must be a nonempty SMILES string')
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
        return [] if return_format == 'list' else {}
    if workers is None:
        cpu_count = getattr(os, 'process_cpu_count', os.cpu_count)() or 1
        workers = max(1, cpu_count // config.xtb_threads)
        if os.name == 'nt':
            workers = min(workers, 61)
    workers = min(workers, len(first_indices))
    results = {}
    if workers == 1:
        for smiles, index in first_indices.items():
            results[smiles] = _fragment_job(smiles, index, config, reference)
    else:
        # Bound submitted work to avoid a Future and serialized config for
        # every molecule in large datasets. Only the parent assembles output.
        with ProcessPoolExecutor(max_workers=workers,
                                 mp_context=multiprocessing.get_context('spawn')) as pool:
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
                        results[smiles] = future.result()
                    for _ in done:
                        submit_next()
            finally:
                for future in pending:
                    future.cancel()
    if return_format == 'dict':
        return {smiles: list(results[smiles]) for smiles in first_indices}
    return [list(results[smiles]) for smiles in inputs]
