"""python -m pamf --smiles ... ; batch input/output uses one record per line."""

import argparse
import json
from pathlib import Path
from .pipeline import PAMFConfig, decompose_smiles
from .scoring import fit_reference


def main():
    parser = argparse.ArgumentParser(description='PAMF: chemistry-constrained, physics-aware fragmentation')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--smiles')
    source.add_argument('--input', type=Path, help='UTF-8 text, one SMILES per nonempty line')
    source.add_argument('--fit-reference', type=Path, help='Fit statistics from training-set PAMF JSONL')
    parser.add_argument('--output', type=Path, help='JSON for single SMILES/reference, JSONL for batch')
    parser.add_argument('--mode', choices=['xtb', 'rules'], default='xtb')
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--min-size', type=int, default=3)
    parser.add_argument('--max-size', type=int, default=25)
    parser.add_argument('--strict-max-size', action='store_true')
    parser.add_argument('--conformers', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--timeout', type=float, default=300)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--unpaired', type=int)
    parser.add_argument('--beam-width', type=int, default=128)
    args = parser.parse_args()
    try:
        if args.output and any(p is not None and args.output.resolve() == p.resolve()
                               for p in (args.input, args.fit_reference, args.reference)):
            raise ValueError('Output must not overwrite an input/reference file')
        if args.fit_reference:
            with args.fit_reference.open(encoding='utf-8') as handle:
                result = fit_reference(json.loads(line) for line in handle if line.strip())
            text = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n'
        else:
            config = PAMFConfig(mode=args.mode,
                                min_heavy_atoms=args.min_size, max_heavy_atoms=args.max_size,
                                strict_max_size=args.strict_max_size, conformer_count=args.conformers,
                                random_seed=args.seed, xtb_timeout=args.timeout, xtb_threads=args.threads,
                                unpaired_electrons=args.unpaired, beam_width=args.beam_width)
            reference = json.loads(args.reference.read_text(encoding='utf-8')) if args.reference else None
            if args.smiles is not None:
                result = decompose_smiles(args.smiles, config, reference=reference)
                text = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n'
            else:
                # Fail fast and do not publish a partial batch as a successful result.
                rows = []
                with args.input.open(encoding='utf-8') as handle:
                    for line_number, line in enumerate(handle, 1):
                        if not line.strip():
                            continue
                        try:
                            result = decompose_smiles(line.strip(), config, reference=reference)
                        except Exception as exc:
                            raise ValueError(f'Input line {line_number}: {exc}') from exc
                        rows.append(json.dumps(result, ensure_ascii=False, allow_nan=False))
                text = '\n'.join(rows)+ ('\n' if rows else '')
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text, encoding='utf-8')
        else:
            print(text, end='')
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        parser.exit(2, f'PAMF error: {exc}\n')


if __name__ == '__main__':
    main()
