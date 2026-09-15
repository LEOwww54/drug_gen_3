"""Fixed-checkpoint mode diagnostic; does not update model weights.

Run from repository root: python -m autoencoder.validate_vq_modes --help
The original list(set(...)) ordering/split was not saved. Samples here are
therefore a reproducible probe, NOT a reconstructed held-out validation set.
"""
import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from autoencoder.fragment_vq import FragmentDataset, FragmentVQAutoencoder, read_fragments


@torch.no_grad()
def measure(model, loader, device, training=False, seed=42):
    """Return per-sample length metrics and latents, restoring model mode/RNG."""
    previous = model.training
    rows = {k: [] for k in ('ce', 'error', 'correct', 'target', 'prediction', 'codes', 'z', 'continuous')}
    devices = [torch.device(device).index or 0] if str(device).startswith('cuda') else []
    captured = []
    hook = model.quantizer.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach()))
    try:
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            model.train(training)
            for batch in loader:
                batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                z, codes, _ = model.encode(batch)
                logits = model.length(z)
                target = batch['n_nodes']
                prediction = logits.argmax(-1)
                values = dict(ce=F.cross_entropy(logits, target, reduction='none'),
                              error=(prediction-target).abs(), correct=prediction.eq(target),
                              target=target, prediction=prediction, codes=codes, z=z,
                              continuous=captured.pop())
                for k, v in values.items():
                    rows[k].append(v.cpu())
    finally:
        hook.remove()
        model.train(previous)
    return {k: torch.cat(v) for k, v in rows.items()}


def summarize(rows):
    result = dict(length_ce=rows['ce'].mean().item(),
                  length_accuracy=rows['correct'].float().mean().item(),
                  length_mae=rows['error'].float().mean().item(),
                  unique_code_sequences=len(torch.unique(rows['codes'], dim=0)),
                  quantization_mse=(rows['z']-rows['continuous']).square().mean().item())
    result['by_length'] = {}
    for lo, hi in [(1, 10), (11, 20), (21, 30), (31, 50), (51, 70)]:
        mask = (rows['target'] >= lo) & (rows['target'] <= hi)
        if mask.any():
            result['by_length'][f'{lo}-{hi}'] = dict(count=int(mask.sum()),
                ce=rows['ce'][mask].mean().item(),
                accuracy=rows['correct'][mask].float().mean().item(),
                mae=rows['error'][mask].float().mean().item())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default='autoencoder/autoencoder/fragment_vq.pth')
    parser.add_argument('--inputs', nargs='+', default=['stru_data_ZINC_250K_pamf_train.json', 'stru_data_ZINC_250K_pamf_test.json'])
    parser.add_argument('--samples', type=int, default=2048)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output', default='autoencoder/vq_mode_diagnostic.json')
    args = parser.parse_args()
    if min(args.samples, args.batch_size, args.repeats) < 1:
        parser.error('samples, batch-size and repeats must be positive')
    model = FragmentVQAutoencoder.load(args.checkpoint, args.device)
    fragments = sorted(set(f for p in args.inputs for f in read_fragments(p)))
    order = torch.randperm(len(fragments), generator=torch.Generator().manual_seed(args.seed))
    chosen = [fragments[i] for i in order[:args.samples].tolist()]
    dataset = FragmentDataset(chosen, model.max_nodes,
                              atom_vocabulary=model.atom_vocabulary,
                              show_progress=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    baseline = measure(model, loader, args.device)
    report = dict(checkpoint=str(Path(args.checkpoint).resolve()), config=model.config,
                  sample_count=len(dataset), seed=args.seed,
                  limitation='Fixed probe from input pool; original train/validation membership and checkpoint epoch are unavailable.',
                  sample_smiles=[s['smiles'] for s in dataset.samples],
                  eval=summarize(baseline), train_mode=[])
    for repeat in range(args.repeats):
        rows = measure(model, loader, args.device, True, args.seed+repeat)
        metrics = summarize(rows)
        metrics.update(code_agreement_per_stage=rows['codes'].eq(baseline['codes']).float().mean(0).tolist(),
                       full_code_agreement=rows['codes'].eq(baseline['codes']).all(-1).float().mean().item(),
                       quantized_mode_mse=(rows['z']-baseline['z']).square().mean().item(),
                       continuous_mode_mse=(rows['continuous']-baseline['continuous']).square().mean().item())
        report['train_mode'].append(metrics)
    again = measure(model, loader, args.device)
    report['eval_repeat_max_logloss_difference'] = (again['ce']-baseline['ce']).abs().max().item()
    report['eval_repeat_codes_identical'] = torch.equal(again['codes'], baseline['codes'])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k != 'sample_smiles'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
