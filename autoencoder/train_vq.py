"""Train/export fragment RVQ tokens: python -m autoencoder.train_vq --help."""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from autoencoder.fragment_vq import FragmentDataset, FragmentVQAutoencoder, read_fragments


def run_epoch(model, loader, device, optimizer=None):
    model.train(optimizer is not None)
    totals, count = {}, 0
    used = [set() for _ in model.quantizer.codebooks]
    with torch.set_grad_enabled(optimizer is not None):
        for batch in loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            pred = model(batch)
            losses = model.loss(pred, batch)
            if not torch.isfinite(losses['total']):
                raise RuntimeError('Non-finite loss')
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                losses['total'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            size = batch['atoms'].shape[0]
            count += size
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + value.item() * size
            for stage, indices in enumerate(pred['codes'].detach().cpu().T.tolist()):
                used[stage].update(indices)
    return {k: v / count for k, v in totals.items()}, [len(s) for s in used]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('mode', choices=['train', 'export'])
    parser.add_argument('--input', required=True, help='Fragment SMILES text or statistics JSON')
    parser.add_argument('--checkpoint', default='autoencoder/fragment_vq.pth')
    parser.add_argument('--output', default='autoencoder/fragment_tokens.json')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--max-nodes', type=int, default=70)
    parser.add_argument('--num-codes', type=int, default=4)
    parser.add_argument('--codebook-size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    if args.batch_size < 1 or args.epochs < 1:
        parser.error('batch-size and epochs must be positive')
    torch.manual_seed(args.seed)
    fragments = read_fragments(args.input)
    if args.mode == 'export':
        model = FragmentVQAutoencoder.load(args.checkpoint, args.device)
        rows = []
        for start in range(0, len(fragments), args.batch_size):
            rows.extend(model.encode_fragments(fragments[start:start + args.batch_size]))
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(dict(checkpoint=str(Path(args.checkpoint).resolve()),
                                                   config=model.config, fragments=rows), indent=2), encoding='utf-8')
        print(f'Exported {len(rows)} fragments to {args.output}')
        return
    dataset = FragmentDataset(fragments, args.max_nodes)
    if len(dataset) < 2:
        parser.error('Training needs at least two distinct canonical fragments')
    val_size = max(1, round(len(dataset) * 0.1))
    train_data, val_data = random_split(dataset, [len(dataset) - val_size, val_size],
                                      generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size)
    model = FragmentVQAutoencoder(max_nodes=args.max_nodes, num_codes=args.num_codes,
                                 codebook_size=args.codebook_size).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    best = float('inf')
    for epoch in range(args.epochs):
        train_loss, usage = run_epoch(model, train_loader, args.device, optimizer)
        val_loss, _ = run_epoch(model, val_loader, args.device)
        print(json.dumps(dict(epoch=epoch + 1, train=train_loss, validation=val_loss,
                              used_codes_per_stage=usage)))
        if val_loss['total'] < best:
            best = val_loss['total']
            model.save(args.checkpoint)


if __name__ == '__main__':
    main()
