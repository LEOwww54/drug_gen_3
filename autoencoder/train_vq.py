"""Train/export fragment RVQ tokens: python -m autoencoder.train_vq --help."""
import argparse
import json
import math
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


def train_vq(epochs, learning_rate, smiles_list, *,
             checkpoint='autoencoder/fragment_vq.pth', batch_size=64,
             max_nodes=70, num_codes=4, codebook_size=256, seed=42, device=None):
    """Train RVQ directly from SMILES and return the best validation model (eval).

    Inputs are encoded as supplied, without molecular decomposition. Canonical
    duplicates are removed; at least two distinct structures are required.
    Invalid/unsupported SMILES raise ValueError. The best weights are saved to
    checkpoint (an existing file at that path will be overwritten).
    """
    for name, value in [('epochs', epochs), ('batch_size', batch_size),
                        ('max_nodes', max_nodes), ('num_codes', num_codes),
                        ('codebook_size', codebook_size)]:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    if (isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float))
            or not math.isfinite(learning_rate) or learning_rate <= 0):
        raise ValueError('learning_rate must be finite and positive')
    if not isinstance(smiles_list, list) or not all(isinstance(s, str) for s in smiles_list):
        raise TypeError('smiles_list must be a list of SMILES strings')
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)
    dataset = FragmentDataset(smiles_list, max_nodes)
    if len(dataset) < 2:
        raise ValueError('Training needs at least two distinct canonical fragments')
    val_size = max(1, round(len(dataset) * 0.1))
    train_data, val_data = random_split(dataset, [len(dataset) - val_size, val_size],
                                      generator=torch.Generator().manual_seed(seed))
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    model = FragmentVQAutoencoder(max_nodes=max_nodes, num_codes=num_codes,
                                 codebook_size=codebook_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    best = float('inf')
    for epoch in range(epochs):
        train_loss, usage = run_epoch(model, train_loader, device, optimizer)
        val_loss, _ = run_epoch(model, val_loader, device)
        print(json.dumps(dict(epoch=epoch + 1, train=train_loss, validation=val_loss,
                              used_codes_per_stage=usage)))
        if val_loss['total'] < best:
            best = val_loss['total']
            model.save(checkpoint)
    return FragmentVQAutoencoder.load(checkpoint, device)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('mode', choices=['train', 'export'])
    parser.add_argument('--input', required=True, help='Fragment SMILES text or statistics JSON')
    parser.add_argument('--checkpoint', default='autoencoder/fragment_vq.pth')
    parser.add_argument('--output', default='autoencoder/fragment_tokens.json')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--learning-rate', '--lr', type=float, default=1e-4)
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
    # read_fragments also supports JSON decomposition records.
    from autoencoder.fragment_vq import canonical_fragment
    train_vq(args.epochs, args.learning_rate,
             [canonical_fragment(f)[0] for f in fragments],
             checkpoint=args.checkpoint, batch_size=args.batch_size,
             max_nodes=args.max_nodes, num_codes=args.num_codes,
             codebook_size=args.codebook_size, seed=args.seed, device=args.device)


if __name__ == '__main__':
    import json, constant
    data = json.load(open('../stru_data.json'))
    subsmiles = [i for i, v in data['pamf'].items()]
    train_vq(50, 1e-4,
             subsmiles,
             checkpoint='autoencoder/fragment_vq.pth', batch_size=128,
             max_nodes=70, num_codes=60,
             codebook_size=5, seed=42, device=constant.device)

    pass
