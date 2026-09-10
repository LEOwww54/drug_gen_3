"""Bond-aware fragment autoencoder with residual vector quantization.

This module deliberately has no dependency on the GPT/tokenizer pipeline.
"""
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from rdkit import Chem


FEATURE_VERSION = 1
BONDS = [None, Chem.BondType.SINGLE, Chem.BondType.DOUBLE,
         Chem.BondType.TRIPLE, Chem.BondType.AROMATIC]


def canonical_fragment(value):
    """Keep dummy attachment atoms, remove arbitrary atom-map/connection IDs."""
    if isinstance(value, dict):
        value = value.get('raw_mol') or value.get('smiles')
    mol = Chem.MolFromSmiles(value) if isinstance(value, str) else Chem.Mol(value)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError(f'Invalid or empty fragment: {value}')
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
        if atom.GetAtomicNum() == 0:
            atom.SetIsotope(0)
    smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    return smiles, Chem.MolFromSmiles(smiles)


def fragment_graph(value, max_nodes):
    smiles, mol = canonical_fragment(value)
    n = mol.GetNumAtoms()
    if n > max_nodes:
        raise ValueError(f'Fragment has {n} atoms; max_nodes={max_nodes}')
    # Padding=0, dummy=1, real elements=atomic number+1.
    atoms = torch.zeros(max_nodes, dtype=torch.long)
    charge = torch.zeros_like(atoms)
    aromatic = torch.zeros_like(atoms)
    hydrogens = torch.zeros_like(atoms)
    bonds = torch.zeros(max_nodes, max_nodes, dtype=torch.long)
    for atom in mol.GetAtoms():
        i, q = atom.GetIdx(), atom.GetFormalCharge()
        h = atom.GetTotalNumHs()
        if not -5 <= q <= 5 or h > 8:
            raise ValueError('Unsupported formal charge or hydrogen count')
        atoms[i] = atom.GetAtomicNum() + 1
        charge[i], aromatic[i], hydrogens[i] = q + 5, int(atom.GetIsAromatic()), h
    for bond in mol.GetBonds():
        if bond.GetBondType() not in BONDS:
            raise ValueError(f'Unsupported bond: {bond.GetBondType()}')
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bonds[i, j] = bonds[j, i] = BONDS.index(bond.GetBondType())
    return dict(atoms=atoms, charge=charge, aromatic=aromatic, hydrogens=hydrogens,
                bonds=bonds, mask=torch.arange(max_nodes) < n,
                n_nodes=torch.tensor(n), smiles=smiles)


class FragmentDataset(Dataset):
    def __init__(self, fragments, max_nodes=70):
        self.samples = []
        seen = set()
        for fragment in fragments:
            sample = fragment_graph(fragment, max_nodes)
            if sample['smiles'] not in seen:
                self.samples.append(sample)
                seen.add(sample['smiles'])
        if not self.samples:
            raise ValueError('No fragments supplied')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def read_fragments(path):
    """Read SMILES lines, a JSON list, or nested {SMILES: count} statistics."""
    path = Path(path)
    if path.suffix.lower() != '.json':
        return [line.strip() for line in path.read_text(encoding='utf-8').splitlines()
                if line.strip()]
    data = json.loads(path.read_text(encoding='utf-8'))
    if isinstance(data, list):
        return data
    fragments = []
    def visit(node):
        if not isinstance(node, dict):
            raise ValueError('Expected nested fragment frequency dictionaries')
        for key, value in node.items():
            if isinstance(value, (int, float)):
                fragments.append(key)
            else:
                visit(value)
    visit(data)
    return fragments


class ResidualVectorQuantizer(nn.Module):
    def __init__(self, dim, num_codes=4, codebook_size=256, commitment=0.25):
        super().__init__()
        if min(dim, num_codes, codebook_size) < 1:
            raise ValueError('Quantizer dimensions must be positive')
        self.commitment = commitment
        self.codebooks = nn.ModuleList([nn.Embedding(codebook_size, dim) for _ in range(num_codes)])
        for table in self.codebooks:
            nn.init.normal_(table.weight, std=0.1)

    def forward(self, z):
        residual, quantized = z, torch.zeros_like(z)
        losses, indices = [], []
        for table in self.codebooks:
            distances = (residual.square().sum(-1, keepdim=True)
                         + table.weight.square().sum(-1)
                         - 2 * residual @ table.weight.T)
            index = distances.argmin(-1)
            code = table(index)
            losses.append(F.mse_loss(code, residual.detach())
                          + self.commitment * F.mse_loss(residual, code.detach()))
            quantized = quantized + code
            residual = residual - code.detach()
            indices.append(index)
        straight_through = z + (quantized - z).detach()
        return straight_through, torch.stack(indices, -1), torch.stack(losses).mean()

    def lookup(self, indices):
        if indices.ndim != 2 or indices.shape[1] != len(self.codebooks):
            raise ValueError('Expected [batch, num_codes] indices')
        return torch.stack([table(indices[:, i]) for i, table in enumerate(self.codebooks)]).sum(0)


class BondBlock(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.heads = heads
        self.bond_bias = nn.Embedding(5, heads)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(dim * 4, dim))

    def forward(self, x, bonds, mask):
        b, n, _ = x.shape
        bias = self.bond_bias(bonds).permute(0, 3, 1, 2)
        bias = bias.masked_fill(~mask[:, None, None, :], float('-inf'))
        h = self.norm1(x)
        x = x + self.attn(h, h, h, attn_mask=bias.reshape(b * self.heads, n, n),
                          need_weights=False)[0]
        return (x + self.ffn(self.norm2(x))) * mask.unsqueeze(-1)


class FragmentVQAutoencoder(nn.Module):
    def __init__(self, max_nodes=70, d_model=128, latent_dim=128, n_heads=4,
                 n_layers=3, num_codes=4, codebook_size=256, dropout=0.1):
        super().__init__()
        self.config = dict(max_nodes=max_nodes, d_model=d_model, latent_dim=latent_dim,
                           n_heads=n_heads, n_layers=n_layers, num_codes=num_codes,
                           codebook_size=codebook_size, dropout=dropout)
        self.max_nodes = max_nodes
        self.atom = nn.Embedding(120, d_model, padding_idx=0)
        self.charge = nn.Embedding(11, d_model)
        self.aromatic = nn.Embedding(2, d_model)
        self.hydrogens = nn.Embedding(9, d_model)
        self.encoder = nn.ModuleList([BondBlock(d_model, n_heads, dropout) for _ in range(n_layers)])
        self.to_latent = nn.Sequential(nn.Linear(d_model, latent_dim), nn.LayerNorm(latent_dim))
        self.quantizer = ResidualVectorQuantizer(latent_dim, num_codes, codebook_size)
        self.from_latent = nn.Linear(latent_dim, d_model)
        self.positions = nn.Parameter(torch.randn(max_nodes, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_model * 4, dropout,
                                           batch_first=True)
        self.decoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.heads = nn.ModuleDict({name: nn.Linear(d_model, size) for name, size in
                                   [('atoms', 120), ('charge', 11), ('aromatic', 2), ('hydrogens', 9)]})
        self.edge = nn.Linear(d_model * 2, 5)
        self.length = nn.Linear(latent_dim, max_nodes + 1)
        self.register_buffer('pairs', torch.triu_indices(max_nodes, max_nodes, 1), persistent=False)

    def encode(self, batch):
        mask = batch['mask']
        x = (self.atom(batch['atoms']) + self.charge(batch['charge'])
             + self.aromatic(batch['aromatic']) + self.hydrogens(batch['hydrogens']))
        for block in self.encoder:
            x = block(x, batch['bonds'], mask)
        pooled = (x * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        z = self.to_latent(pooled)
        quantized, codes, loss = self.quantizer(z)
        return quantized, codes, loss

    def decode(self, z):
        x = self.from_latent(z)[:, None, :] + self.positions[None, :, :]
        x = self.decoder(x)
        result = {name: head(x) for name, head in self.heads.items()}
        i, j = self.pairs
        result['bonds'] = self.edge(torch.cat([x[:, i], x[:, j]], -1))
        result['length'] = self.length(z)
        return result

    def forward(self, batch):
        z, codes, vq_loss = self.encode(batch)
        return dict(self.decode(z), z=z, codes=codes, vq_loss=vq_loss)

    def loss(self, pred, batch):
        mask = batch['mask']
        losses = {name: F.cross_entropy(pred[name][mask], batch[name][mask]) for name in self.heads}
        i, j = self.pairs
        valid = mask[:, i] & mask[:, j]
        # No-bond (class 0) is a real target; only padding pairs are ignored.
        losses['bonds'] = (F.cross_entropy(pred['bonds'][valid], batch['bonds'][:, i, j][valid])
                           if valid.any() else pred['bonds'].sum() * 0)
        losses['length'] = F.cross_entropy(pred['length'], batch['n_nodes'])
        losses['vq'] = pred['vq_loss']
        losses['total'] = sum(losses.values())
        return losses

    @torch.no_grad()
    def encode_fragments(self, fragments):
        """Stable export in eval mode; accepts SMILES or decomposition records."""
        from torch.utils.data import default_collate
        training = self.training
        self.eval()
        try:
            samples = [fragment_graph(f, self.max_nodes) for f in fragments]
            if not samples:
                return []
            batch = default_collate(samples)
            device = next(self.parameters()).device
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            _, codes, _ = self.encode(batch)
            return [dict(smiles=s['smiles'], codes=row,
                         tokens=[f'<FZ{i + 1}_{code}>' for i, code in enumerate(row)])
                    for s, row in zip(samples, codes.cpu().tolist())]
        finally:
            self.train(training)

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(feature_version=FEATURE_VERSION, config=self.config,
                        model_state_dict=self.state_dict()), path)

    @classmethod
    def load(cls, path, device='cpu'):
        state = torch.load(path, map_location=device, weights_only=True)
        if state['feature_version'] != FEATURE_VERSION:
            raise ValueError('Incompatible fragment feature version')
        model = cls(**state['config']).to(device)
        model.load_state_dict(state['model_state_dict'])
        return model.eval()
