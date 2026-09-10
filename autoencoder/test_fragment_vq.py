import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import default_collate
from autoencoder.fragment_vq import FragmentDataset, FragmentVQAutoencoder, fragment_graph


class FragmentVQTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)
        self.model = FragmentVQAutoencoder(max_nodes=10, d_model=16, latent_dim=16,
                                          n_heads=2, n_layers=1, num_codes=2, codebook_size=8)
        self.batch = default_collate([fragment_graph(s, 10) for s in
                                     ['[1*]CC(=O)[O-]', 'c1ccncc1', 'C']])

    def test_gradients_and_quantized_decoder(self):
        pred = self.model(self.batch)
        loss = self.model.loss(pred, self.batch)['total']
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(self.model.atom.weight.grad.abs().sum().item(), 0)
        for table in self.model.quantizer.codebooks:
            self.assertGreater(table.weight.grad.abs().sum().item(), 0)
        torch.testing.assert_close(pred['z'], self.model.quantizer.lookup(pred['codes']))

    def test_export_is_stable_and_checkpoint_roundtrip(self):
        self.model.train()
        first = self.model.encode_fragments(['[1*]CC', '[82*]CC'])
        self.assertEqual(first[0], first[1])
        self.assertTrue(self.model.training)
        self.assertEqual(first, self.model.encode_fragments(['[1*]CC', '[82*]CC']))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'vq.pth'
            self.model.save(path)
            restored = FragmentVQAutoencoder.load(path)
            self.assertEqual(first, restored.encode_fragments(['[1*]CC', '[82*]CC']))

    def test_bond_topology_reaches_encoder(self):
        self.model.eval()
        observed = []
        hook = self.model.quantizer.register_forward_pre_hook(lambda _, args: observed.append(args[0].detach()))
        self.model.encode(self.batch)
        changed = dict(self.batch, bonds=torch.zeros_like(self.batch['bonds']))
        self.model.encode(changed)
        hook.remove()
        self.assertFalse(torch.allclose(observed[0], observed[1]))

    def test_no_bond_targets_and_single_atom(self):
        pred = self.model(self.batch)
        pred['bonds'].retain_grad()
        self.model.loss(pred, self.batch)['bonds'].backward()
        i, j = self.model.pairs
        valid = self.batch['mask'][:, i] & self.batch['mask'][:, j]
        no_bond = valid & (self.batch['bonds'][:, i, j] == 0)
        self.assertGreater(pred['bonds'].grad[no_bond].abs().sum().item(), 0)
        single = default_collate([fragment_graph('C', 10)])
        self.assertTrue(torch.isfinite(self.model.loss(self.model(single), single)['total']))

    def test_features_and_rejections(self):
        self.assertIn(1, self.batch['atoms'][0].tolist())
        self.assertIn(4, self.batch['charge'][0].tolist())
        self.assertEqual(self.batch['aromatic'][1].sum().item(), 6)
        self.assertEqual(len(FragmentDataset(['[1*]CC', 'CC[2*]'], 10)), 1)
        with self.assertRaises(ValueError):
            fragment_graph('CCCC', 2)
        with self.assertRaises(ValueError):
            fragment_graph('', 10)


if __name__ == '__main__':
    unittest.main()
