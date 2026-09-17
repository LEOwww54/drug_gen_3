"""Small CPU training tests; never touch real datasets or checkpoints."""
import json
import pickle
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import constant
import gpt.tokenizer as tokenization
from gpt.dataset import PAMFDataSet, _split_validation_test, get_pamf_dataloader_without_split
import train as training


class PAMFTrainingTests(unittest.TestCase):
    def fixtures(self, folder):
        paths = {key: Path(folder)/name for key, name in
                 [('train', 'train.pkl'), ('test', 'test.pkl'),
                  ('tokenizer', 'vocab.json'), ('checkpoint', 'checkpoints')]}
        rows = {0: dict(frag='<start> [C] <m- 1> i1 </s>', oring='CC', props=[0.2, 1., 0.8]),
                2: dict(frag='<start> [C] </s>', oring='CC', props=[0.3, 2., 0.9])}
        for split, records in [('train', rows), ('test', {4: dict(
                frag='<start> unseen </s>', oring='O', props=[0.1, 0., 0.9]),
                5: dict(frag='<start> [C] </s>', oring='C', props=[0.2, 1., 0.8])})]:
            with paths[split].open('wb') as handle:
                pickle.dump({'mol': records, 'protein_dict': []}, handle)
        return paths

    def test_holdout_split_is_disjoint_complete_and_reproducible(self):
        for size in (4, 5):
            valid, test = _split_validation_test(list(range(size)), 2, None)
            repeated, _ = _split_validation_test(list(range(size)), 2, None)
            self.assertEqual(len(valid.dataset), size // 2)
            self.assertEqual(len(test.dataset), size - size // 2)
            self.assertFalse(set(valid.dataset.indices) & set(test.dataset.indices))
            self.assertEqual(set(valid.dataset.indices) | set(test.dataset.indices), set(range(size)))
            self.assertEqual(valid.dataset.indices, repeated.dataset.indices)
        with self.assertRaisesRegex(ValueError, 'At least two'):
            _split_validation_test([0], 2, None)

    def test_explicit_validation_preserves_test_set(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = self.fixtures(folder)
            tok = tokenization.get_new_pamf_tokenizer('ZINC_250K', train_file=paths['train'],
                                                     output_path=paths['tokenizer'])
            _, valid, test = get_pamf_dataloader_without_split(
                tok, paths['train'], paths['test'], valid_file=paths['train'])
            self.assertEqual(len(valid[0].dataset), 2)
            self.assertEqual(len(test[0].dataset), 2)

    def test_train_only_vocab_and_row_alignment(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = self.fixtures(folder)
            tok = tokenization.get_new_pamf_tokenizer('ZINC_250K', train_file=paths['train'],
                                                     output_path=paths['tokenizer'])
            for removed in constant.REMOVED_SPECIAL_TOKENS:
                self.assertIsNone(tok.token_to_id(removed))
            for index, token in enumerate(constant.SPECIAL_TOKENS):
                self.assertEqual(tok.token_to_id(token), index)
            self.assertIsNone(tok.token_to_id('unseen'))
            self.assertEqual(tok.encode('[C] i1').ids, [tok.token_to_id('[C]'), tok.token_to_id('i1')])
            self.assertIsNone(tok.token_to_id('<m- 1>'))
            expected = [tok.token_to_id('<m-'), tok.token_to_id('1>')]
            self.assertTrue(all(i is not None for i in expected))
            self.assertEqual(tok.encode('<m- 1>').ids, expected)
            self.assertEqual(tokenization.tokenizer_from_file(str(paths['tokenizer'])).encode('<m- 1>').ids,
                             expected)
            dataset = PAMFDataSet(paths['train'], tok)
            self.assertEqual(dataset[0]['decoder_input'][2:4], expected)
            self.assertEqual(dataset.row_ids, [0, 2])
            self.assertEqual(dataset.original_smiles, ['CC', 'CC'])
            self.assertEqual(dataset[1]['props'], [0.3, 2., 0.9])
            first = dataset[0]['decoder_input']
            batch = dataset.padding_batch([dataset[0], dataset[1]])
            self.assertEqual(dataset[0]['decoder_input'], first)
            self.assertEqual(batch[-1].shape, (2, 3))
            test = PAMFDataSet(paths['test'], tok)
            self.assertEqual(test.unknown_tokens, 1)
            self.assertIn(constant.UNK_TOKEN_ID, test[0]['decoder_input'])

    def test_legacy_tokenizer_is_rejected(self):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        legacy = ['<pad>', '<s>', '</s>', '<unk>', '<cls>', '<start>', '<sep>', '<sep1>']
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'legacy.json'
            Tokenizer(WordLevel(vocab={t: i for i, t in enumerate(legacy)},
                                unk_token='<unk>')).save(str(path))
            with self.assertRaisesRegex(ValueError, 'Legacy or incompatible'):
                tokenization.tokenizer_from_file(path)

    def test_both_training_entries_one_cpu_epoch(self):
        small = dict(d_model=24, emb_size=24, d_ff=48, d_k=12, d_v=12,
                     n_heads=2, n_layers=1, device=torch.device('cpu'))
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            for source, fn, conditional, p_type in [
                ('ZINC_250K', training.train_fragGPT_ZINC_250K_pamf, ('prop',), 'Riemannian'),
                ('ZINC_refined', training.train_fragGPT_ZINC_refined_pamf, ('unconditional',), 'lora_1')]:
                with self.subTest(source=source), tempfile.TemporaryDirectory() as folder:
                    paths = self.fixtures(folder)
                    with patch.object(tokenization, 'pamf_training_paths', return_value=paths), \
                            patch.dict(training.gpt.__dict__, small), \
                            patch.object(training.gpt.torch, 'load', wraps=torch.load) as load_mock:
                        model = fn(1, s=True, batch_size=2, p_type=p_type, conditional=conditional,
                                   output_dir=paths['checkpoint'])
                    run_dir = Path(model.training_output_dir)
                    self.assertEqual(run_dir.parent, paths['checkpoint'].resolve())
                    self.assertTrue(run_dir.name.startswith(f'{conditional[0]}_{p_type}_'))
                    self.assertEqual(load_mock.call_count, 1)
                    self.assertEqual(Path(load_mock.call_args.args[0]), run_dir/'GPT.pt')
                    self.assertTrue((run_dir/'GPT.pt').is_file())
                    self.assertTrue((run_dir/'frag_tokenizer.json').is_file())
                    metadata = json.loads((run_dir/'training_config.json').read_text())
                    self.assertEqual(metadata['source'], source)
                    self.assertEqual(model.projection.out_features, metadata['vocab_size'])
        finally:
            torch.set_num_threads(previous_threads)


if __name__ == '__main__':
    unittest.main()
