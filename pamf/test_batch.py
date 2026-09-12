"""Exercise real spawn workers, not mocked executors."""
from copy import deepcopy
import os
import unittest

from . import PAMFConfig, fragment_smiles, fragment_smiles_batch


class BatchTests(unittest.TestCase):
    def test_order_duplicates_and_independent_results(self):
        inputs = ['CCCCCC', 'CC', 'CCCOCCC', 'CCCCCC']
        config = PAMFConfig(mode='rules')
        expected = [fragment_smiles(s, config) for s in inputs]
        reference = {'unused': {'nested': [1, 2]}}
        original = deepcopy(reference)
        actual = fragment_smiles_batch(inputs, config, workers=2, reference=reference)
        self.assertEqual(actual, expected)
        self.assertEqual(reference, original)
        actual[0].append('changed')
        self.assertEqual(actual[3], expected[3])
        mapping = fragment_smiles_batch(inputs, config, workers=2, return_format='dict')
        self.assertEqual(list(mapping), list(dict.fromkeys(inputs)))
        self.assertEqual(mapping, dict(zip(inputs, expected)))

    def test_serial_empty_and_validation(self):
        config = PAMFConfig(mode='rules')
        self.assertEqual(fragment_smiles_batch([], config), [])
        self.assertEqual(fragment_smiles_batch([], config, return_format='dict'), {})
        self.assertEqual(fragment_smiles_batch(['CC'], config, workers=1), [['CC']])
        for kwargs in ({'workers': 0}, {'workers': True}, {'return_format': 'other'}):
            with self.assertRaises(ValueError):
                fragment_smiles_batch([], config, **kwargs)
        with self.assertRaises(TypeError):
            fragment_smiles_batch('CC', config)
        with self.assertRaisesRegex(ValueError, r'input\[1\]'):
            fragment_smiles_batch(['CC', None], config)

    def test_worker_error_has_input_context(self):
        with self.assertRaisesRegex(RuntimeError, r'input\[1\].*invalid'):
            fragment_smiles_batch(['CC', 'invalid'], PAMFConfig(mode='rules'), workers=2)

    @unittest.skipUnless(os.environ.get('PAMF_RUN_XTB') == '1', 'Opt-in real xTB')
    def test_real_xtb_parallel_matches_serial(self):
        inputs = ['CCCOCCC', 'CCCCCC', 'CCCOCCC']
        config = PAMFConfig(conformer_count=2, xtb_threads=1)
        serial = fragment_smiles_batch(inputs, config, workers=1)
        self.assertEqual(fragment_smiles_batch(inputs, config, workers=2), serial)


if __name__ == '__main__':
    unittest.main()
