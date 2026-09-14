"""In-process backend contracts and opt-in native integration tests."""
import os
import unittest
from unittest.mock import patch

from . import PAMFConfig, decompose_smiles, fragment_smiles_batch
from .chemistry import parse_smiles
from .geometry import conformers
from .tblite_backend import run_tblite, BOHR_IN_ANGSTROM


class TBLiteConfigTests(unittest.TestCase):
    def test_default_and_validation(self):
        self.assertEqual(PAMFConfig().xtb_backend, 'cli')
        for kwargs in ({'xtb_backend': 'other'}, {'tblite_max_iterations': 0}):
            with self.assertRaises(ValueError):
                PAMFConfig(**kwargs)

    def test_missing_dependency_no_fallback(self):
        mol, ids, _ = conformers(parse_smiles('CCC'), count=1)
        with patch.dict('sys.modules', {'tblite.interface': None}):
            with self.assertRaisesRegex(RuntimeError, 'tblite backend is unavailable'):
                run_tblite(mol, ids[0])


@unittest.skipUnless(os.environ.get('PAMF_RUN_TBLITE') == '1', 'Opt-in native tblite tests')
class TBLiteNativeTests(unittest.TestCase):
    def test_default_pipeline_without_subprocess(self):
        with patch('pamf.electronic.subprocess.run', side_effect=AssertionError('CLI called')):
            result = decompose_smiles('CCCOCCC', PAMFConfig(conformer_count=2, xtb_backend='tblite'))
        self.assertEqual(result['electronics']['backend'], 'tblite')
        self.assertTrue(result['reconstruction_valid'])
        self.assertTrue(all(r['wbo'] > 0 for r in result['candidates']))

    def test_direct_reference_units_and_charge(self):
        import numpy as np
        from tblite.interface import Calculator
        from threadpoolctl import threadpool_limits
        for smiles in ('CCO', 'CC[NH3+]', '[CH3]'):
            mol, ids, _ = conformers(parse_smiles(smiles), count=1)
            data = run_tblite(mol, ids[0])
            charge = sum(a.GetFormalCharge() for a in mol.GetAtoms())
            uhf = sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms())
            with threadpool_limits(limits=1):
                calc = Calculator('GFN2-xTB', np.array([a.GetAtomicNum() for a in mol.GetAtoms()]),
                                  mol.GetConformer(ids[0]).GetPositions()/BOHR_IN_ANGSTROM,
                                  charge=charge, uhf=uhf)
                calc.set('verbosity', 0)
                expected = calc.singlepoint()
            np.testing.assert_allclose(data['charges'], expected.get('charges'), atol=1e-8)
            self.assertAlmostEqual(sum(data['charges']), charge, places=6)
            self.assertEqual(len(data['wbo']), mol.GetNumAtoms()*(mol.GetNumAtoms()-1)//2)
            expected_orders = expected.get('bond-orders')
            if uhf:
                expected_orders = expected_orders.reshape(2, mol.GetNumAtoms(), mol.GetNumAtoms())[0]
            self.assertAlmostEqual(data['wbo'][(0, 1)], float(np.asarray(expected_orders[0, 1]).item()))

    def test_parallel_matches_serial(self):
        inputs = ['CCCCCC', 'CCCOCCC', 'CCCCCC']
        config = PAMFConfig(conformer_count=2, xtb_backend='tblite')
        serial = fragment_smiles_batch(inputs, config, workers=1, on_error='raise')
        parallel = fragment_smiles_batch(inputs, config, workers=2, on_error='raise')
        self.assertEqual(parallel, serial)
        self.assertIsNot(parallel[0], parallel[2])

    def test_iteration_failure(self):
        with self.assertRaises(RuntimeError):
            decompose_smiles('CCCOCCC', PAMFConfig(conformer_count=1, tblite_max_iterations=1, xtb_backend='tblite'))


if __name__ == '__main__':
    unittest.main()
