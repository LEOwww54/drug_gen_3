import os
import pickle
import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import mock_open, patch

from pamf import PAMFConfig
from decompose.base import _mol_decom_mp, _pamf_frag_records, _mol_decom_mp_to_pkl_file
from decomposer import mol_decom_mp
from rdkit import Chem
from decompose.base import _pamf_statistics_smiles, _pamf_one_record, _fragment_statistics


class AdapterTests(unittest.TestCase):
    def test_refined_pamf_splits_options_and_return_order(self):
        from ZINC_refined.dataprocess import mol_decomp_mp_ZINC_refined_pamf_pkl
        splits = {'train': ['CC', 'CC'], 'test': ['CO']}
        config = PAMFConfig(mode='rules')
        reference = object()
        with patch('ZINC_refined.dataprocess.data_from_ZINC_refined',
                   return_value=(splits, ['CC', 'CC', 'CO'])) as load, \
                patch('ZINC_refined.dataprocess.mol_decom_mp', side_effect=[
                    ([], ['train1', 'train2'], [], [], {}),
                    ([], ['test1'], [], [], {})]) as run:
            result = mol_decomp_mp_ZINC_refined_pamf_pkl(
                2, pamf_config=config, pamf_reference=reference)
        load.assert_called_once_with(n=0)
        self.assertEqual(result, ['train1', 'train2', 'test1'])
        self.assertEqual(run.call_count, 2)
        for split, call in zip(('train', 'test'), run.call_args_list):
            self.assertEqual(call.kwargs, dict(
                smiles=splits[split], n_core=2, output_format='pkl',
                output_path=[f'gpt/frag_file/frag_decom_ZINC_refined_pamf_{split}.pkl'],
                method='pamf', pamf_config=config, pamf_reference=reference,
                statistics_path=f'stru_data_ZINC_refined_pamf_{split}.json'))

    def test_custom_statistics_paths_do_not_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            train_path = Path(tmp) / 'nested' / 'train.json'
            test_path = Path(tmp) / 'nested' / 'test.json'
            for smiles, path in [('CC', train_path), ('CO', test_path)]:
                record = _pamf_one_record(smiles, [smiles], False)
                with patch('decompose.base._pamf_frag_records', return_value=[record]):
                    mol_decom_mp([smiles], 1, method='pamf', properties=[0],
                                 stat_only=True, statistics_path=path)
            self.assertEqual(json.loads(train_path.read_text()), {'pamf': {'CC': 1}})
            self.assertEqual(json.loads(test_path.read_text()), {'pamf': {'CO': 1}})

    def test_dataset_statistics_filenames(self):
        from ZINC_250K.dataprocess import mol_decomp_mp_ZINC_250K_pamf_pkl
        from ZINC_refined.dataprocess import _mol_decomp_mp_ZINC_refined
        result = ([], [], [], [], {})
        with patch('ZINC_250K.dataprocess.data_from_ZINC_250K', return_value=(['CC'], ['CO'], [])), \
                patch('ZINC_250K.dataprocess.mol_decom_mp', return_value=result) as run:
            mol_decomp_mp_ZINC_250K_pamf_pkl(1)
            self.assertEqual([c.kwargs['statistics_path'] for c in run.call_args_list],
                             ['stru_data_ZINC_250K_pamf_train.json', 'stru_data_ZINC_250K_pamf_test.json'])
        with patch('ZINC_refined.dataprocess.data_from_ZINC_refined',
                   return_value=({'train': ['CC'], 'test': ['CO']}, [])), \
                patch('ZINC_refined.dataprocess.mol_decom_mp', return_value=result) as run:
            _mol_decomp_mp_ZINC_refined(1)
            self.assertEqual([c.kwargs['statistics_path'] for c in run.call_args_list],
                             ['stru_data_ZINC_refined_legacy_train.json', 'stru_data_ZINC_refined_legacy_test.json'])

    def test_parallel_record_conversion_failure_and_order(self):
        inputs = ['CCCCCC', 'CC', 'CO', 'CCCCCC', 'CN']
        batches = [['[1*]CCC', '[1*]CCC'], ['invalid'], ['[1*]CCC', '[1*]CCC'], ['CN']]
        with patch('pamf.fragment_smiles_batch', return_value=(batches, [0, 1, 3, 4])):
            records = _pamf_frag_records(inputs, 2, prop_calu=False)
        self.assertEqual(len(records), len(inputs))
        self.assertIsNone(records[1])
        self.assertIsNone(records[2])
        for index in (0, 3, 4):
            self.assertEqual(records[index][0]['original_smiles'], inputs[index])
        self.assertEqual(records[0][0]['fragments'][0]['statistics_smiles'], ['CCC'])
        self.assertEqual(Chem.MolToSmiles(records[3][0]['fragments'][0]['raw_mol']), '[1*]CCC')
        self.assertTrue(records[0][0]['fragments'][0]['raw_mol_props'])
        records[0][0]['fragments'][0]['raw_mol'].GetAtomWithIdx(0).SetIsotope(99)
        self.assertEqual(records[3][0]['fragments'][0]['raw_mol'].GetAtomWithIdx(0).GetIsotope(), 1)

    def test_kekulize_before_dummy_removal(self):
        for smiles in ['[1*]Nc1ncnc2c1cnn2[2*]', '[1*]n1ncc2c(N[2*])ncnc21',
                       '[1*]n1cccc1', '[1*]n1nnnc1']:
            with self.subTest(smiles=smiles):
                mol = Chem.MolFromSmiles(smiles)
                before = Chem.MolToSmiles(mol)
                aromatic = [a.GetIsAromatic() for a in mol.GetAtoms()]
                result = _pamf_statistics_smiles(mol)
                self.assertEqual(len(result), 1)
                self.assertNotIn('*', result[0])
                restored = Chem.MolFromSmiles(result[0])
                self.assertIsNotNone(restored)
                self.assertTrue(any(a.GetIsAromatic() for a in restored.GetAtoms()))
                self.assertEqual(Chem.MolToSmiles(mol), before)
                self.assertEqual([a.GetIsAromatic() for a in mol.GetAtoms()], aromatic)
        actual = _pamf_statistics_smiles(Chem.MolFromSmiles('[1*]Nc1ncnc2c1cnn2[2*]'))
        expected = Chem.MolToSmiles(Chem.MolFromSmiles('NC1=C2C=NNC2=NC=N1'))
        self.assertEqual(actual, [expected])

    def test_statistics_remove_dummy_atoms_only(self):
        record = _pamf_one_record('CCCCCC', ['[1*]CCC', '[2*]CCC',
                                           '[3*]c1ccccc1', '[4*]CC[5*]'], False)[0]
        self.assertEqual(_fragment_statistics([record]),
                         {'pamf': {'CCC': 2, 'c1ccccc1': 1, 'CC': 1}})
        self.assertIn('*', Chem.MolToSmiles(record['fragments'][0]['raw_mol']))
        self.assertEqual(record['fragments'][0]['smiles'], '[1*]CCC')
        self.assertEqual(_pamf_statistics_smiles(Chem.MolFromSmiles('C*C')), ['C', 'C'])
        self.assertEqual(_pamf_statistics_smiles(Chem.MolFromSmiles('*')), [])
        self.assertEqual(_pamf_statistics_smiles(Chem.MolFromSmiles('[1*]C(=O)[O-]')),
                         ['O=C[O-]'])

    def test_pickle_fragment_smiles_alignment(self):
        for method in ('pamf', 'legacy'):
            inputs = ['CCCCCC', 'CC', 'CCCCCC']
            properties = [10, 20, 30]
            if method == 'pamf':
                inputs.insert(1, 'invalid')
                properties.insert(1, 99)
            with self.subTest(method=method), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'fragments.pkl'
                # Suppress only statistics output; exercise the actual PKL writer.
                real_open = open
                def statistics_open(file, *args, **kwargs):
                    if str(file) == 'stru_data.json':
                        return mock_open()(file, *args, **kwargs)
                    return real_open(file, *args, **kwargs)
                with patch('decompose.base.open', side_effect=statistics_open, create=True):
                    output = mol_decom_mp(inputs, 1, properties=properties,
                                          output_path=[path], method=method,
                                          pamf_config=PAMFConfig(mode='rules'))
                self.assertEqual(len(output), 5)
                with path.open('rb') as handle:
                    rows = pickle.load(handle)['mol']
                self.assertEqual([r['oring'] for r in rows.values()], ['CCCCCC', 'CC', 'CCCCCC'])
                self.assertEqual([r['props'] for r in rows.values()], [10, 20, 30])
                self.assertEqual(rows[0]['fragment_smiles'], rows[2]['fragment_smiles'])
                self.assertEqual(rows[1]['fragment_smiles'], ['CC'])
                self.assertTrue(rows[0]['fragment_smiles'])
                if method == 'pamf':
                    self.assertEqual(rows[0]['fragment_smiles'], ['[1*]CCC'] * 2)

    def test_fragment_smiles_token_failure_alignment(self):
        with patch('decompose.base.open', mock_open(), create=True), \
                patch('decompose.base._process_list_parallel',
                      return_value=[{'error': 'conversion failed'}, ('ok', ['ok'])]):
            output = _mol_decom_mp(['CCCCCC', 'CC'], 1, properties=[10, 20],
                                  method='pamf', pamf_config=PAMFConfig(mode='rules'),
                                  return_fragment_smiles=True)
        self.assertEqual(output[2], ['CC'])
        self.assertEqual(output[5], [['CC']])
        with self.assertRaisesRegex(ValueError, 'fragment_smiles length'):
            _mol_decom_mp_to_pkl_file(['sentence'], ['CC'], [10], fragment_smiles=[])

    def test_failed_molecule_filters_properties_and_duplicates(self):
        inputs = ['CCCCCC', 'invalid', 'CCCOCCC', 'CCCCCC']
        with patch('decompose.base.open', mock_open(), create=True):
            output = _mol_decom_mp(inputs, 2, properties=[10, 20, 30, 40],
                                  method='pamf', pamf_config=PAMFConfig(mode='rules'))
        self.assertEqual(output[2], ['CCCCCC', 'CCCOCCC', 'CCCCCC'])
        self.assertEqual(output[3], [10, 30, 40])
        self.assertEqual(len(output[0]), 3)
        self.assertEqual(output[1][0], output[1][2])

    def test_token_failure_and_all_failed(self):
        with patch('decompose.base.open', mock_open(), create=True), \
                patch('decompose.base._process_list_parallel',
                      return_value=[{'error': 'conversion failed'}, ('ok', ['ok'])]):
            output = _mol_decom_mp(['CCCCCC', 'CC'], 1, properties=[10, 20],
                                  method='pamf', pamf_config=PAMFConfig(mode='rules'))
        self.assertEqual(output[:4], (['ok'], [['ok']], ['CC'], [20]))
        self.assertEqual(output[4], {'pamf': {'CC': 1}})
        with patch('decompose.base.open', mock_open(), create=True):
            output = _mol_decom_mp(['invalid'], 1, properties=[10],
                                  method='pamf', pamf_config=PAMFConfig(mode='rules'))
        self.assertEqual(output, ([], [], [], [], {}))

    def test_parallel_output_alignment(self):
        inputs = ['CCCCCC', 'CCCOCCC', 'CCCCCC']
        properties = [{'id': i} for i in range(3)]
        with patch('decompose.base.open', mock_open(), create=True):
            sentences, tokens, originals, props, stats = _mol_decom_mp(
                inputs, 2, properties=properties, method='pamf',
                pamf_config=PAMFConfig(mode='rules'))
        self.assertEqual(originals, inputs)
        self.assertEqual(props, properties)
        self.assertEqual(len(sentences), 3)
        self.assertEqual(len(tokens), 3)
        self.assertEqual(tokens[0], tokens[2])
        self.assertNotEqual(tokens[0], tokens[1])
        self.assertIn('pamf', stats)

    def test_adapter_links_and_auto_properties(self):
        rows = _pamf_frag_records(['CCCCCC'], 1, PAMFConfig(mode='rules'))
        record = rows[0][0]
        self.assertIsNotNone(record['prop'])
        self.assertEqual([r['smiles'] for r in record['fragments']], ['[1*]CCC']*2)
        for row in record['fragments']:
            self.assertEqual(len(row['raw_mol_props']), row['raw_mol'].GetNumAtoms())
        with patch('decompose.base.open', mock_open(), create=True):
            output = _mol_decom_mp(['CCCCCC'], 1, method='pamf',
                                  pamf_config=PAMFConfig(mode='rules'))
        self.assertEqual(len(output[3]), 1)

    def test_legacy_and_misalignment(self):
        with patch('decompose.base.open', mock_open(), create=True):
            output = _mol_decom_mp(['CCCCCC', 'CCCCCC'], 2, properties=[10, 20])
        self.assertEqual(output[2], ['CCCCCC', 'CCCCCC'])
        self.assertEqual(output[3], [10, 20])
        with self.assertRaisesRegex(ValueError, 'same length'):
            _mol_decom_mp(['CC'], 1, properties=[])
        with patch('decompose.base._process_list_parallel', return_value=[]):
            with self.assertRaisesRegex(ValueError, 'length'):
                _mol_decom_mp(['CC'], 1)

    @unittest.skipUnless(os.environ.get('PAMF_RUN_XTB') == '1', 'Opt-in real xTB')
    def test_real_xtb_adapter(self):
        with patch('decompose.base.open', mock_open(), create=True):
            output = _mol_decom_mp(['CCCCCC', 'CCCOCCC', 'CCCCCC'], 2,
                                  properties=[0, 1, 2], method='pamf',
                                  pamf_config=PAMFConfig(conformer_count=2))
        self.assertEqual(output[2], ['CCCCCC', 'CCCOCCC', 'CCCCCC'])
        self.assertEqual(output[3], [0, 1, 2])
        self.assertEqual(len(output[0]), 3)
        self.assertEqual(output[1][0], output[1][2])


if __name__ == '__main__':
    unittest.main()
