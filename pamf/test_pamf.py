"""Run from the repository root: python -m unittest pamf.test_pamf -v."""

import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from rdkit import Chem
from . import PAMFConfig, decompose_smiles, fragment_smiles, reassemble
from .chemistry import analyze_bonds, parse_smiles
from .electronic import add_electronic_features, parse_charges, parse_wbo, run_xtb
from .geometry import conformers, torsion_variability
from .optimizer import select_cuts
from .scoring import fit_reference, score_candidates


class ChemistryTests(unittest.TestCase):
    def test_roundtrip_and_pairing(self):
        smiles_list = [
            'CCCCCC', 'CCCOCCNCCC', 'c1ccccc1CCCc1ccccc1',
            'CCNC(=O)NCCC', 'CCOC(=O)NCCC', 'CCNS(=O)(=O)CCC',
            'CCC[NH2+]CCC.[Cl-]', '[13CH3]CCCCCC',
            'CCC[C@H](O)CCCCCC', 'CCC/C=C/CCCCC',
            '[CH3:7]CCCCC', 'C1CCC2(CC1)CCCC2', 'c1ccc2ccccc2c1',
            'CC(=O)[O-]', 'CC[N+](=O)[O-]', 'NC(=[NH2+])N',
        ]
        for smiles in smiles_list:
            with self.subTest(smiles=smiles):
                result = decompose_smiles(smiles, PAMFConfig(mode='rules'))
                self.assertEqual(reassemble(result['fragments']), result['parent_smiles'])
                self.assertEqual(result['parent_smiles'], result['reconstructed_smiles'])
                counts = {}
                for f in result['fragments']:
                    for atom in Chem.MolFromSmiles(f).GetAtoms():
                        if atom.GetAtomicNum() == 0:
                            counts[atom.GetIsotope()] = counts.get(atom.GetIsotope(), 0)+1
                self.assertEqual(counts, {int(k): 2 for k in result['connections']})
                json.dumps(result, allow_nan=False)

    def test_meaningful_split_and_small_inputs(self):
        fragments = fragment_smiles('CCCCCC', PAMFConfig(mode='rules'))
        self.assertEqual(fragments, ['[1*]CCC', '[1*]CCC'])
        self.assertEqual(reassemble(fragments), 'CCCCCC')
        for smiles in ('C', 'CC', '[Na+]', 'O'):
            self.assertEqual(len(fragment_smiles(smiles, PAMFConfig(mode='rules'))), 1)

    def test_protected_units(self):
        for smiles in ('CCNC(=O)NCCC', 'CCOC(=O)NCCC', 'CCNS(=O)(=O)CCC',
                       'CC=CC(=O)CCC', 'CCCc1ccccc1', 'CCOc1ccccc1',
                       'CCNC(=[NH2+])NCCC', 'CC[N+](=O)[O-]'):
            mol = parse_smiles(smiles)
            candidates, protected = analyze_bonds(mol)
            self.assertTrue(protected)
            for row in candidates:
                b = mol.GetBondWithIdx(row['bond_idx'])
                self.assertFalse(b.IsInRing() or b.GetIsConjugated())
                self.assertNotIn(str(b.GetIdx()), protected)
            # Protect all internal bonds of each recognized functional unit.
            from .chemistry import PROTECTED_GROUPS
            for smarts in PROTECTED_GROUPS.values():
                query = Chem.MolFromSmarts(smarts)
                for match in mol.GetSubstructMatches(query):
                    for b in query.GetBonds():
                        idx = mol.GetBondBetweenAtoms(match[b.GetBeginAtomIdx()], match[b.GetEndAtomIdx()]).GetIdx()
                        self.assertIn(str(idx), protected)

    def test_invalid_and_malformed(self):
        for value in ('', 'not a smiles', 'C[1*]', None):
            with self.assertRaises(ValueError):
                fragment_smiles(value, PAMFConfig(mode='rules'))
        for fragments in ([], ['C[1*]'], ['C*', 'N*'], ['C[1*]', 'N[1*]', 'O[1*]'],
                          ['C=[1*]', 'N[1*]'], ['[1*][1*]']):
            with self.assertRaises(ValueError):
                reassemble(fragments)

    def test_deterministic_and_constraints(self):
        config = PAMFConfig(mode='rules')
        self.assertEqual(decompose_smiles('CCCOCCNCCC', config), decompose_smiles('CCCOCCNCCC', config))
        ring = 'C1CCCCCCCCCCC1'
        result = decompose_smiles(ring, PAMFConfig(mode='rules', max_heavy_atoms=6))
        self.assertFalse(result['optimization']['size_limit_satisfied'])
        with self.assertRaisesRegex(ValueError, 'No feasible'):
            decompose_smiles(ring, PAMFConfig(mode='rules', max_heavy_atoms=6, strict_max_size=True))


class OptimizerTests(unittest.TestCase):
    def test_joint_selection_beats_independent_cuts(self):
        mol = parse_smiles('CCCCCCCCCC')
        rows = [dict(bond_idx=i, score=s) for i, s in ((2, 2.0), (3, 3.0), (6, 2.0))]
        config = PAMFConfig(mode='rules', imbalance_penalty=0)
        cuts, meta = select_cuts(mol, rows, config)
        self.assertEqual(cuts, (3, 6))
        self.assertTrue(meta['optimality_proven'])
        # Independent exhaustive oracle: remove bonds directly in RDKit.
        values = []
        for count in range(len(rows)+1):
            for subset in itertools.combinations(rows, count):
                editable = Chem.RWMol(mol)
                for row in subset:
                    bond = mol.GetBondWithIdx(row['bond_idx'])
                    editable.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
                sizes = [len(g) for g in Chem.GetMolFrags(editable)]
                if min(sizes) >= 3:
                    values.append(sum(r['score'] for r in subset)-0.8*(len(sizes)-1))
        self.assertAlmostEqual(meta['objective'], max(values))

    def test_beam_reports_approximation(self):
        result = decompose_smiles('CCCCCCCCCCCC', PAMFConfig(mode='rules', exact_candidate_limit=0))
        self.assertFalse(result['optimization']['optimality_proven'])
        self.assertTrue(all(n >= 3 for n in result['optimization']['fragment_heavy_atom_counts']))


class PhysicsTests(unittest.TestCase):
    def test_parsers(self):
        self.assertEqual(parse_wbo('1 2 9.5D-1\n2 3 0.02\n', 3), {(0, 1): 0.95, (1, 2): 0.02})
        self.assertEqual(parse_charges('0.1 -0.1', 2), [0.1, -0.1])
        for data in ('0 2 1.0', '1 4 1.0', '1 2 nan', '1 1 0.4', '1 2 1.0\n2 1 2.0'):
            with self.assertRaises(ValueError):
                parse_wbo(data, 3)
        with self.assertRaises(ValueError):
            parse_charges('0.5', 2)

    def test_geometry_is_real_and_repeatable(self):
        mol = parse_smiles('CCCOCCC')
        first, ids, meta = conformers(mol, count=4, seed=19)
        second, ids2, meta2 = conformers(mol, count=4, seed=19)
        self.assertEqual(meta, meta2)
        self.assertEqual(Chem.MolToXYZBlock(first, confId=ids[0]), Chem.MolToXYZBlock(second, confId=ids2[0]))
        rows, _ = analyze_bonds(mol)
        values = torsion_variability(first, rows, ids)
        self.assertTrue(all(v is None or 0 <= v <= 1 for v in values.values()))

    def test_cross_wbo_and_reference(self):
        mol = parse_smiles('CCCCCC')
        rows, _ = analyze_bonds(mol)
        data = dict(wbo={(i, i+1): 0.9+i*0.02 for i in range(5)},
                    charges=[0.0]*6, polarizabilities=[None]*6)
        data['wbo'][(1, 4)] = 0.1
        add_electronic_features(mol, rows, data)
        middle = next(r for r in rows if r['bond_idx'] == 2)
        self.assertAlmostEqual(middle['cross_wbo'], 1.04)
        reference = fit_reference([dict(candidates=rows)])
        score_candidates(rows, PAMFConfig(), reference)
        self.assertTrue(all('z_wbo' in r for r in rows))
        with self.assertRaises(ValueError):
            score_candidates(rows, PAMFConfig(), {})
        data['wbo'].pop((0, 1))
        with self.assertRaisesRegex(ValueError, 'missing'):
            add_electronic_features(mol, rows, data)

    def test_xtb_process_protocol_and_mapping(self):
        mol, ids, _ = conformers(parse_smiles('CCCCCC'), count=2)
        workdirs = []
        def fake_run(command, **kwargs):
            work = Path(kwargs['cwd'])
            workdirs.append(work)
            self.assertEqual(command[0], 'xtb')
            self.assertTrue(Path(command[1]).is_absolute())
            self.assertEqual(Path(command[1]), work/'molecule.xyz')
            self.assertNotIn('\\', command[1])
            self.assertEqual(int((work/'molecule.xyz').read_text().splitlines()[0]), mol.GetNumAtoms())
            self.assertIn('--gfn', command)
            self.assertEqual(command[command.index('--chrg')+1], '0')
            self.assertEqual(command[command.index('--uhf')+1], '0')
            self.assertGreater(kwargs['timeout'], 0)
            (work/'charges').write_text('\n'.join(['0.0']*mol.GetNumAtoms()))
            (work/'wbo').write_text('\n'.join(f'{b.GetBeginAtomIdx()+1} {b.GetEndAtomIdx()+1} 0.9' for b in mol.GetBonds()))
            (work/'scratch').mkdir()
            (work/'scratch'/'restart').write_text('temporary')
            return subprocess.CompletedProcess(command, 0, '* xtb version fake-test\n', '')
        # Exercise paths containing spaces as well as nested scratch cleanup.
        temp_directory = tempfile.TemporaryDirectory
        with temp_directory(prefix='pamf path with spaces ') as parent:
            with patch('pamf.electronic.tempfile.TemporaryDirectory',
                       side_effect=lambda **kw: temp_directory(dir=parent, **kw)), \
                    patch('pamf.electronic._resolve_xtb', return_value='xtb'), \
                    patch('pamf.electronic.subprocess.run', side_effect=fake_run):
                data = run_xtb(mol, ids[0])
            self.assertTrue(workdirs)
            self.assertTrue(all(not work.exists() for work in workdirs))
        self.assertEqual(len(data['charges']), mol.GetNumAtoms())
        self.assertEqual(data['wbo'][(0, 1)], 0.9)

    def test_xtb_custom_executable(self):
        mol, ids, _ = conformers(parse_smiles('CCC'), count=1)
        with tempfile.TemporaryDirectory() as folder:
            executable = Path(folder) / 'custom xtb.exe'
            executable.write_bytes(b'placeholder')

            def fake_run(command, **kwargs):
                work = Path(kwargs['cwd'])
                self.assertEqual(Path(command[0]), executable.resolve())
                (work/'charges').write_text('\n'.join(['0.0']*mol.GetNumAtoms()))
                (work/'wbo').write_text('\n'.join(
                    f'{b.GetBeginAtomIdx()+1} {b.GetEndAtomIdx()+1} 0.9' for b in mol.GetBonds()))
                return subprocess.CompletedProcess(command, 0, '', '')

            with patch('pamf.electronic.subprocess.run', side_effect=fake_run):
                run_xtb(mol, ids[0], executable=executable)

        with self.assertRaisesRegex(FileNotFoundError, 'PAMFConfig'):
            run_xtb(mol, ids[0], executable=Path(folder)/'missing-xtb.exe')

    def test_xtb_discovery_after_path_update(self):
        from .electronic import _resolve_xtb
        with tempfile.TemporaryDirectory(prefix='xtb installed ') as folder:
            executable = Path(folder) / ('xtb.exe' if os.name == 'nt' else 'xtb')
            executable.write_bytes(b'placeholder')
            executable.chmod(0o755)
            with patch.dict(os.environ, {'PATH': ''}), \
                    patch('pamf.electronic._windows_registered_path', return_value=folder):
                self.assertEqual(_resolve_xtb('xtb'), str(executable.resolve()))

    def test_full_pipeline_with_mock_electronics(self):
        def provider(mol, *args, **kwargs):
            return dict(wbo={tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))): 0.9+0.001*b.GetIdx()
                             for b in mol.GetBonds()}, charges=[0.0]*mol.GetNumAtoms(),
                        polarizabilities=[None]*mol.GetNumAtoms(), metadata={'method': 'mock-test'})
        with patch('pamf.pipeline.run_xtb', side_effect=provider):
            result = decompose_smiles('CCCOCCC', PAMFConfig(conformer_count=3))
        self.assertTrue(result['reconstruction_valid'])
        self.assertEqual(result['electronics']['parent_to_xyz_index'], list(range(1, 8)))
        self.assertIsNotNone(result['geometry'])
        self.assertTrue(all('wbo' in r for r in result['candidates']))

    def test_failures_do_not_fallback(self):
        with patch('pamf.electronic.subprocess.run', side_effect=FileNotFoundError('xtb')):
            with self.assertRaises(FileNotFoundError):
                decompose_smiles('CCCCCC', PAMFConfig(conformer_count=1))
        mol, ids, _ = conformers(parse_smiles('CCC'), count=1)
        for failure in ('timeout', 'failed', 'missing', 'parse'):
            workdirs = []
            def fake_run(command, **kwargs):
                work = Path(kwargs['cwd'])
                workdirs.append(work)
                (work/'restart').write_text('temporary')
                if failure == 'timeout':
                    raise subprocess.TimeoutExpired(command, 1)
                if failure == 'parse':
                    (work/'wbo').write_text('malformed')
                    (work/'charges').write_text('0')
                return subprocess.CompletedProcess(command, int(failure == 'failed'), '', 'failed')
            with self.subTest(failure=failure), patch('pamf.electronic.subprocess.run', side_effect=fake_run):
                with self.assertRaises((RuntimeError, ValueError)):
                    run_xtb(mol, ids[0])
                self.assertTrue(workdirs)
                self.assertTrue(all(not work.exists() for work in workdirs))

    def test_input_contracts(self):
        with self.assertRaisesRegex(ValueError, 'connected molecule'):
            decompose_smiles('CCCCCC.[Cl-]')
        for kwargs in ({'mode': 'auto'}, {'min_heavy_atoms': 0}, {'max_heavy_atoms': 2},
                       {'wbo_weight': float('nan')}, {'conformer_count': 0}, {'xtb_timeout': 0}):
            with self.assertRaises(ValueError):
                PAMFConfig(**kwargs)
        mol = parse_smiles('CCCCCC')
        candidates, _ = analyze_bonds(mol, ('[C]-[C]',))
        self.assertFalse(candidates)

    @unittest.skipUnless(os.environ.get('PAMF_RUN_XTB') == '1', 'Opt-in real xTB test; set PAMF_RUN_XTB=1')
    def test_real_xtb(self):
        result = decompose_smiles('CCCOCCC', PAMFConfig(conformer_count=3))
        self.assertEqual(result['electronics']['method'], 'GFN2-xTB')
        self.assertTrue(result['reconstruction_valid'])


class CLITests(unittest.TestCase):
    def cli(self, *args):
        return subprocess.run([sys.executable, '-m', 'pamf', *map(str, args)],
                              capture_output=True, text=True, timeout=30)

    def test_single_and_batch(self):
        result = self.cli('--smiles', 'CCCCCC', '--mode', 'rules')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)['fragments'], ['[1*]CCC', '[1*]CCC'])
        with tempfile.TemporaryDirectory(prefix='pamf-cli-test-') as directory:
            source, output = Path(directory)/'input.smi', Path(directory)/'out.jsonl'
            source.write_text('CCCCCC\n\nCCCOCCC\n')
            result = self.cli('--input', source, '--output', output, '--mode', 'rules')
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(len(output.read_text().splitlines()), 2)
            original_output = output.read_text()
            source.write_text('CCCCCC\nINVALID\n')
            result = self.cli('--input', source, '--output', output, '--mode', 'rules')
            self.assertEqual(result.returncode, 2)
            self.assertIn('Input line 2', result.stderr)
            self.assertEqual(output.read_text(), original_output)

    def test_reference_fit(self):
        with tempfile.TemporaryDirectory(prefix='pamf-ref-test-') as directory:
            source, output = Path(directory)/'train.jsonl', Path(directory)/'reference.json'
            rows = [dict(chemical_class='C(SP3)-C(SP3)', wbo=v, cross_wbo=v+0.1) for v in (0.8, 1.0)]
            source.write_text(json.dumps(dict(candidates=rows))+'\n')
            result = self.cli('--fit-reference', source, '--output', output)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertAlmostEqual(json.loads(output.read_text())['C(SP3)-C(SP3)']['wbo']['mean'], 0.9)


if __name__ == '__main__':
    unittest.main()
