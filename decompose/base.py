import os
import multiprocessing as mp
from rdkit.Chem import MACCSkeys
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import Recap
from decompose.molDecom import _get_prop
from rdkit import RDConfig, Chem
import re

from constant import *
from utils import _process_list_parallel
from decompose.molDecom import decompose_smiles
from decompose.SMILES2FST import VirtualAtomConnectionProcessor
from tqdm import tqdm
from decompose_1.test3 import decompose_smiles_list
from rdkit.Contrib.SA_Score import sascorer
from decompose_1 import test3

def maccs_from_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        mol = Chem.MolFromSmarts(smiles)
        if mol is None:
            return None
    fp = MACCSkeys.GenMACCSKeys(mol)
    return fp

def ecfp_from_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        mol = Chem.MolFromSmarts(smiles)
        if mol is None:
            return None

    try:
        mol = Chem.RemoveHs(mol)
        fp = rdFingerprintGenerator.GetMorganGenerator(radius=4)
        ecfp = fp.GetFingerprint(mol)
    except:
        ecfp = None

    return ecfp


def mol_fragment_recap(mol):
    hierarchy = Recap.RecapDecompose(mol)
    # 获取叶子节点（最终片段）
    leaves = hierarchy.GetLeaves()
    return leaves

def sdf_to_mol(sdf=r'D:/3D structures.sdf'):
    mol = Chem.SDMolSupplier(sdf)

def _mol_decomp(smiles):
    return decompose_smiles(smiles, func_group_list, link_index=True, prop_calu=False)

def _mol_decomp_with_prop(smiles):
    return decompose_smiles(smiles, func_group_list, link_index=True, prop_calu=True)

def _mol_decomp_with_prop1(smiles):
    return decompose_smiles_list(smiles)

def _pamf_frag_records(smiles, n_core, config=None, reference=None, prop_calu=True):
    """Adapt ordered PAMF fragments to the legacy singleton-record protocol."""
    from pamf import fragment_smiles_batch
    batches, indices = fragment_smiles_batch(smiles, config, workers=n_core,
                                             return_format='list', reference=reference,
                                             return_indices=True)
    records = [None] * len(smiles)
    jobs = ((index, smiles[index], fragments, prop_calu)
            for index, fragments in zip(indices, batches))
    def collect(results):
        for index, record, error in tqdm(results, total=len(indices),
                                          desc='PAMF record conversion', unit='mol'):
            records[index] = record
            if error is not None:
                tqdm.write(f'PAMF skipped input[{index}] {smiles[index]!r} during adaptation: {error}')
    if n_core == 1 or not indices:
        collect(map(_pamf_one_record_safe, jobs))
    else:
        with mp.get_context('spawn').Pool(processes=min(n_core, mp.cpu_count(), len(indices))) as pool:
            collect(pool.imap_unordered(_pamf_one_record_safe, jobs, chunksize=1))
    return records


def _pamf_one_record_safe(job):
    """Picklable worker; only the parent updates aligned records and progress."""
    index, original, fragments, prop_calu = job
    try:
        return index, _pamf_one_record(original, fragments, prop_calu), None
    except Exception as exc:
        return index, None, f'{type(exc).__name__}: {exc}'


def _pamf_statistics_smiles(mol):
    """Canonical independent components after deleting dummy attachment atoms.

    Work on a copy so attachment IDs used by tokenization/PKL remain intact.
    Kekulize before deletion, then sanitize to recompute hydrogens/aromaticity.
    """
    clean = Chem.RWMol(mol)
    Chem.Kekulize(clean, True)
    for atom in clean.GetAtoms():
        atom.SetAtomMapNum(0)
    for index in reversed([a.GetIdx() for a in clean.GetAtoms()
                           if a.GetAtomicNum() == 0]):
        clean.RemoveAtom(index)
    clean = clean.GetMol()
    Chem.SanitizeMol(clean)
    return [Chem.MolToSmiles(part, canonical=True, isomericSmiles=True)
            for part in Chem.GetMolFrags(clean, asMols=True)]


def _pamf_one_record(original, fragments, prop_calu):
    rows = []
    for structure in fragments:
        mol = Chem.MolFromSmiles(structure)
        if mol is None:
            raise ValueError(f'Invalid PAMF fragment: {structure!r}')
        # Preserve attachment isotopes in raw_mol. Rank a separate copy
        # without attachment IDs so arbitrary cut numbering is not symmetry.
        unlabelled = Chem.Mol(mol)
        for atom in unlabelled.GetAtoms():
            atom.SetAtomMapNum(0)
            if atom.GetAtomicNum() == 0:
                atom.SetIsotope(0)
        ranks = Chem.CanonicalRankAtoms(unlabelled, breakTies=False)
        rows.append(dict(smiles=structure, raw_mol=mol,
                         raw_mol_props={i: {'_symmetry': int(rank)}
                                        for i, rank in enumerate(ranks)},
                         type='pamf', smiles_wo_index=Chem.MolToSmiles(unlabelled),
                         statistics_smiles=_pamf_statistics_smiles(mol)))
    return [dict(original_smiles=original, fragments=rows,
                 prop=_get_prop(Chem.MolFromSmiles(original)) if prop_calu else None)]


def _pamf_tokens_safe(frags):
    """One molecule per job; catch conversion errors before they escape the pool."""
    try:
        return _mol_data([_mol_decom_frag_decom(frags)])
    except Exception as exc:
        return {'error': str(exc)}


def _mol_decom_mp(smiles, n_core, properties=None, statistic_only=False, version=1,
                  *, method='legacy', pamf_config=None, pamf_reference=None,
                  return_fragment_smiles=False, statistics_path='stru_data.json'):
    """Select legacy (version 0/1) or pamf; all returned rows follow input order.

    PAMF failures are skipped; properties are filtered by the same input indices.
    Set return_fragment_smiles to append an aligned list of fragment SMILES lists.
    statistics_path selects the JSON output; None disables writing statistics.
    """
    smiles = list(smiles)
    if n_core < 1:
        raise ValueError('n_core must be a positive integer')
    if method not in ('legacy', 'pamf'):
        raise ValueError('method must be legacy or pamf')
    if properties is not None and len(properties) != len(smiles):
        raise ValueError('properties must have the same length as smiles')
    print('decomposing molecules...')

    results = []
    accepted_records = []
    fragment_smiles = []
    oring = []
    formula = []
    frags_stat = {}
    frags_type = {}
    prop_calu = False
    if properties is not None and len(properties) == len(smiles):
        props = []
    else:
        props = []
        prop_calu = True

    if method == 'pamf':
        frags = _pamf_frag_records(smiles, n_core, pamf_config, pamf_reference, prop_calu)
    elif version == 1:
        pf = _mol_decomp_with_prop1
        ff = _mol_decomp_with_prop1
    elif version == 0:
        pf = _mol_decomp_with_prop
        ff = _mol_decomp
    else:
        raise ValueError('legacy version must be 0 or 1')

    if method == 'legacy' and prop_calu:
        frags = _process_list_parallel(smiles, num_cores=n_core, process_func=pf)
    elif method == 'legacy':
        frags = _process_list_parallel(smiles, num_cores=n_core, process_func=ff)

    if len(frags) != len(smiles):
        raise ValueError('Decomposition output length does not match input')

    ii = 0
    print('processing frag data...')
    for input_index, frag in enumerate(tqdm(frags)):
        if method == 'pamf' and frag is None:
            continue
        try:
            tmp = []
            molecule_fragment_smiles = []
            frag = frag[0]
            original_smiles = smiles[input_index]
            if frag['original_smiles'] != original_smiles:
                raise ValueError('Decomposition output order does not match input')

            flag = True
            for t in frag['fragments']:
                structure = t['smiles']
                raw_mol = t['raw_mol']

                if 'raw_mol_props' in t:
                    raw_mol_props = t['raw_mol_props']
                else:
                    raw_mol_props = None

                if 'J' in structure:
                    flag = False
                    break
                    # if smiles['type'] == 'ring_system' and not full_ring:
                    #     structure = short_ring(structure)
                tmp.append((raw_mol, raw_mol_props))
                molecule_fragment_smiles.append(Chem.MolToSmiles(raw_mol, isomericSmiles=True))
            if not flag:
                raise ValueError('Unsupported J fragment')
        except Exception as e:
            if method == 'pamf':
                tqdm.write(f'PAMF skipped input[{input_index}] {smiles[input_index]!r}: {e}')
                continue
            raise ValueError(f'Decomposition input[{input_index}] {smiles[input_index]!r}: {e}') from e

        results.append(tmp)
        accepted_records.append(frag)
        fragment_smiles.append(molecule_fragment_smiles)
        oring.append(original_smiles)
        props.append(frag['prop'] if properties is None else properties[input_index])
        pass

    import json
    frags_type = _fragment_statistics(accepted_records)
    _save_fragment_statistics(frags_type, statistics_path)
    if statistic_only:
        output = (None, None, None, None, frags_type)
        return output + (None,) if return_fragment_smiles else output
    else:
        print('advance molecular decomposition')
        if method == 'pamf':
            converted = _process_list_parallel(results, num_cores=n_core,
                                               process_func=_pamf_tokens_safe) if results else []
            kept = []
            sentences, results2 = [], []
            for index, row in enumerate(converted):
                if isinstance(row, dict) and 'error' in row:
                    tqdm.write(f'PAMF skipped {oring[index]!r} during token conversion: {row["error"]}')
                    continue
                kept.append(index)
                sentences.append(row[0])
                results2.append(row[1])
            frags_type = _fragment_statistics([accepted_records[i] for i in kept])
            _save_fragment_statistics(frags_type, statistics_path)
            output = (sentences, results2, [oring[i] for i in kept], [props[i] for i in kept], frags_type)
            return output + ([fragment_smiles[i] for i in kept],) if return_fragment_smiles else output
        import constant
        index = 0
        mols = _process_list_parallel(results, num_cores=n_core, process_func=_mol_decom_frag_decom)
        results2 = []

        count = len(frags)
        sentences = []
        print('processing mol data...')
        XX = _process_list_parallel(mols, num_cores=n_core, process_func=_mol_data)
        if len(mols) != len(smiles) or len(XX) != len(smiles):
            raise ValueError('Token conversion output length does not match input')
        for xx in XX:
            sentences.append(xx[0])
            results2.append(xx[1])

    print('molecule decomposing done')
    output = (sentences, results2, oring, props, frags_type)
    return output + (fragment_smiles,) if return_fragment_smiles else output

def _save_fragment_statistics(statistics, path):
    if path is None:
        return
    import json
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(statistics, handle, ensure_ascii=False, indent=4)


def _fragment_statistics(records):
    counts = {}
    for record in records:
        for fragment in record['fragments']:
            group = counts.setdefault(fragment['type'], {})
            structures = (fragment['statistics_smiles'] if fragment['type'] == 'pamf'
                          else [fragment['smiles_wo_index']])
            for smiles in structures:
                group[smiles] = group.get(smiles, 0) + 1
    return counts


def _mol_data(mol):
    mol = mol
    tmp = []
    tmp.append(START_TOKEN)
    sentence = f'{START_TOKEN} '

    for frags in mol[0][0]:
        for frag in frags[1]:
            # s = _remove_H_protect_elements(frag)
            s = _remove_colon_and_digits(frag)
            sentence += (s + ' ')
            tmp.append(s)
            # sentence += (constant.SEP_TOKEN + '\t')
    sentence += f'{EOS_TOKEN}'
    tmp.append(EOS_TOKEN)

    return (sentence, tmp)

def _mol_decom_frag_decom(frags):
    results = []
    processor = VirtualAtomConnectionProcessor()
    for frag in frags:
        result = []
        for i in frag:
            ddd = processor.format_final_output(processor.process_mol(i[0], i[1]))
            if ddd is not None:
                result.append(ddd)
            else:
                raise ValueError('Fragment token conversion failed')
        results.append(result)

    return results

def _remove_colon_and_digits(text):

    result = re.sub(r':\d+', '', text)
    result = re.sub(r':(?=\D|$)', '', result)

    return result

def _remove_H_protect_elements(text):
    """
    删除字符串中的H字符，但保护其他以H开头的元素符号
    - 如果H后面是数字，同时删除H和这个数字
    - 如果H是单独的或后面不是元素符号，只删除H
    - 保护He, Hf, Hg, Ho, Hs等元素
    """
    # 定义以H开头的元素符号
    H_elements = ['He', 'Hf', 'Hg', 'Ho', 'Hs']

    # 先保护其他H元素，将它们替换为临时标记
    protected_text = text
    temp_markers = {}

    for i, element in enumerate(H_elements):
        marker = f'__ELEMENT_{i}__'
        temp_markers[marker] = element
        protected_text = protected_text.replace(element, marker)

    # 现在处理剩余的H字符
    # 删除H后面跟着数字的情况
    result = re.sub(r'H\d', '', protected_text)
    # 删除单独的H字符
    result = re.sub(r'H', '', result)

    # 恢复保护的元素符号
    for marker, element in temp_markers.items():
        result = result.replace(marker, element)

    return result

def _mol_decom_mp_to_pkl_file(sentences, oring, props, protein=None, pocket=None, pkl_path='gpt/frag_file/frag.pkl',
                              *, fragment_smiles=None):
    import pickle
    gpt_folder = "gpt"
    frag_file_path = os.path.join(gpt_folder, "frag_file")
    if not os.path.exists(frag_file_path):
        os.makedirs(frag_file_path)
        print(f"创建文件夹: {frag_file_path}")
    else:
        pass

    print('processing pkl file...')
    result = {}
    if protein is None:
        protein = [-1] * len(sentences)
    if props is None:
        props = [None] * len(sentences)
    if oring is None:
        oring = [None] * len(sentences)
    if pocket is None:
        pocket = [None] * len(sentences)
    if fragment_smiles is None:
        fragment_smiles = [None] * len(sentences)
    for name, rows in [('oring', oring), ('props', props), ('protein', protein),
                       ('pocket', pocket), ('fragment_smiles', fragment_smiles)]:
        if len(rows) != len(sentences):
            raise ValueError(f'{name} length must match sentences')
    for i in tqdm(range(len(sentences))):
        result[i] = {}
        try:
            result[i]['oring'] = oring[i]
            result[i]['props'] = props[i]
            result[i]['frag'] = sentences[i]
            result[i]['fragment_smiles'] = fragment_smiles[i]
            result[i]['protein'] = protein[i]
            result[i]['pocket'] = pocket[i]
        except Exception as e:
            print('data error')
            return
    print('writing pkl file...')
    result2 = {}
    result2['mol'] = result
    result2['protein_dict'] = [-1] * len(sentences)
    with open(pkl_path, 'wb') as handle:
        pickle.dump(result2, handle)

    return result

def _re_calculate_prop_by_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles[0])
    props = _get_prop(mol)
    return props

if "__main__" == __name__:
    x = _mol_decom_mp(['COC(=O)Cc1csc(NC(=O)Cc2coc3cc(C)ccc23)n1'],1)
    print()
