import os
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
    batches = fragment_smiles_batch(smiles, config, workers=n_core,
                                    return_format='list', reference=reference)
    if len(batches) != len(smiles):
        raise ValueError('PAMF output length does not match input')
    records = []
    for original, fragments in zip(smiles, batches):
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
                             type='pamf', smiles_wo_index=Chem.MolToSmiles(unlabelled)))
        records.append([dict(original_smiles=original, fragments=rows,
                             prop=_get_prop(Chem.MolFromSmiles(original)) if prop_calu else None)])
    return records


def _mol_decom_mp(smiles, n_core, properties=None, statistic_only=False, version=1,
                  *, method='legacy', pamf_config=None, pamf_reference=None):
    """Select legacy (version 0/1) or pamf; all returned rows follow input order.

    Any failed row raises instead of silently shifting molecules/properties.
    """
    smiles = list(smiles)
    if type(n_core) is not int or n_core < 1:
        raise ValueError('n_core must be a positive integer')
    if method not in ('legacy', 'pamf'):
        raise ValueError('method must be legacy or pamf')
    if properties is not None and len(properties) != len(smiles):
        raise ValueError('properties must have the same length as smiles')
    print('decomposing molecules...')

    results = []
    oring = []
    formula = []
    frags_stat = {}
    frags_type = {}
    prop_calu = False
    if properties is not None and len(properties) == len(smiles):
        props = list(properties)
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
        try:
            tmp = []
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

                type = t['type']
                if type not in frags_type:
                    frags_type[type] = {}

                o = t['smiles_wo_index']
                if o not in frags_type[type]:
                    frags_type[type][o] = 1
                else:
                    frags_type[type][o] += 1

                if 'J' in structure:
                    flag = False
                    break
                    # if smiles['type'] == 'ring_system' and not full_ring:
                    #     structure = short_ring(structure)
                tmp.append((raw_mol, raw_mol_props))
            if not flag:
                raise ValueError('Unsupported J fragment')
        except Exception as e:
            raise ValueError(f'Decomposition input[{input_index}] {smiles[input_index]!r}: {e}') from e

        results.append(tmp)
        oring.append(original_smiles)
        if properties is None:
            props.append(frag['prop'])
        pass

    import json
    with open("stru_data.json", "w", encoding="utf-8") as f:
        json.dump(frags_type, f, ensure_ascii=False, indent=4)
    if statistic_only:
        return None, None, None, None, frags_type
    else:
        print('advance molecular decomposition')
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
    return sentences, results2, oring, props, frags_type

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

def _mol_decom_mp_to_pkl_file(sentences, oring, props, protein=None, pocket=None, pkl_path='gpt/frag_file/frag.pkl'):
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
    for i in tqdm(range(len(sentences))):
        result[i] = {}
        try:
            result[i]['oring'] = oring[i]
            result[i]['props'] = props[i]
            result[i]['frag'] = sentences[i]
            result[i]['protein'] = protein[i]
            result[i]['pocket'] = pocket[i]
        except Exception as e:
            print('data error')
            return
    print('writing pkl file...')
    result2 = {}
    result2['mol'] = result
    result2['protein_dict'] = [-1] * len(sentences)
    pickle.dump(result2, open(pkl_path, 'wb'))

    return result

def _re_calculate_prop_by_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles[0])
    props = _get_prop(mol)
    return props

if "__main__" == __name__:
    x = _mol_decom_mp(['COC(=O)Cc1csc(NC(=O)Cc2coc3cc(C)ccc23)n1'],1)
    print()
