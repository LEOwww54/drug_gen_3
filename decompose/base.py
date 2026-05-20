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
from decompose.molDecom_2 import VirtualAtomConnectionProcessor
from tqdm import tqdm

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

def _mol_decom_mp(smiles, n_core, properties=None, statistic_only=False):
    print('decomposing molecules...')

    results = []
    oring = []
    formula = []
    frags_stat = {}
    prop_calu = False
    if properties is not None and len(properties) == len(smiles):
        props = properties
    else:
        props = []
        prop_calu = True

    if prop_calu:
        frags = _process_list_parallel(smiles, num_cores=n_core, process_func=_mol_decomp_with_prop)
    else:
        frags = _process_list_parallel(smiles, num_cores=n_core, process_func=_mol_decomp)

    ii = 0
    print('processing frag data...')
    for frag in tqdm(frags):
        try:
            tmp = []
            frag = frag[0]
            os = frag['original_smiles']

            flag = True
            for smiles in frag['fragments']:
                structure = smiles['smiles']

                o = smiles['smiles_wo_index']
                if o not in frags_stat:
                    frags_stat[o] = 1
                else:
                    frags_stat[o] += 1

                if 'J' in structure:
                    flag = False
                    break
                    # if smiles['type'] == 'ring_system' and not full_ring:
                    #     structure = short_ring(structure)
                tmp.append(structure)
            if not flag:
                continue
        except Exception as e:
            continue

        results.append(tmp)
        oring.append(os)
        formula.append(frag['formula'])
        if properties is None:
            props.append(frag['prop'])
        pass

    if statistic_only:
        return None, None, None, None, frags_stat
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
        for xx in XX:
            sentences.append(xx[0])
            results2.append(xx[1])

    print('molecule decomposing done')
    return sentences, results2, oring, props, frags_stat

def _mol_data(mol):
    mol = mol
    tmp = []
    tmp.append(START_TOKEN)
    sentence = f'{START_TOKEN} '

    for frags in mol[0][0]:
        for frag in frags[1]:
            s = _remove_H_protect_elements(frag)
            s = _remove_colon_and_digits(s)
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
            ddd = processor.format_final_output(processor.process_smiles(i))
            if ddd is not None:
                result.append(ddd)
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
    _mol_decom_mp(["CC(=O)O[C@H]1[C@H]2[C@@]([C@H]3[C@@]([C@]4(C[C@@H]5[C@]6(C[C@@H](C(=C([C@@H](O6)C(=O)[C@]5(C4=C(C3=O)C)OC(=O)C)OC(=O)c7ccccc7)O)C)OC(=O)C)O2)OC(=O)c8ccccc8)(C1(C)C)OC(=O)C"],1)