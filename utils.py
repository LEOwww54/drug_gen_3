from typing import List, Any, Callable
import multiprocessing as mp
from tqdm import tqdm
from rdkit import Chem

def _process_list_parallel(
        input_list: List[Any],
        num_cores: int,
        process_func: Callable
):
    """
    使用多进程处理列表

    Args:
        input_list: 输入列表
        num_cores: 使用的核心数
        process_func: 处理函数

    Returns:
        按输入顺序排列的结果列表
    """
    # 限制核心数不超过CPU数量
    num_cores = min(num_cores, mp.cpu_count())

    # 使用进程池
    with mp.Pool(processes=num_cores) as pool:
        results = [pool.apply_async(process_func, args=([x],)) for x in input_list]
        print('process running...')
        results = [p.get() for p in tqdm(results)]
        print('all process done')

    return results


def extract_submol(mol, atom_indices) -> tuple[Chem.Mol, dict[int, int], dict[int, int]]:
    """
    从原始分子中提取包含指定原子的子分子。

    Args:
        mol: RDKit 分子对象 (rdkit.Chem.rdchem.Mol)
        atom_indices: 要保留的原子索引列表 (list of ints)

    Returns:
        rdkit.Chem.rdchem.Mol: 提取出的子分子对象，如果失败则返回 None
    """
    # 参数校验
    if not mol or not atom_indices:
        return None

    # 1. 创建一个可编辑的分子对象
    em = Chem.EditableMol(Chem.Mol())

    # 2. 添加原子
    # 创建一个映射表：原原子索引 -> 新原子索引
    index_map = {}
    new_2_old = {}
    for idx in atom_indices:
        atom = mol.GetAtomWithIdx(idx)
        # 深拷贝原子，保留其所有属性（如手性、形式电荷等）
        new_atom = Chem.Atom(atom.GetSymbol())
        new_atom.SetFormalCharge(atom.GetFormalCharge())
        # 可根据需要复制更多属性，如 chiral tag, isotope 等
        # new_atom.SetChiralTag(atom.GetChiralTag())
        i = em.AddAtom(new_atom)
        index_map[idx] = i
        new_2_old[i] = idx

    # 3. 添加键
    # 使用 bond set 避免重复添加相同的键
    added_bonds = set()
    # 遍历所有被选中的原子
    for idx in atom_indices:
        atom = mol.GetAtomWithIdx(idx)
        # 检查该原子的所有邻居
        for nbr in atom.GetNeighbors():
            nbr_idx = nbr.GetIdx()
            # 如果邻居原子也在要保留的集合中
            if nbr_idx in index_map:
                # 确保每条键只添加一次 (用排序后的 (idx1, idx2) 作为键)
                bond_key = tuple(sorted((idx, nbr_idx)))
                if bond_key not in added_bonds:
                    bond = mol.GetBondBetweenAtoms(idx, nbr_idx)
                    # 添加键，保留键的类型 (单键、双键等)
                    em.AddBond(index_map[idx], index_map[nbr_idx], bond.GetBondType())
                    added_bonds.add(bond_key)

    # 4. 获取最终分子并尝试进行标准化 (Sanitize)
    sub_mol = em.GetMol()
    try:
        # 标准化是必要的，它可以校正化合价并计算芳香性等，但有时对不完整的子结构会失败
        # 如果失败，可以尝试返回未标准化的分子或处理错误
        Chem.SanitizeMol(sub_mol)
    except Exception as e:
        print(f"Warning: Sanitization failed for the extracted sub-molecule: {e}")
        # 不抛出异常，返回未标准化的分子，通常仍可用

    return sub_mol, index_map, new_2_old