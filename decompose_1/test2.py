"""
第二步修复版：为骨架和侧链添加虚拟原子标记
修复虚拟原子的连接位置问题
"""

from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem.rdchem import BondType, Atom

# 导入第一步的函数
from decompose_1.test1 import decompose_to_scaffold_and_side_chains, bond_type_to_str


def create_virtual_atom(label: str, bond_type: str) -> Atom:
    """创建带属性的虚拟原子"""
    virt_atom = Atom(0)  # 原子序数0表示虚拟原子
    virt_atom.SetProp("_VirtualLabel", label)
    virt_atom.SetProp("_BondType", bond_type)
    virt_atom.SetIsotope(0)
    return virt_atom


def extract_substructure_with_virtual_atoms(original_mol: Chem.Mol,
                                            atom_indices: Set[int],
                                            connections: List[Tuple[int, str, str]]) -> Tuple[Optional[Chem.Mol], str]:
    """
    提取子结构并添加虚拟原子标记

    策略：
    1. 保留子结构中的所有原子
    2. 为每个连接点添加一个虚拟原子
    3. 虚拟原子连接到对应的原原子上

    Args:
        original_mol: 原始分子
        atom_indices: 子结构包含的原子索引
        connections: 连接信息 [(原子索引, 虚拟标签, 键类型), ...]
                     注意：同一个原子可以有多个连接

    Returns:
        带有虚拟原子的子结构分子
    """
    if not atom_indices:
        return None

    # 排序原子索引以保持一致性
    sorted_atoms = sorted(atom_indices)

    # 创建原子索引映射（原索引 -> 新分子中的索引）
    idx_map = {orig: new for new, orig in enumerate(sorted_atoms)}

    # 创建新分子
    rw_mol = Chem.RWMol()

    # 1. 添加子结构中的所有原子
    for orig_idx in sorted_atoms:
        orig_atom = original_mol.GetAtomWithIdx(orig_idx)
        new_atom = Chem.Atom(orig_atom.GetAtomicNum())
        new_atom.SetFormalCharge(orig_atom.GetFormalCharge())
        new_atom.SetIsAromatic(orig_atom.GetIsAromatic())
        new_atom.SetNumExplicitHs(orig_atom.GetNumExplicitHs())
        # 复制其他属性
        if orig_atom.HasProp("_VirtualLabel"):
            new_atom.SetProp("_VirtualLabel", orig_atom.GetProp("_VirtualLabel"))
        rw_mol.AddAtom(new_atom)

    # 2. 添加子结构内部的键
    for orig_idx1 in sorted_atoms:
        for orig_idx2 in sorted_atoms:
            if orig_idx1 < orig_idx2:
                bond = original_mol.GetBondBetweenAtoms(orig_idx1, orig_idx2)
                if bond is not None:
                    rw_mol.AddBond(idx_map[orig_idx1], idx_map[orig_idx2], bond.GetBondType())

    smiles_wo_conn = Chem.MolToSmiles(rw_mol.GetMol(), canonical=True)

    # 3. 为每个连接点添加虚拟原子
    # 按原原子分组连接，以便同一原子添加多个虚拟原子
    connections_by_atom = defaultdict(list)
    for orig_idx, virt_label, bond_type_str in connections:
        if orig_idx in idx_map:
            connections_by_atom[orig_idx].append((virt_label, bond_type_str))

    # 键类型映射
    bond_type_map = {
        "-": BondType.SINGLE,
        "=": BondType.DOUBLE,
        "#": BondType.TRIPLE,
        ":": BondType.AROMATIC
    }

    # 为每个连接点添加虚拟原子
    for orig_idx, conn_list in connections_by_atom.items():
        target_idx = idx_map[orig_idx]

        for virt_label, bond_type_str in conn_list:
            # 创建虚拟原子（原子序数0）
            virt_atom = Chem.Atom(0)
            virt_atom.SetProp("_VirtualLabel", virt_label)
            virt_atom.SetIsotope(int(virt_label))

            # 添加虚拟原子
            virt_idx = rw_mol.AddAtom(virt_atom)

            # 在原原子和虚拟原子之间添加键
            bond_type = bond_type_map.get(bond_type_str, BondType.SINGLE)
            rw_mol.AddBond(target_idx, virt_idx, bond_type)

    # 4. 清理并返回
    try:
        rw_mol.UpdatePropertyCache()
        # 尝试 sanitize，但虚拟原子可能会导致问题，所以捕获异常
        try:
            Chem.SanitizeMol(rw_mol)
        except:
            # 如果 sanitize 失败，仍然返回分子（虚拟原子可能导致问题）
            pass
    except:
        pass

    return rw_mol.GetMol(), smiles_wo_conn


def add_virtual_connections(decomposition_result: Dict) -> Dict:
    """
    为骨架和侧链添加虚拟连接信息

    Args:
        decomposition_result: decompose_to_scaffold_and_side_chains 的返回结果

    Returns:
        添加了连接信息的增强结果
    """
    mol = decomposition_result['original_mol']
    scaffold_atoms = decomposition_result['scaffold_atoms']
    side_chain_components = decomposition_result['side_chain_components']
    attachment_points = decomposition_result['attachment_points']

    # 构建原子到组件的映射
    atom_to_component = {}

    # 骨架作为一个组件
    for atom in scaffold_atoms:
        atom_to_component[atom] = 'scaffold'

    # 每个侧链连通分量作为一个独立组件
    component_id = 1
    component_atoms = {}
    for comp in side_chain_components:
        comp_name = f'sidechain_{component_id}'
        component_atoms[comp_name] = set(comp)
        for atom in comp:
            atom_to_component[atom] = comp_name
        component_id += 1

    # 为每对连接的组件分配连接ID
    connection_id = 0
    component_pairs = {}  # (comp1, comp2) -> connection_id

    # 存储每个组件的连接信息
    component_connections = defaultdict(list)  # component_name -> list of (atom_idx, virt_label, bond_type)

    # 遍历所有连接点
    for ap in attachment_points:
        scaffold_atom = ap['scaffold_atom']
        side_atom = ap['side_chain_atom']
        bond_type = ap['bond_type']

        comp1 = atom_to_component.get(scaffold_atom)
        comp2 = atom_to_component.get(side_atom)

        if comp1 and comp2 and comp1 != comp2:
            # 创建组件对键
            pair_key = tuple(sorted([comp1, comp2]))

            if pair_key not in component_pairs:
                connection_id += 1
                component_pairs[pair_key] = connection_id

            conn_id = component_pairs[pair_key]
            virt_label = f"{conn_id}"

            # 记录连接信息
            component_connections[comp1].append((scaffold_atom, virt_label, bond_type))
            component_connections[comp2].append((side_atom, virt_label, bond_type))

    # 生成带虚拟原子的子结构
    result = {
        'original_mol': mol,
        'scaffold': None,
        'side_chains': [],
        'connections': dict(component_connections)
    }

    # 处理骨架
    if scaffold_atoms:
        scaffold_connections = component_connections.get('scaffold', [])
        scaffold_mol, _ = extract_substructure_with_virtual_atoms(
            mol, scaffold_atoms, scaffold_connections
        )
        if scaffold_mol and scaffold_mol.GetNumAtoms() > 0:
            result['scaffold'] = {
                'atoms': list(scaffold_atoms),
                'smiles': Chem.MolToSmiles(scaffold_mol),
                'connections': scaffold_connections
            }

    # 处理每个侧链分量
    for comp_name, comp_atoms in component_atoms.items():
        if comp_atoms:
            side_connections = component_connections.get(comp_name, [])
            side_mol, _ = extract_substructure_with_virtual_atoms(
                mol, comp_atoms, side_connections
            )
            if side_mol and side_mol.GetNumAtoms() > 0:
                result['side_chains'].append({
                    'name': comp_name,
                    'atoms': list(comp_atoms),
                    'smiles': Chem.MolToSmiles(side_mol),
                    'connections': side_connections
                })

    result['max_conneciont_id'] = connection_id
    return result


def extract_side_chain_with_correct_virtual_atom(original_mol: Chem.Mol,
                                                  side_atoms: Set[int],
                                                  attachment_info: Tuple[int, str, str]) -> Optional[Chem.Mol]:
    """
    专门为侧链提取带正确虚拟原子的分子

    Args:
        original_mol: 原始分子
        side_atoms: 侧链原子集合
        attachment_info: (连接点原子索引, 虚拟标签, 键类型)
    """
    if not side_atoms:
        return None

    # 创建原子映射
    sorted_atoms = sorted(side_atoms)
    idx_map = {orig: new for new, orig in enumerate(sorted_atoms)}

    # 创建新分子
    rw_mol = Chem.RWMol()

    # 添加原子
    for orig_idx in sorted_atoms:
        orig_atom = original_mol.GetAtomWithIdx(orig_idx)
        new_atom = Chem.Atom(orig_atom.GetAtomicNum())
        new_atom.SetFormalCharge(orig_atom.GetFormalCharge())
        new_atom.SetIsAromatic(orig_atom.GetIsAromatic())
        rw_mol.AddAtom(new_atom)

    # 添加内部键
    for orig_idx1 in sorted_atoms:
        for orig_idx2 in sorted_atoms:
            if orig_idx1 < orig_idx2:
                bond = original_mol.GetBondBetweenAtoms(orig_idx1, orig_idx2)
                if bond:
                    rw_mol.AddBond(idx_map[orig_idx1], idx_map[orig_idx2], bond.GetBondType())

    # 处理连接点
    if attachment_info:
        attach_atom, virt_label, bond_type_str = attachment_info

        if attach_atom in idx_map:
            attach_idx = idx_map[attach_atom]

            # 获取该原子的内部邻居
            orig_atom = original_mol.GetAtomWithIdx(attach_atom)
            internal_neighbors = []

            for neighbor in orig_atom.GetNeighbors():
                n_idx = neighbor.GetIdx()
                if n_idx in side_atoms:
                    bond = original_mol.GetBondBetweenAtoms(attach_atom, n_idx)
                    internal_neighbors.append((idx_map[n_idx], bond.GetBondType()))

            # 删除原原子
            rw_mol.RemoveAtom(attach_idx)

            # 创建虚拟原子并添加
            virt_atom = Chem.Atom(0)
            virt_atom.SetProp("_VirtualLabel", virt_label)
            virt_idx = rw_mol.AddAtom(virt_atom)

            # 连接虚拟原子到内部邻居
            for nbr_idx, nbr_bond_type in internal_neighbors:
                if nbr_idx > attach_idx:
                    nbr_idx -= 1
                if nbr_idx < rw_mol.GetNumAtoms():
                    rw_mol.AddBond(nbr_idx, virt_idx, nbr_bond_type)

    try:
        rw_mol.UpdatePropertyCache()
        Chem.SanitizeMol(rw_mol)
    except:
        pass

    return rw_mol.GetMol()




if __name__ == "__main__":
    print("RDKit 版本:", Chem.rdBase.rdkitVersion)