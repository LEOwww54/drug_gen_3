"""
第三步：完整的分子分解系统（修正版）
实现骨架内部和侧链内部的多层次分解
"""

from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Chem.Scaffolds import MurckoScaffold

import constant
from constant import func_group_list
# 导入之前的函数
from decompose_1.test1 import decompose_to_scaffold_and_side_chains, bond_type_to_str
from decompose_1.test2 import extract_substructure_with_virtual_atoms
from utils import extract_submol

class SubstructureType:
    """子结构类型常量"""
    RING_SYSTEM = "ring_system"           # 环系统（通用）
    RING_SYSTEM_SCAFFOLD = "ring_system_scaffold"  # 骨架上的环系统
    BRIDGE = "bridge"                     # 桥梁（通用）
    BRIDGE_SCAFFOLD = "bridge_scaffold"   # 骨架上的桥梁
    FUNCTIONAL = "functional"             # 功能基团
    RESIDUAL = "residual"                 # 残留结构
    LINKER = "linker"                     # 连接子
    SIDE_CHAIN = 'side_chain'


def find_maximum_ring_systems(mol: Chem.Mol) -> List[Set[int]]:
    """寻找分子中的最大环系统"""
    ring_info = mol.GetRingInfo()

    try:
        all_rings = []
        for ring in ring_info.AtomRings():
            all_rings.append(set(ring))
    except:
        return []

    if not all_rings:
        return []

    # 合并融合环
    merged_rings = []
    for ring in all_rings:
        found = False
        for existing in merged_rings:
            if existing.intersection(ring):
                existing.update(ring)
                found = True
                break
        if not found:
            merged_rings.append(set(ring))

    # 重复合并直到稳定
    changed = True
    while changed:
        changed = False
        new_merged = []
        for ring_set in merged_rings:
            found = False
            for existing in new_merged:
                if existing.intersection(ring_set):
                    existing.update(ring_set)
                    found = True
                    changed = True
                    break
            if not found:
                new_merged.append(ring_set)
        merged_rings = new_merged

    return merged_rings


def extract_functional_substructures(mol: Chem.Mol, used_atoms: Set[int]) -> List[Dict]:
    """提取化学功能子结构"""

    functional_groups = []

    for smarts in constant.func_group_list:
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is None:
            continue

        matches = mol.GetSubstructMatches(pattern)
        for match in matches:
            match_set = set(match)
            if not used_atoms.intersection(match_set):
                functional_groups.append({
                    'atoms': list(match),
                    'smarts': smarts
                })
                used_atoms.update(match_set)

    return functional_groups


def decompose_component_with_virtual_atoms(mol: Chem.Mol,
                                           component_atoms: Set[int],
                                           connections: List[Tuple[int, str, str]],
                                           connection_counter: int,
                                           is_scaffold: bool = False) -> Tuple[List[Dict], int]:
    """
    递归分解一个组件，内部子结构之间也添加虚拟原子连接

    Args:
        mol: 原始分子
        component_atoms: 组件包含的原子
        connections: 组件与外部的连接
        connection_counter: 当前连接ID计数器
        is_scaffold: 是否为骨架组件

    Returns:
        (子结构列表, 更新后的连接计数器)
    """
    global_connections = []
    global_connections.extend(connections)

    if not component_atoms:
        return [], connection_counter

    # 1. 寻找组件内的环系统（直接在原始分子上操作）
    # 获取组件内所有原子的环信息
    ring_info = mol.GetRingInfo()

    # 找出完全包含在组件内的环
    component_rings = []
    try:
        for ring in ring_info.AtomRings():
            ring_set = set(ring)
            if ring_set.issubset(component_atoms):
                component_rings.append(ring_set)
    except:
        pass

    if not component_rings:
        # 没有环，直接处理为非环结构
        return decompose_non_ring_component(mol, component_atoms, connections,
                                            connection_counter, is_scaffold)

    # 2. 合并融合环（共享至少一个原子）
    merged_rings = []
    for ring in component_rings:
        found = False
        for existing in merged_rings:
            if existing.intersection(ring):
                existing.update(ring)
                found = True
                break
        if not found:
            merged_rings.append(set(ring))

    # 重复合并直到稳定
    changed = True
    while changed:
        changed = False
        new_merged = []
        for ring_set in merged_rings:
            found = False
            for existing in new_merged:
                if existing.intersection(ring_set):
                    existing.update(ring_set)
                    found = True
                    changed = True
                    break
            if not found:
                new_merged.append(ring_set)
        merged_rings = new_merged

    # 3. 标记已使用的原子
    used_atoms = set()
    substructures = []
    internal_connections = {}  # 记录组件内部的连接关系

    # 4. 处理环系统
    for ring_atoms in merged_rings:
        if used_atoms.intersection(ring_atoms):
            continue

        # 找出该环系统的内部连接点（连接到环外其他原子的位置）
        ring_connections = []
        for atom in ring_atoms:
            # 检查外部连接
            for conn in connections:
                if conn[0] == atom:
                    ring_connections.append(conn)

            # 检查环内连接到其他未处理子结构的原子
            orig_atom = mol.GetAtomWithIdx(atom)
            for neighbor in orig_atom.GetNeighbors():
                n_idx = neighbor.GetIdx()
                # 如果邻居在组件内但不在当前环中，且未被使用
                if n_idx in component_atoms and n_idx not in ring_atoms:
                    bond = mol.GetBondBetweenAtoms(atom, n_idx)
                    bond_type = bond_type_to_str(bond.GetBondType())

                    # 获取或创建连接ID
                    pair_key = tuple(sorted([atom, n_idx]))
                    if pair_key not in internal_connections:
                        connection_counter += 1
                        internal_connections[pair_key] = connection_counter

                    conn_id = internal_connections[pair_key]
                    virt_label = f"{conn_id}"
                    ring_connections.append((atom, virt_label, bond_type))

        # 生成带虚拟原子的环系统
        sub_mol_with_virt = extract_substructure_with_virtual_atoms(mol, ring_atoms, ring_connections)
        if sub_mol_with_virt and sub_mol_with_virt.GetNumAtoms() > 0:
            Chem.Kekulize(sub_mol_with_virt, True, canonical=True)
            sub_type = SubstructureType.RING_SYSTEM_SCAFFOLD if is_scaffold else SubstructureType.RING_SYSTEM
            substructures.append({
                'type': sub_type,
                'name': f"ring_system_{len(substructures) + 1}",
                'atoms': list(ring_atoms),
                'smiles': Chem.MolToSmiles(sub_mol_with_virt),
                'connections': ring_connections,
                'is_scaffold': is_scaffold
            })
            used_atoms.update(ring_atoms)

    # 5. 处理剩余的非环结构
    remaining_atoms = component_atoms - used_atoms
    if remaining_atoms:
        # 按连通分量分组
        visited = set()
        for start in remaining_atoms:
            if start in visited:
                continue

            # BFS找连通分量
            component = []
            queue = [start]
            visited.add(start)

            while queue:
                current = queue.pop(0)
                component.append(current)
                for neighbor in mol.GetAtomWithIdx(current).GetNeighbors():
                    n_idx = neighbor.GetIdx()
                    if n_idx in remaining_atoms and n_idx not in visited:
                        visited.add(n_idx)
                        queue.append(n_idx)

            if component:
                comp_set = set(component)

                # 分解这个非环分量
                sub_results, connection_counter = decompose_non_ring_component(
                    mol, comp_set, connections, connection_counter, is_scaffold,
                    internal_connections, used_atoms
                )
                substructures.extend(sub_results)
                used_atoms.update(comp_set)

    return substructures, connection_counter


def decompose_non_ring_component(mol: Chem.Mol,
                                 component_atoms: Set[int],
                                 external_connections: List[Tuple[int, str, str]],
                                 connection_counter: int,
                                 is_scaffold: bool = False,
                                 internal_connections: Dict = None,
                                 already_used: Set[int] = None) -> Tuple[List[Dict], int]:
    """
    递归分解非环组件，使用预定义结构列表进行迭代匹配

    策略：
    1. 遍历预定义结构列表
    2. 对于每个预定义结构，查找所有匹配
    3. 每次匹配一个结构，添加虚拟原子连接
    4. 从组件中删除已匹配的原子
    5. 继续匹配同一个结构，直到没有匹配
    6. 然后继续下一个预定义结构
    """
    if internal_connections is None:
        internal_connections = {}
    if already_used is None:
        already_used = set()

    if not component_atoms:
        return [], connection_counter

    # 创建临时子分子用于匹配
    atom_list = sorted(component_atoms)
    temp_mol,_,_ = extract_submol(mol, atom_list)

    if temp_mol.GetNumAtoms() == 0:
        return [], connection_counter

    # 记录已匹配的原子
    matched_atoms_global = set()
    substructures = []

    # 遍历每个预定义结构
    for smarts in func_group_list:
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is None:
            continue

        # 持续匹配同一个结构，直到没有更多匹配
        while True:
            # 获取当前剩余的原子
            remaining_atoms = component_atoms - matched_atoms_global
            if not remaining_atoms:
                break

            # 创建当前剩余原子的临时分子
            remaining_list = sorted(remaining_atoms)
            current_temp_mol, old_2_new, new_2_old = extract_submol(mol, remaining_list)
            if current_temp_mol.GetNumAtoms() == 0:
                break

            # 查找匹配
            matches = current_temp_mol.GetSubstructMatches(pattern)
            if not matches:
                break

            # 取第一个匹配
            match = matches[0]

            # 将匹配的原子映射回原分子索引
            matched_atoms = [new_2_old[i] for i in match]
            matched_set = set(matched_atoms)

            # 避免重复匹配
            if matched_set.intersection(matched_atoms_global):
                break

            # 查找这个子结构的连接点
            sub_connections = []

            # 外部连接
            for atom in matched_atoms:
                for conn in external_connections:
                    if conn[0] == atom:
                        sub_connections.append(conn)

            # 内部连接到已处理的结构
            for atom in matched_atoms:
                orig_atom = mol.GetAtomWithIdx(atom)
                for neighbor in orig_atom.GetNeighbors():
                    n_idx = neighbor.GetIdx()
                    if n_idx not in matched_atoms:
                        bond = mol.GetBondBetweenAtoms(atom, n_idx)
                        bond_type = bond_type_to_str(bond.GetBondType())

                        if n_idx in atom_list:
                            pair_key = tuple(sorted([atom, n_idx]))
                            if pair_key not in internal_connections:
                                connection_counter += 1
                                internal_connections[pair_key] = connection_counter

                            conn_id = internal_connections[pair_key]
                            virt_label = f"{conn_id}"
                            sub_connections.append((atom, virt_label, bond_type))
                        else:
                            pair_key = tuple(sorted([atom, n_idx]))
                            if pair_key in internal_connections:
                                conn_id = internal_connections[pair_key]
                                virt_label = f"{conn_id}"
                                sub_connections.append((atom, virt_label, bond_type))

            # 检查子结构内部是否有需要连接的原子
            for atom in matched_atoms:
                orig_atom = mol.GetAtomWithIdx(atom)
                for neighbor in orig_atom.GetNeighbors():
                    n_idx = neighbor.GetIdx()
                    if n_idx in matched_set and n_idx != atom:
                        # 内部连接，不需要虚拟原子
                        pass

            # 生成带虚拟原子的子结构
            sub_mol_with_virt = extract_substructure_with_virtual_atoms(mol, matched_set, sub_connections)

            if sub_mol_with_virt and sub_mol_with_virt.GetNumAtoms() > 0:

                substructures.append({
                    'type': "",
                    'name': "",
                    'atoms': matched_atoms,
                    'smiles': Chem.MolToSmiles(sub_mol_with_virt),
                    'connections': sub_connections,
                    'is_scaffold': is_scaffold
                })

                # 标记为已使用
                matched_atoms_global.update(matched_set)
                already_used.update(matched_set)

                # 继续匹配同一个结构
                continue
            else:
                break

    # 处理剩余未匹配的原子
    remaining_atoms = component_atoms - matched_atoms_global
    if remaining_atoms:
        # 按连通分量分组
        visited = set()
        remaining_list = list(remaining_atoms)

        for start in remaining_list:
            if start in visited:
                continue

            # BFS找连通分量
            component = []
            queue = [start]
            visited.add(start)

            while queue:
                current = queue.pop(0)
                component.append(current)
                for neighbor in mol.GetAtomWithIdx(current).GetNeighbors():
                    n_idx = neighbor.GetIdx()
                    if n_idx in remaining_atoms and n_idx not in visited:
                        visited.add(n_idx)
                        queue.append(n_idx)

            if component:
                comp_set = set(component)

                # 查找该分量的连接点
                comp_connections = []
                for atom in comp_set:
                    # 外部连接
                    for conn in external_connections:
                        if conn[0] == atom:
                            comp_connections.append(conn)

                    # 内部连接到已处理的结构
                    orig_atom = mol.GetAtomWithIdx(atom)
                    for neighbor in orig_atom.GetNeighbors():
                        n_idx = neighbor.GetIdx()
                        if n_idx in already_used:
                            bond = mol.GetBondBetweenAtoms(atom, n_idx)
                            bond_type = bond_type_to_str(bond.GetBondType())

                            pair_key = tuple(sorted([atom, n_idx]))
                            if pair_key not in internal_connections:
                                connection_counter += 1
                                internal_connections[pair_key] = connection_counter

                            conn_id = internal_connections[pair_key]
                            virt_label = f"{conn_id}"
                            comp_connections.append((atom, virt_label, bond_type))

                # 确定类型
                if len(comp_set) == 1:
                    atom = mol.GetAtomWithIdx(list(comp_set)[0])
                    atom_symbol = atom.GetSymbol()
                    sub_name = f"atom_{atom_symbol}"
                    sub_type = SubstructureType.SIDE_CHAIN
                else:
                    sub_type = SubstructureType.BRIDGE if not is_scaffold else SubstructureType.BRIDGE_SCAFFOLD
                    sub_name = f"chain_{len(comp_set)}"

                # 生成带虚拟原子的子结构
                sub_mol_with_virt = extract_substructure_with_virtual_atoms(mol, comp_set, comp_connections)

                if sub_mol_with_virt and sub_mol_with_virt.GetNumAtoms() > 0:
                    substructures.append({
                        'type': sub_type,
                        'name': sub_name,
                        'atoms': list(comp_set),
                        'smiles': Chem.MolToSmiles(sub_mol_with_virt),
                        'connections': comp_connections,
                        'is_scaffold': is_scaffold
                    })
                    already_used.update(comp_set)

    return substructures, connection_counter

def decompose_smiles(smiles: str):
    result = decompose_molecule_complete(smiles)

    return [i['smiles'] for i in result['substructures']]

def decompose_molecule_complete(smiles: str, start_connection_id: int = 0) -> Dict:
    """
    完整的分子分解函数（层次化分解）

    Args:
        smiles: 输入SMILES
        start_connection_id: 起始连接ID

    Returns:
        包含所有子结构的分解结果
    """
    # 1. 基础分解（骨架和侧链）
    decomposition = decompose_to_scaffold_and_side_chains(smiles)
    mol = decomposition['original_mol']
    scaffold_atoms = decomposition['scaffold_atoms']
    side_chain_components = decomposition['side_chain_components']
    attachment_points = decomposition['attachment_points']

    # 2. 构建原子到组件的映射
    atom_to_component = {}
    for atom in scaffold_atoms:
        atom_to_component[atom] = 'scaffold'

    component_id = 1
    component_atoms_map = {}
    for comp in side_chain_components:
        comp_name = f'sidechain_{component_id}'
        component_atoms_map[comp_name] = set(comp)
        for atom in comp:
            atom_to_component[atom] = comp_name
        component_id += 1

    # 3. 为组件间连接分配ID
    connection_counter = start_connection_id
    component_pairs = {}
    component_connections = defaultdict(list)

    for ap in attachment_points:
        scaffold_atom = ap['scaffold_atom']
        side_atom = ap['side_chain_atom']
        bond_type = ap['bond_type']

        comp1 = atom_to_component.get(scaffold_atom)
        comp2 = atom_to_component.get(side_atom)

        if comp1 and comp2 and comp1 != comp2:
            pair_key = tuple(sorted([comp1, comp2]))
            if pair_key not in component_pairs:
                connection_counter += 1
                component_pairs[pair_key] = connection_counter

            conn_id = component_pairs[pair_key]
            virt_label = f"{conn_id}"

            component_connections[comp1].append((scaffold_atom, virt_label, bond_type))
            component_connections[comp2].append((side_atom, virt_label, bond_type))

    # 4. 递归分解每个组件
    all_substructures = []

    # 分解骨架
    if scaffold_atoms:
        scaffold_connections = component_connections.get('scaffold', [])
        scaffold_substructures, connection_counter = decompose_component_with_virtual_atoms(
            mol, set(scaffold_atoms), scaffold_connections, connection_counter, is_scaffold=True
        )
        all_substructures.extend(scaffold_substructures)

    # 分解每个侧链
    for comp_name, comp_atoms in component_atoms_map.items():
        side_connections = component_connections.get(comp_name, [])
        side_substructures, connection_counter = decompose_component_with_virtual_atoms(
            mol, comp_atoms, side_connections, connection_counter, is_scaffold=False
        )
        all_substructures.extend(side_substructures)

    # 5. 返回结果
    canonical_smiles = Chem.MolToSmiles(mol, canonical=True)

    return {
        'original_smiles': smiles,
        'canonical_smiles': canonical_smiles,
        'substructures': all_substructures,
        'substructure_count': len(all_substructures),
        'max_connection_id': connection_counter,
        'summary': get_summary(all_substructures)
    }


def get_summary(substructures: List[Dict]) -> Dict:
    """生成类型统计"""
    summary = defaultdict(int)
    for sub in substructures:
        summary[sub['type']] += 1
    return dict(summary)


def print_decomposition_result(result: Dict):
    """打印分解结果"""
    print("\n" + "=" * 80)
    print("分子分解结果")
    print("=" * 80)

    print(f"\n原始SMILES: {result['original_smiles']}")
    print(f"规范SMILES: {result['canonical_smiles']}")
    print(f"最大连接ID: {result['max_connection_id']}")
    print(f"子结构数量: {result['substructure_count']}")
    print(f"类型统计: {result['summary']}")

    print(f"\n子结构列表:")
    print("-" * 60)

    for i, sub in enumerate(result['substructures']):
        print(f"\n[{i+1}] 类型: {sub['type']}")
        print(f"    名称: {sub['name']}")
        print(f"    SMILES: {sub['smiles']}")
        print(f"    原子: {sub['atoms']}")
        if sub['connections']:
            conn_str = ", ".join([f"{c[1]}{c[2]}" for c in sub['connections']])
            print(f"    连接: {conn_str}")
        print(f"    骨架: {sub['is_scaffold']}")


def test_all():
    """测试所有分子"""
    test_molecules = [
        ("CS(=O)(=O)C1=CC=C(N2CCN(CC3=CC=C(NS(=O)(=O)C4=CC=NC5=CC=CN=C54)C=C3)CC2)C=C1", "苯酚"),
        ("CC(=O)O[C@H]1[C@H]2[C@@]([C@H]3[C@@]([C@]4(C[C@@H]5[C@]6(C[C@@H](C(=C([C@@H](O6)C(=O)[C@]5(C4=C(C3=O)C)OC(=O)C)OC(=O)c7ccccc7)O)C)OC(=O)C)O2)OC(=O)c8ccccc8)(C1(C)C)OC(=O)C",
         "紫杉醇")
    ]

    for smiles, name in test_molecules:
        print("\n" + "=" * 60)
        print(f"测试: {name} ({smiles})")
        print("=" * 60)

        try:
            result = decompose_molecule_complete(smiles)
            final_smiles = [i['smiles'] for i in result['substructures']]
            print_decomposition_result(result)
        except Exception as e:
            print(f"错误: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    print("RDKit 版本:", Chem.rdBase.rdkitVersion)
    test_all()