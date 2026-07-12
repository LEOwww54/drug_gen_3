"""
对称性分析函数（修复版4 - 真正的Weisfeiler-Lehman）
使用 igraph 计算带有虚拟原子的分子的对称性
"""

from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict, Counter
import igraph as ig
from rdkit import Chem
from rdkit.Chem.rdchem import BondType, Atom
import hashlib


def get_bond_color(bond: Chem.Bond) -> str:
    """获取键的颜色编码"""
    bond_type = bond.GetBondType()

    if bond_type == BondType.SINGLE:
        return "1"
    elif bond_type == BondType.DOUBLE:
        return "2"
    elif bond_type == BondType.TRIPLE:
        return "3"
    elif bond_type == BondType.AROMATIC:
        return "A"
    else:
        return "U"


def get_atom_initial_color(atom: Chem.Atom, include_virtual: bool = True) -> str:
    """
    获取原子的初始颜色（基础属性）

    这个颜色应该尽可能详细地反映原子的化学环境
    """
    atomic_num = atom.GetAtomicNum()

    # 虚拟原子特殊处理
    if atomic_num == 0:
        if include_virtual and atom.HasProp("_VirtualLabel"):
            return f"VIRT_{atom.GetProp('_VirtualLabel')}"
        else:
            return "VIRT"

    # 真实原子的详细属性
    symbol = atom.GetSymbol()
    charge = atom.GetFormalCharge()
    h_count = atom.GetTotalNumHs()
    is_aromatic = atom.GetIsAromatic()
    degree = atom.GetDegree()
    implicit_valence = atom.GetImplicitValence()
    explicit_valence = atom.GetExplicitValence()
    total_valence = atom.GetTotalValence()
    hybridization = atom.GetHybridization()
    no_implicit = atom.GetNoImplicit()
    chiral_tag = atom.GetChiralTag()

    # 组合成一个详细的签名
    return (f"{atomic_num}_{symbol}_{charge}_{h_count}_{is_aromatic}_{degree}_"
            f"{implicit_valence}_{explicit_valence}_{total_valence}_{hybridization}_{no_implicit}_{chiral_tag}")


def get_bond_initial_color(bond: Chem.Bond) -> str:
    """获取键的初始颜色"""
    bond_type = bond.GetBondType()
    is_aromatic = bond.GetIsAromatic()
    is_conjugated = bond.GetIsConjugated()
    is_in_ring = bond.IsInRing()

    return f"{get_bond_color(bond)}_{is_aromatic}_{is_conjugated}_{is_in_ring}"


def weisfeiler_lehman_colors(mol: Chem.Mol, include_virtual: bool = True,
                            max_iterations: int = 10) -> Dict[int, str]:
    """
    使用真正的 Weisfeiler-Lehman 算法计算每个原子的最终颜色

    Args:
        mol: RDKit分子对象
        include_virtual: 是否包含虚拟原子
        max_iterations: 最大迭代次数

    Returns:
        原子索引 -> 最终颜色字符串
    """
    # 1. 初始化：获取所有原子的初始颜色
    current_colors = {}
    atoms_to_consider = []

    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        if atom.GetAtomicNum() == 0 and not include_virtual:
            continue
        atoms_to_consider.append(idx)
        current_colors[idx] = get_atom_initial_color(atom, include_virtual)

    # 如果没有原子，返回空字典
    if not current_colors:
        return {}

    # 2. 迭代更新颜色
    for iteration in range(max_iterations):
        new_colors = {}

        for idx in atoms_to_consider:
            atom = mol.GetAtomWithIdx(idx)

            # 获取当前原子的颜色
            current_color = current_colors[idx]

            # 获取所有邻居的颜色（包括键的信息）
            neighbor_info = []
            for neighbor in atom.GetNeighbors():
                neighbor_idx = neighbor.GetIdx()

                # 忽略氢原子（除非我们特别想要包含它们）
                if neighbor.GetAtomicNum() == 1:
                    continue

                # 如果邻居不在我们的考虑范围内，跳过
                if neighbor_idx not in current_colors:
                    continue

                # 获取键的信息
                bond = mol.GetBondBetweenAtoms(idx, neighbor_idx)
                if bond is None:
                    continue

                bond_color = get_bond_initial_color(bond)
                neighbor_color = current_colors[neighbor_idx]

                # 组合键和邻居的信息
                neighbor_info.append(f"{bond_color}|{neighbor_color}")

            # 排序邻居信息以确保一致性
            neighbor_info.sort()

            # 组合当前颜色和邻居颜色
            color_tuple = (current_color, tuple(neighbor_info))

            # 使用哈希生成新颜色
            color_str = str(color_tuple)
            new_color = hashlib.md5(color_str.encode()).hexdigest()[:12]
            new_colors[idx] = new_color

        # 检查是否收敛
        if new_colors == current_colors:
            break

        current_colors = new_colors

    return current_colors


def mol_to_igraph_with_wl_colors(mol: Chem.Mol, include_virtual: bool = True) -> ig.Graph:
    """
    使用 Weisfeiler-Lehman 颜色将 RDKit 分子转换为 igraph 图
    """
    # 计算 Weisfeiler-Lehman 颜色
    wl_colors = weisfeiler_lehman_colors(mol, include_virtual, max_iterations=10)

    # 获取所有需要考虑的原子
    atoms = []
    atom_colors = []
    atom_indices = []

    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        if atom.GetAtomicNum() == 0 and not include_virtual:
            continue
        if idx not in wl_colors:
            continue

        atoms.append(atom)
        atom_indices.append(idx)
        atom_colors.append(wl_colors[idx])

    # 创建图
    g = ig.Graph()
    g.add_vertices(len(atoms))

    # 设置顶点颜色和原始索引
    g.vs['color'] = atom_colors
    g.vs['orig_idx'] = atom_indices

    # 添加边和边的颜色
    edges = []
    edge_colors = []

    for i, atom1 in enumerate(atoms):
        idx1 = atom1.GetIdx()
        for j, atom2 in enumerate(atoms):
            if i < j:
                idx2 = atom2.GetIdx()
                bond = mol.GetBondBetweenAtoms(idx1, idx2)
                if bond is not None:
                    edges.append((i, j))
                    edge_colors.append(get_bond_initial_color(bond))

    g.add_edges(edges)
    g.es['color'] = edge_colors

    return g


def compute_weisfeiler_lehman_labels(g: ig.Graph) -> List[int]:
    """
    使用Weisfeiler-Lehman算法计算图的规范标签（备用方案）
    """
    n = g.vcount()
    if n == 0:
        return []

    # 初始标签：使用顶点颜色
    colors = g.vs['color']

    # 将颜色字符串映射到整数
    color_to_int = {color: i for i, color in enumerate(set(colors))}
    current_labels = [color_to_int[color] for color in colors]

    # 迭代细化标签
    max_iter = n
    for _ in range(max_iter):
        # 收集每个节点的邻居标签（排序后）
        new_labels = []
        for i in range(n):
            neighbor_labels = sorted([current_labels[neighbor] for neighbor in g.neighbors(i)])
            # 组合当前标签和邻居标签
            label_tuple = (current_labels[i], tuple(neighbor_labels))
            new_labels.append(hash(str(label_tuple)))

        # 规范化标签（映射到连续整数）
        unique_labels = {}
        canonical = []
        next_label = 0
        for label in new_labels:
            if label not in unique_labels:
                unique_labels[label] = next_label
                next_label += 1
            canonical.append(unique_labels[label])

        # 检查是否收敛
        if canonical == current_labels:
            break
        current_labels = canonical

    return current_labels


def compute_orbits_from_automorphism(aut, g: ig.Graph) -> List[List[int]]:
    """
    从automorphism_group的返回值中提取轨道
    """
    # 情况1: 返回值有orbits方法
    if hasattr(aut, 'orbits'):
        return aut.orbits()

    # 情况2: 返回值是列表，包含生成元
    if isinstance(aut, list):
        n = g.vcount()
        return compute_orbits_from_generators(aut, n)

    # 情况3: 返回值有generators属性
    if hasattr(aut, 'generators'):
        n = g.vcount()
        return compute_orbits_from_generators(aut.generators, n)

    # 情况4: 未知类型，返回每个顶点单独一个轨道
    return [[i] for i in range(g.vcount())]


def compute_orbits_from_generators(generators: List, n: int) -> List[List[int]]:
    """
    从置换生成元计算轨道
    """
    # 初始化并查集
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    # 应用每个生成元
    for gen in generators:
        if isinstance(gen, (list, tuple)):
            for i, j in enumerate(gen):
                if i != j:
                    union(i, j)
        elif hasattr(gen, 'list') and callable(gen.list):
            perm_list = gen.list()
            for i, j in enumerate(perm_list):
                if i != j:
                    union(i, j)

    # 收集轨道
    orbit_dict = defaultdict(list)
    for i in range(n):
        orbit_dict[find(i)].append(i)

    return list(orbit_dict.values())


def compute_symmetry_orbits_via_nauty(g: ig.Graph) -> Dict[int, List[int]]:
    """
    使用igraph的automorphism_group计算轨道
    """
    n = g.vcount()
    if n == 0:
        return {}

    # 获取顶点颜色
    colors = g.vs['color']
    color_to_int = {color: i for i, color in enumerate(set(colors))}
    color_indices = [color_to_int[color] for color in colors]

    # 尝试多种方法调用automorphism_group
    aut = None
    methods_tried = []

    # 方法1: 使用color参数（整数列表）
    try:
        aut = g.automorphism_group(color=color_indices)
        methods_tried.append("color=color_indices")
    except Exception as e:
        pass

    # 方法2: 使用colors参数
    if aut is None:
        try:
            aut = g.automorphism_group(colors=color_indices)
            methods_tried.append("colors=color_indices")
        except Exception as e:
            pass

    # 方法3: 使用vertex_color参数
    if aut is None:
        try:
            aut = g.automorphism_group(vertex_color=color_indices)
            methods_tried.append("vertex_color=color_indices")
        except Exception as e:
            pass

    # 方法4: 不使用颜色参数
    if aut is None:
        try:
            aut = g.automorphism_group()
            methods_tried.append("no colors")
        except Exception as e:
            pass

    if aut is None:
        print(f"All automorphism_group methods failed. Tried: {methods_tried}")
        return None

    # 从返回值提取轨道
    orbits = compute_orbits_from_automorphism(aut, g)

    # 根据颜色进一步细分轨道
    refined_orbits = []
    for orbit in orbits:
        color_groups = defaultdict(list)
        for vertex in orbit:
            if vertex < len(g.vs):
                color = g.vs[vertex]['color']
                color_groups[color].append(vertex)
        refined_orbits.extend(color_groups.values())

    return {i: list(orbit) for i, orbit in enumerate(refined_orbits)}


def compute_symmetry_orbits(mol: Chem.Mol, include_virtual: bool = True) -> Dict[int, int]:
    """
    计算分子的对称轨道（等价类）

    Args:
        mol: RDKit分子对象（可包含虚拟原子）
        include_virtual: 是否在对称性分析中包含虚拟原子

    Returns:
        字典: 原子索引 -> 轨道ID（等价类编号），只包含真实原子
    """
    # 使用 Weisfeiler-Lehman 颜色构建图
    g = mol_to_igraph_with_wl_colors(mol, include_virtual)

    if g.vcount() == 0:
        return {}

    # 尝试使用igraph计算轨道
    orbits_dict = compute_symmetry_orbits_via_nauty(g)

    atom_to_orbit = {}

    if orbits_dict is not None:
        # 构建原子索引到轨道ID的映射
        for orbit_id, vertices in orbits_dict.items():
            for vertex_id in vertices:
                if vertex_id < len(g.vs):
                    orig_idx = g.vs[vertex_id]['orig_idx']
                    # 检查这个顶点是否是真实原子
                    orig_atom = mol.GetAtomWithIdx(orig_idx)
                    if orig_atom.GetAtomicNum() != 0:  # 只保留真实原子
                        atom_to_orbit[orig_idx] = orbit_id
    else:
        # 降级方案：使用Weisfeiler-Lehman算法
        print("Falling back to Weisfeiler-Lehman canonical labeling...")
        wl_labels = compute_weisfeiler_lehman_labels(g)

        for vertex_id, label in enumerate(wl_labels):
            if vertex_id < len(g.vs):
                orig_idx = g.vs[vertex_id]['orig_idx']
                orig_atom = mol.GetAtomWithIdx(orig_idx)
                if orig_atom.GetAtomicNum() != 0:  # 只保留真实原子
                    atom_to_orbit[orig_idx] = label

    # 过滤掉只包含虚拟原子的等价类，并重新编号
    used_orbit_ids = set(atom_to_orbit.values())

    old_to_new = {}
    new_id = 0
    for old_id in sorted(used_orbit_ids):
        old_to_new[old_id] = new_id
        new_id += 1

    filtered_result = {}
    for atom_idx, old_orbit_id in atom_to_orbit.items():
        if old_orbit_id in old_to_new:
            filtered_result[atom_idx] = old_to_new[old_orbit_id]

    return filtered_result


def add_symmetry_labels_to_mol(mol: Chem.Mol,
                                include_virtual: bool = True,
                                property_name: str = "SymmetryClass") -> Chem.Mol:
    """
    为分子添加对称性标签作为原子属性
    """
    # 深拷贝分子
    result_mol = Chem.Mol(mol)

    # 计算对称性轨道
    atom_to_orbit = compute_symmetry_orbits(mol, include_virtual)

    # 添加属性
    for atom in result_mol.GetAtoms():
        idx = atom.GetIdx()
        if idx in atom_to_orbit:
            atom.SetProp(property_name, str(atom_to_orbit[idx]))
        else:
            atom.SetProp(property_name, "-1")

    return result_mol


def get_symmetry_equivalent_atoms(mol: Chem.Mol, include_virtual: bool = True) -> tuple[Dict[int, List[int]], Dict[int, int]]:
    """
    获取对称等价原子组
    """
    atom_to_orbit = compute_symmetry_orbits(mol, include_virtual)

    orbit_to_atoms = defaultdict(list)
    atoms_to_orbit = {}

    for atom_idx, orbit_id in atom_to_orbit.items():
        orbit_to_atoms[orbit_id].append(atom_idx)
        atoms_to_orbit[atom_idx] = orbit_id

    return dict(orbit_to_atoms), atoms_to_orbit


def print_symmetry_info(mol: Chem.Mol, include_virtual: bool = True):
    """打印分子的对称性信息"""
    print("\n" + "=" * 60)
    print("对称性分析结果")
    print("=" * 60)

    orbit_to_atoms, atoms_to_orbit = get_symmetry_equivalent_atoms(mol, include_virtual)

    print(f"\n共发现 {len(orbit_to_atoms)} 个等价类:")
    print("-" * 40)

    for orbit_id, atoms in sorted(orbit_to_atoms.items()):
        atom_symbols = []
        for idx in atoms:
            atom = mol.GetAtomWithIdx(idx)
            atomic_num = atom.GetAtomicNum()
            if atomic_num == 0:
                if atom.HasProp("_VirtualLabel"):
                    symbol = f"*{atom.GetProp('_VirtualLabel')}"
                else:
                    symbol = "*"
            else:
                symbol = atom.GetSymbol()
            atom_symbols.append(f"{idx}({symbol})")

        print(f"  等价类 {orbit_id}: {', '.join(atom_symbols)}")


def test_symmetry_analysis():
    """测试对称性分析"""

    # 测试1: 不对称取代的环己烷
    print("\n" + "=" * 80)
    print("测试1: 不对称取代的环己烷 (C1CCC(=C(O)C)CC1)")
    print("=" * 80)

    mol_smiles = "C1CCC(=C(O)C)CC1"
    mol = Chem.MolFromSmiles(mol_smiles)

    if mol:
        print(f"\n输入SMILES: {mol_smiles}")

        # 使用新的 Weisfeiler-Lehman 方法
        print("\n使用 Weisfeiler-Lehman 方法:")
        atom_to_orbit = compute_symmetry_orbits(mol, include_virtual=False)
        orbit_to_atoms = defaultdict(list)
        for atom_idx, orbit_id in atom_to_orbit.items():
            orbit_to_atoms[orbit_id].append(atom_idx)

        print(f"  共发现 {len(orbit_to_atoms)} 个等价类:")
        for orbit_id, atoms in sorted(orbit_to_atoms.items()):
            atom_symbols = []
            for idx in atoms:
                atom = mol.GetAtomWithIdx(idx)
                symbol = atom.GetSymbol()
                atom_symbols.append(f"{idx}({symbol})")
            print(f"    等价类 {orbit_id}: {', '.join(atom_symbols)}")

        print(f"\n总计 {len(atom_to_orbit)} 个原子，分为 {len(orbit_to_atoms)} 个等价类")
        print("期望结果：9个原子应该分为9个等价类（没有两个原子完全等价）")

        if len(orbit_to_atoms) == 9:
            print("✓ 成功！所有原子都被正确地区分为不同的等价类")
        else:
            print(f"✗ 警告：发现 {len(orbit_to_atoms)} 个等价类，期望 9 个")

    # 测试2: 苯
    print("\n" + "=" * 80)
    print("测试2: 苯 (完全对称)")
    print("=" * 80)

    benzene_smiles = "c1ccccc1"
    benzene_mol = Chem.MolFromSmiles(benzene_smiles)

    if benzene_mol:
        print(f"\n输入SMILES: {benzene_smiles}")
        atom_to_orbit = compute_symmetry_orbits(benzene_mol, include_virtual=False)
        orbit_to_atoms = defaultdict(list)
        for atom_idx, orbit_id in atom_to_orbit.items():
            orbit_to_atoms[orbit_id].append(atom_idx)

        print(f"  共发现 {len(orbit_to_atoms)} 个等价类:")
        for orbit_id, atoms in sorted(orbit_to_atoms.items()):
            atom_symbols = []
            for idx in atoms:
                atom = benzene_mol.GetAtomWithIdx(idx)
                symbol = atom.GetSymbol()
                atom_symbols.append(f"{idx}({symbol})")
            print(f"    等价类 {orbit_id}: {', '.join(atom_symbols)}")

        print(f"\n总计 {len(atom_to_orbit)} 个原子，分为 {len(orbit_to_atoms)} 个等价类")
        print("期望结果：6个碳原子全部等价（1个等价类）")


if __name__ == "__main__":
    print("RDKit 版本:", Chem.rdBase.rdkitVersion)
    print("igraph 版本:", ig.__version__)
    test_symmetry_analysis()