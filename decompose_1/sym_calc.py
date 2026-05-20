"""
对称性分析函数（修复版2）
使用 igraph 计算带有虚拟原子的分子的对称性
"""

from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict
import igraph as ig
from rdkit import Chem
from rdkit.Chem.rdchem import BondType, Atom


def get_atom_color(atom: Chem.Atom, include_virtual: bool = True) -> str:
    """
    获取原子的颜色编码（用于图同构判断）
    """
    atomic_num = atom.GetAtomicNum()

    # 虚拟原子（原子序数0）特殊处理
    if atomic_num == 0:
        if include_virtual and atom.HasProp("_VirtualLabel"):
            return f"VIRT_{atom.GetProp('_VirtualLabel')}"
        else:
            return "VIRT"

    # 真实原子的颜色编码
    charge = atom.GetFormalCharge()
    h_count = atom.GetTotalNumHs()
    is_aromatic = 1 if atom.GetIsAromatic() else 0
    degree = atom.GetDegree()

    return f"{atomic_num}_{charge}_{h_count}_{is_aromatic}_{degree}"


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


def mol_to_igraph_with_colors(mol: Chem.Mol, include_virtual: bool = True) -> ig.Graph:
    """将RDKit分子转换为带颜色的igraph图"""
    # 获取所有非氢原子
    atoms = []
    atom_colors = []
    atom_indices = []

    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0 and not include_virtual:
            continue
        atoms.append(atom)
        atom_indices.append(atom.GetIdx())
        atom_colors.append(get_atom_color(atom, include_virtual))

    # 创建图
    g = ig.Graph()
    g.add_vertices(len(atoms))

    # 设置顶点颜色
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
                    edge_colors.append(get_bond_color(bond))

    g.add_edges(edges)
    g.es['color'] = edge_colors

    return g


def compute_weisfeiler_lehman_labels(g: ig.Graph) -> List[int]:
    """
    使用Weisfeiler-Lehman算法计算图的规范标签
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

    Args:
        aut: automorphism_group的返回值
        g: 原始图对象

    Returns:
        轨道列表，每个轨道是顶点索引列表
    """
    # 情况1: 返回值有orbits方法
    if hasattr(aut, 'orbits'):
        return aut.orbits()

    # 情况2: 返回值是列表，包含生成元
    if isinstance(aut, list):
        n = g.vcount()
        # 使用生成元计算轨道
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

    Args:
        generators: 置换生成元列表
        n: 顶点数量

    Returns:
        轨道列表
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
        # 假设gen是一个置换，可以是列表、元组或Permutation对象
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

    # 方法4: 使用分组颜色参数
    if aut is None:
        try:
            aut = g.automorphism_group(group_colors=color_indices)
            methods_tried.append("group_colors=color_indices")
        except Exception as e:
            pass

    # 方法5: 不使用颜色参数
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

    # 如果需要，根据颜色进一步细分轨道
    refined_orbits = []
    for orbit in orbits:
        color_groups = defaultdict(list)
        for vertex in orbit:
            color = g.vs[vertex]['color']
            color_groups[color].append(vertex)
        refined_orbits.extend(color_groups.values())

    return {i: list(orbit) for i, orbit in enumerate(refined_orbits)}


def compute_symmetry_orbits(mol: Chem.Mol, include_virtual: bool = True) -> Dict[int, int]:
    """
    计算分子的对称轨道（等价类）

    注意：虚拟原子不占用等价类ID，包含虚拟原子的等价类会被删除，
    剩余等价类ID会被重新编号为连续整数

    Args:
        mol: RDKit分子对象（可包含虚拟原子）
        include_virtual: 是否在对称性分析中包含虚拟原子

    Returns:
        字典: 原子索引 -> 轨道ID（等价类编号），只包含真实原子
    """
    # 转换为igraph图
    g = mol_to_igraph_with_colors(mol, include_virtual)

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
    # 首先，找出所有被使用的轨道ID（即至少有一个真实原子的轨道）
    used_orbit_ids = set(atom_to_orbit.values())

    # 创建映射：旧轨道ID -> 新轨道ID（连续）
    old_to_new = {}
    new_id = 0
    for old_id in sorted(used_orbit_ids):
        old_to_new[old_id] = new_id
        new_id += 1

    # 应用映射
    filtered_result = {}
    #atom_to_class = {}
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

    orbit_to_atoms = get_symmetry_equivalent_atoms(mol, include_virtual)

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

    # 测试1: 苯酚
    print("\n" + "=" * 80)
    print("测试1: 苯酚 (带虚拟原子)")
    print("=" * 80)

    phenol_smiles = "[1*]c1ccccc1"
    phenol_mol = Chem.MolFromSmiles(phenol_smiles)

    if phenol_mol:
        for atom in phenol_mol.GetAtoms():
            if atom.GetAtomicNum() == 0:
                atom.SetProp("_VirtualLabel", "[1*]")

        print(f"\n输入SMILES: {phenol_smiles}")
        result_mol = add_symmetry_labels_to_mol(phenol_mol, include_virtual=True)
        print_symmetry_info(result_mol, include_virtual=True)

    # 测试2: 苯
    print("\n" + "=" * 80)
    print("测试2: 苯 (完全对称)")
    print("=" * 80)

    benzene_smiles = "c1ccccc1"
    benzene_mol = Chem.MolFromSmiles(benzene_smiles)

    if benzene_mol:
        print(f"\n输入SMILES: {benzene_smiles}")
        result_mol = add_symmetry_labels_to_mol(benzene_mol, include_virtual=False)

        orbit_to_atoms = get_symmetry_equivalent_atoms(result_mol, include_virtual=False)
        print(f"\n共发现 {len(orbit_to_atoms)} 个等价类:")
        for orbit_id, atoms in sorted(orbit_to_atoms.items()):
            atom_symbols = [f"{idx}({result_mol.GetAtomWithIdx(idx).GetSymbol()})" for idx in atoms]
            print(f"  等价类 {orbit_id}: {', '.join(atom_symbols)}")


if __name__ == "__main__":
    print("RDKit 版本:", Chem.rdBase.rdkitVersion)
    print("igraph 版本:", ig.__version__)
    test_symmetry_analysis()