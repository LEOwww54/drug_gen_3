"""
第一步：骨架分解函数
功能：输入SMILES，输出骨架和侧链的原子索引
"""

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from typing import List, Tuple, Set, Dict, Optional
from rdkit.Chem import QED


def decompose_to_scaffold_and_side_chains(smiles: str) -> Dict:
    """
    将分子分解为骨架和侧链

    Args:
        smiles: 输入SMILES字符串

    Returns:
        包含以下键的字典:
        - 'original_mol': RDKit分子对象
        - 'scaffold_mol': 骨架分子对象
        - 'scaffold_atoms': 骨架原子在原分子中的索引集合
        - 'side_chain_atoms': 侧链原子在原分子中的索引集合
        - 'side_chain_components': 侧链的连通分量列表
        - 'attachment_points': 骨架上的连接点信息
    """

    # 1. 解析分子
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    # 使用规范SMILES确保一致性
    canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
    mol = Chem.MolFromSmiles(canonical_smiles)
    Chem.SanitizeMol(mol)

    print(f"\n原始分子:")
    print(f"  SMILES: {canonical_smiles}")
    print(f"  原子数: {mol.GetNumAtoms()}")

    # 2. 提取Bemis-Murcko骨架
    scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)

    if scaffold_mol.GetNumAtoms() == 0:
        print("  警告: 分子无环，无法提取骨架")
        return {
            'original_mol': mol,
            'scaffold_mol': None,
            'scaffold_atoms': set(),
            'side_chain_atoms': set(range(mol.GetNumAtoms())),
            'side_chain_components': [],
            'attachment_points': []
        }

    print(f"\n骨架分子:")
    print(f"  SMILES: {Chem.MolToSmiles(scaffold_mol)}")
    print(f"  原子数: {scaffold_mol.GetNumAtoms()}")

    # 3. 找到骨架在原分子中的对应原子
    scaffold_atoms = find_scaffold_atoms_in_original(mol, scaffold_mol)

    print(f"\n骨架映射:")
    print(f"  骨架原子索引: {sorted(scaffold_atoms)}")

    # 4. 侧链原子 = 所有原子 - 骨架原子
    all_atoms = set(range(mol.GetNumAtoms()))
    side_chain_atoms = all_atoms - scaffold_atoms

    print(f"\n侧链:")
    print(f"  侧链原子索引: {sorted(side_chain_atoms)}")

    # 5. 找出侧链的连通分量
    side_chain_components = find_connected_components(mol, list(side_chain_atoms))

    print(f"  侧链连通分量数: {len(side_chain_components)}")
    for i, comp in enumerate(side_chain_components):
        print(f"    分量{i+1}: {comp}")

    # 6. 找出骨架上的连接点（与侧链相连的原子）
    attachment_points = find_attachment_points(mol, scaffold_atoms, side_chain_atoms)

    print(f"\n连接点:")
    for ap in attachment_points:
        atom = mol.GetAtomWithIdx(ap['scaffold_atom'])
        print(f"  骨架原子{ap['scaffold_atom']}({atom.GetSymbol()}) "
              f"→ 侧链原子{ap['side_chain_atom']}({mol.GetAtomWithIdx(ap['side_chain_atom']).GetSymbol()}) "
              f"键型:{ap['bond_type']}")

    return {
        'original_mol': mol,
        'scaffold_mol': scaffold_mol,
        'scaffold_atoms': scaffold_atoms,
        'side_chain_atoms': side_chain_atoms,
        'side_chain_components': side_chain_components,
        'attachment_points': attachment_points
    }

def _get_prop(mol):
        qed = QED.qed(mol)
        props = QED.properties(mol)
        score = sascorer.calculateScore(mol)
        score = (10 - score) / 9

        return [qed, props.ALOGP, score]

def find_scaffold_atoms_in_original(original_mol: Chem.Mol, scaffold_mol: Chem.Mol) -> Set[int]:
    """
    找到骨架分子在原分子中对应的原子索引

    策略：使用子结构匹配
    """
    # 获取骨架的规范SMILES
    scaffold_smiles = Chem.MolToSmiles(scaffold_mol)
    scaffold_pattern = Chem.MolFromSmarts(scaffold_smiles)

    if scaffold_pattern is None:
        return set()

    # 在原分子中查找骨架
    matches = original_mol.GetSubstructMatches(scaffold_pattern)

    if matches:
        # 取第一个匹配
        scaffold_atoms = set(matches[0])
        print(f"  子结构匹配成功: {scaffold_atoms}")
        return scaffold_atoms

    # 如果直接匹配失败，尝试使用原子映射
    print("  子结构匹配失败，尝试原子映射...")

    # 方法2：基于原子类型和连接性的启发式匹配
    scaffold_atoms = heuristic_scaffold_mapping(original_mol, scaffold_mol)

    return scaffold_atoms


def heuristic_scaffold_mapping(original_mol: Chem.Mol, scaffold_mol: Chem.Mol) -> Set[int]:
    """
    启发式方法：将环原子作为骨架
    """
    ring_info = original_mol.GetRingInfo()

    try:
        # 获取所有环原子
        ring_atoms = set()
        for ring in ring_info.AtomRings():
            ring_atoms.update(ring)

        if ring_atoms:
            print(f"  启发式映射: 使用环原子作为骨架 {ring_atoms}")
            return ring_atoms
    except:
        pass

    return set()


def find_connected_components(mol: Chem.Mol, atom_indices: List[int]) -> List[List[int]]:
    """
    找出原子列表中的连通分量
    """
    if not atom_indices:
        return []

    atom_set = set(atom_indices)
    visited = set()
    components = []

    for start in atom_indices:
        if start in visited:
            continue

        # BFS
        component = []
        queue = [start]
        visited.add(start)

        while queue:
            current = queue.pop(0)
            component.append(current)

            for neighbor in mol.GetAtomWithIdx(current).GetNeighbors():
                n_idx = neighbor.GetIdx()
                if n_idx in atom_set and n_idx not in visited:
                    visited.add(n_idx)
                    queue.append(n_idx)

        components.append(component)

    return components


def find_attachment_points(mol: Chem.Mol,
                          scaffold_atoms: Set[int],
                          side_chain_atoms: Set[int]) -> List[Dict]:
    """
    找出骨架和侧链之间的连接点
    """
    attachment_points = []

    for scaffold_idx in scaffold_atoms:
        atom = mol.GetAtomWithIdx(scaffold_idx)

        for neighbor in atom.GetNeighbors():
            n_idx = neighbor.GetIdx()

            if n_idx in side_chain_atoms:
                bond = mol.GetBondBetweenAtoms(scaffold_idx, n_idx)
                bond_type = bond_type_to_str(bond.GetBondType())

                attachment_points.append({
                    'scaffold_atom': scaffold_idx,
                    'side_chain_atom': n_idx,
                    'bond_type': bond_type
                })

    return attachment_points


def bond_type_to_str(bond_type) -> str:
    """将键类型转换为字符串"""
    from rdkit.Chem.rdchem import BondType

    mapping = {
        BondType.SINGLE: "-",
        BondType.DOUBLE: "=",
        BondType.TRIPLE: "#",
        BondType.AROMATIC: ":",
    }
    return mapping.get(bond_type, "-")


def test_decomposition():
    """测试骨架分解函数"""

    test_molecules = [
        ("CS(=O)(=O)C1=CC=C(N2CCN(CC3=CC=C(NS(=O)(=O)C4=CC=NC5=CC=CN=C54)C=C3)CC2)C=C1", "苯酚"),
        ("CC(=O)O[C@H]1[C@H]2[C@@]([C@H]3[C@@]([C@]4(C[C@@H]5[C@]6(C[C@@H](C(=C([C@@H](O6)C(=O)[C@]5(C4=C(C3=O)C)OC(=O)C)OC(=O)c7ccccc7)O)C)OC(=O)C)O2)OC(=O)c8ccccc8)(C1(C)C)OC(=O)C", "紫杉醇")
    ]

    print("=" * 80)
    print("测试骨架分解")
    print("=" * 80)

    for smiles, name in test_molecules:
        print("\n" + "=" * 60)
        print(f"测试: {name}")
        print("=" * 60)

        try:
            result = decompose_to_scaffold_and_side_chains(smiles)

            # 可视化结果
            visualize_decomposition(result)

        except Exception as e:
            print(f"错误: {e}")
            import traceback
            traceback.print_exc()


def visualize_decomposition(result: Dict):
    """
    可视化分解结果
    打印原子归属信息
    """
    mol = result['original_mol']
    scaffold_atoms = result['scaffold_atoms']
    side_chain_atoms = result['side_chain_atoms']

    print("\n原子归属详情:")
    print("-" * 40)

    for idx in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(idx)
        symbol = atom.GetSymbol()

        if idx in scaffold_atoms:
            # 检查是否是连接点
            is_attachment = any(ap['scaffold_atom'] == idx for ap in result['attachment_points'])
            marker = "[骨架]" + ("*" if is_attachment else "")
        else:
            marker = "[侧链]"

        print(f"  原子{idx:2d}: {symbol:2s} {marker}")


if __name__ == "__main__":
    print("RDKit 版本:", Chem.rdBase.rdkitVersion)
    test_decomposition()