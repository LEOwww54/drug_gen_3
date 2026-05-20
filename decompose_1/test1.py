"""
第一步：骨架分解函数
功能：输入SMILES，输出骨架和侧链的原子索引
"""

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from typing import List, Tuple, Set, Dict, Optional



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

    # 2. 提取Bemis-Murcko骨架
    scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)

    if scaffold_mol.GetNumAtoms() == 0:
        return {
            'original_mol': mol,
            'scaffold_mol': None,
            'scaffold_atoms': set(),
            'side_chain_atoms': set(range(mol.GetNumAtoms())),
            'side_chain_components': [list(range(mol.GetNumAtoms()))],
            'attachment_points': []
        }


    # 3. 找到骨架在原分子中的对应原子
    scaffold_atoms = find_scaffold_atoms_in_original(mol, scaffold_mol)

    # 4. 侧链原子 = 所有原子 - 骨架原子
    all_atoms = set(range(mol.GetNumAtoms()))
    side_chain_atoms = all_atoms - scaffold_atoms

    # 5. 找出侧链的连通分量
    side_chain_components = find_connected_components(mol, list(side_chain_atoms))

    # 6. 找出骨架上的连接点（与侧链相连的原子）
    attachment_points = find_attachment_points(mol, scaffold_atoms, side_chain_atoms)

    return {
        'original_mol': mol,
        'scaffold_mol': scaffold_mol,
        'scaffold_atoms': scaffold_atoms,
        'side_chain_atoms': side_chain_atoms,
        'side_chain_components': side_chain_components,
        'attachment_points': attachment_points
    }

def find_scaffold_atoms_in_original(original_mol: Chem.Mol, scaffold_mol: Chem.Mol) -> Set[int]:
    """
    找到骨架分子在原分子中对应的原子索引

    策略：使用子结构匹配
    """

    # 在原分子中查找骨架
    matches = original_mol.GetSubstructMatches(scaffold_mol)

    if matches:
        # 取第一个匹配
        scaffold_atoms = set(matches[0])
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