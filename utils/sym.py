"""
分子对称性感知的 Tokenization 系统
修复 igraph 兼容性问题
"""

import re
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict

import igraph as ig
from rdkit import Chem
from rdkit.Chem import rdmolops, CanonicalRankAtoms
from rdkit.Chem import Descriptors, Lipinski


class SymmetryAwareTokenizer:
    """对称性感知的分子 Tokenizer"""

    def __init__(self, use_symmetry: bool = True, method: str = 'rdkit'):
        """
        Args:
            use_symmetry: 是否使用对称性分析
            method: 对称性分析方法 ('rdkit' 推荐, 'igraph' 实验性)
        """
        self.use_symmetry = use_symmetry
        self.method = method if method == 'igraph' else 'rdkit'

        # 功能子结构的 SMARTS 模式库
        self.functional_patterns = {
            'hydroxyl': '[OX2H]',
            'carboxyl': 'C(=O)[OX2H]',
            'amino': '[NX3;H2,H1;!$(N~[O,N])]',
            'nitro': '[$([NX3+](=O)[O-]),$([N+](=O)[O-])]',
            'sulfonyl': 'S(=O)=O',
            'cyano': 'C#N',
            'amide': 'C(=O)[NX3]',
            'ester': 'C(=O)O[C]',
            'ether': '[OD2]([C])[C]',
            'benzene': 'c1ccccc1',
            'pyridine': 'c1ccncc1',
            'pyrimidine': 'c1cncnc1',
        }

    def process_smiles_list(self, smiles_list: List[str]) -> List[Dict]:
        """处理 SMILES 列表，返回 tokenized 结果"""
        results = []

        for idx, smiles in enumerate(smiles_list):
            try:
                result = self.tokenize_molecule(smiles)
                results.append({
                    'index': idx,
                    'smiles': smiles,
                    'canonical_smiles': result['canonical_smiles'],
                    'tokens': result['tokens'],
                    'symmetry_info': result['symmetry_info'],
                    'properties': result['properties']
                })
            except Exception as e:
                print(f"Error processing SMILES {idx} ({smiles}): {e}")
                results.append({
                    'index': idx,
                    'smiles': smiles,
                    'error': str(e)
                })

        return results

    def tokenize_molecule(self, smiles: str) -> Dict:
        """对单个分子进行 tokenization"""
        # 1. 转换为规范 SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
        mol = Chem.MolFromSmiles(canonical_smiles)
        Chem.SanitizeMol(mol)

        # 2. 提取功能子结构
        functional_substructures = self._extract_functional_substructures(mol)

        # 3. 剩余结构处理
        remaining = self._get_remaining_structure(mol, functional_substructures)

        # 4. 对称性分析
        symmetry_info = self._analyze_symmetry(mol) if self.use_symmetry else {}

        # 5. 生成 tokens
        tokens = self._generate_tokens(mol, functional_substructures, remaining, symmetry_info)

        # 6. 计算分子性质
        properties = self._calculate_properties(mol)

        return {
            'canonical_smiles': canonical_smiles,
            'tokens': tokens,
            'symmetry_info': symmetry_info,
            'properties': properties
        }

    def _extract_functional_substructures(self, mol: Chem.Mol) -> List[Dict]:
        """提取功能子结构"""
        substructures = []
        used_atoms = set()

        for name, smarts in self.functional_patterns.items():
            pattern = Chem.MolFromSmarts(smarts)
            if pattern is None:
                continue

            matches = mol.GetSubstructMatches(pattern)
            for match in matches:
                # 检查是否与已使用的原子重叠
                if not used_atoms.intersection(match):
                    substructures.append({
                        'name': name,
                        'atoms': list(match),
                        'smarts': smarts
                    })
                    used_atoms.update(match)

        return substructures

    def _get_remaining_structure(self, mol: Chem.Mol, substructures: List[Dict]) -> Dict:
        """获取剩余未定义结构"""
        used_atoms = set()
        for sub in substructures:
            used_atoms.update(sub['atoms'])

        remaining_atoms = [idx for idx in range(mol.GetNumAtoms()) if idx not in used_atoms]

        # 分析剩余结构的连通分量
        components = self._find_connected_components(mol, remaining_atoms)

        return {
            'atoms': remaining_atoms,
            'components': components
        }

    def _find_connected_components(self, mol: Chem.Mol, atoms: List[int]) -> List[List[int]]:
        """查找剩余原子的连通分量"""
        if not atoms:
            return []

        # 构建子图
        atom_set = set(atoms)
        visited = set()
        components = []

        for start in atoms:
            if start in visited:
                continue

            # BFS 查找连通分量
            component = []
            queue = [start]
            visited.add(start)

            while queue:
                current = queue.pop(0)
                component.append(current)

                # 查找邻居
                for neighbor in mol.GetAtomWithIdx(current).GetNeighbors():
                    n_idx = neighbor.GetIdx()
                    if n_idx in atom_set and n_idx not in visited:
                        visited.add(n_idx)
                        queue.append(n_idx)

            components.append(component)

        return components

    def _analyze_symmetry(self, mol: Chem.Mol) -> Dict:
        """分析分子对称性"""
        if self.method == 'igraph':
            return self._analyze_symmetry_igraph(mol)
        else:
            return self._analyze_symmetry_rdkit(mol)

    def _analyze_symmetry_igraph(self, mol: Chem.Mol) -> Dict:
        """使用 igraph 分析对称性（改进版，兼容性更好）"""
        try:
            # 构建 igraph 图
            g = ig.Graph()

            # 添加顶点，带颜色标签用于区分不同原子类型
            colors = []
            for atom in mol.GetAtoms():
                # 原子类型编码用于着色
                color = self._get_atom_color(atom)
                colors.append(color)

            g.add_vertices(mol.GetNumAtoms())

            # 添加边
            edges = []
            for bond in mol.GetBonds():
                edges.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
            g.add_edges(edges)

            # 使用颜色作为顶点属性
            g.vs['color'] = colors

            # 方法1: 尝试计算自同构群（需要 nauty 支持）
            try:
                # 注意：不同版本的 igraph API 不同
                if hasattr(g, 'automorphism_group'):
                    # 尝试获取轨道（需要 nauty）
                    # 有些版本支持 color 参数
                    aut = g.automorphism_group(color=colors)

                    # 尝试从 aut 对象获取轨道
                    orbits = self._extract_orbits_from_automorphism(aut, g)
                    if orbits:
                        return self._build_equivalence_classes(orbits, mol)
            except Exception as e1:
                print(f"igraph automorphism failed: {e1}")

            # 方法2: 使用图同构的替代方法 - 基于规范标签
            # 计算每个节点的规范标签
            canonical_labels = self._compute_canonical_labels(g, colors)

            # 根据规范标签分组
            eq_groups = defaultdict(list)
            for idx, label in enumerate(canonical_labels):
                eq_groups[label].append(idx)

            return {
                'method': 'igraph_canonical',
                'equivalence_classes': {f"E{i}": atoms for i, atoms in enumerate(eq_groups.values())},
                'num_classes': len(eq_groups)
            }

        except Exception as e:
            print(f"igraph symmetry analysis failed, falling back to RDKit: {e}")
            return self._analyze_symmetry_rdkit(mol)

    def _get_atom_color(self, atom) -> int:
        """获取原子颜色编码"""
        # 编码原子类型：原子序数 * 100 + 形式电荷 + 氢原子数
        return (atom.GetAtomicNum() * 100 +
                (atom.GetFormalCharge() + 5) * 10 +
                min(atom.GetTotalNumHs(), 5))

    def _compute_canonical_labels(self, g: ig.Graph, colors: List[int]) -> List[int]:
        """计算图的规范标签（基于 Weisfeiler-Lehman 算法）"""
        n = g.vcount()

        # 初始标签为颜色
        labels = colors.copy()

        # 迭代细化标签
        max_iter = n
        for _ in range(max_iter):
            # 收集每个节点的邻居标签
            new_labels = []
            for i in range(n):
                neighbor_labels = sorted([labels[neighbor] for neighbor in g.neighbors(i)])
                # 组合当前标签和邻居标签
                label_tuple = (labels[i], tuple(neighbor_labels))
                new_labels.append(hash(str(label_tuple)))

            # 检查是否收敛
            if new_labels == labels:
                break
            labels = new_labels

        # 规范化标签（映射到连续整数）
        unique_labels = {}
        canonical = []
        next_label = 0
        for label in labels:
            if label not in unique_labels:
                unique_labels[label] = next_label
                next_label += 1
            canonical.append(unique_labels[label])

        return canonical

    def _extract_orbits_from_automorphism(self, aut, g: ig.Graph) -> Optional[List[List[int]]]:
        """从自同构群对象中提取轨道"""
        # 尝试不同的 API
        if hasattr(aut, 'orbits'):
            return aut.orbits()
        elif hasattr(aut, 'orbit'):
            # 计算每个节点的轨道
            n = g.vcount()
            visited = set()
            orbits = []
            for i in range(n):
                if i not in visited:
                    orbit = aut.orbit(i)
                    orbits.append(list(orbit))
                    visited.update(orbit)
            return orbits
        return None

    def _build_equivalence_classes(self, orbits: List[List[int]], mol: Chem.Mol) -> Dict:
        """构建等价类字典"""
        eq_classes = {}
        for idx, orbit in enumerate(orbits):
            # 过滤掉虚拟节点（如果有）
            real_atoms = [atom for atom in orbit if atom < mol.GetNumAtoms()]
            if real_atoms:
                eq_classes[f"E{idx}"] = real_atoms
        return {
            'method': 'igraph_automorphism',
            'equivalence_classes': eq_classes,
            'num_classes': len(eq_classes)
        }

    def _analyze_symmetry_rdkit(self, mol: Chem.Mol) -> Dict:
        """使用 RDKit 分析对称性（稳定方法）"""
        try:
            # 使用 RDKit 的规范原子排名
            ranks = CanonicalRankAtoms(mol, breakTies=False)

            eq_classes = defaultdict(list)
            for idx, rank in enumerate(ranks):
                eq_classes[f"E{rank}"].append(idx)

            return {
                'method': 'rdkit',
                'equivalence_classes': dict(eq_classes),
                'num_classes': len(eq_classes)
            }
        except Exception as e:
            print(f"RDKit symmetry analysis failed: {e}")
            # 降级：每个原子都是独立的等价类
            eq_classes = {f"E{i}": [i] for i in range(mol.GetNumAtoms())}
            return {
                'method': 'fallback',
                'equivalence_classes': eq_classes,
                'num_classes': len(eq_classes)
            }

    def _generate_tokens(self, mol: Chem.Mol, substructures: List[Dict],
                        remaining: Dict, symmetry_info: Dict) -> str:
        """生成 token 字符串"""
        tokens = []

        # 获取原子索引到等价类的映射
        atom_to_eq = {}
        if symmetry_info and 'equivalence_classes' in symmetry_info:
            for eq_class, atoms in symmetry_info['equivalence_classes'].items():
                for atom_idx in atoms:
                    atom_to_eq[atom_idx] = eq_class

        # 处理功能子结构
        for sub in substructures:
            sub_tokens = self._substructure_to_tokens(mol, sub, atom_to_eq)
            tokens.append(f"{{ {sub_tokens} }}")

        # 处理剩余结构
        if remaining['components']:
            for comp in remaining['components']:
                comp_tokens = self._component_to_tokens(mol, comp, atom_to_eq)
                tokens.append(f"[REM {{ {comp_tokens} }}]")

        return " ".join(tokens)

    def _substructure_to_tokens(self, mol: Chem.Mol, substructure: Dict,
                               atom_to_eq: Dict) -> str:
        """将子结构转换为 token 字符串"""
        atoms = substructure['atoms']
        name = substructure['name']

        # 构建子结构的连接关系
        token_parts = [f"sub[{name}]"]

        # 添加原子和键信息
        for atom_idx in atoms:
            atom = mol.GetAtomWithIdx(atom_idx)
            eq_class = atom_to_eq.get(atom_idx, f"A{atom_idx}")
            atom_symbol = atom.GetSymbol()

            # 获取键信息
            bonds = []
            for neighbor in atom.GetNeighbors():
                if neighbor.GetIdx() in atoms:  # 只考虑子结构内部的键
                    bond = mol.GetBondBetweenAtoms(atom_idx, neighbor.GetIdx())
                    bond_type = self._bond_type_to_str(bond.GetBondType())
                    bonds.append(f"{bond_type}{neighbor.GetIdx()}")

            bond_str = "-".join(bonds) if bonds else ""
            token_parts.append(f"atom[{atom_symbol}_{eq_class}]({bond_str})")

        return " ".join(token_parts)

    def _component_to_tokens(self, mol: Chem.Mol, atoms: List[int],
                            atom_to_eq: Dict) -> str:
        """将剩余连通分量转换为 token 字符串"""
        token_parts = ["rem"]

        for atom_idx in atoms:
            atom = mol.GetAtomWithIdx(atom_idx)
            eq_class = atom_to_eq.get(atom_idx, f"A{atom_idx}")
            atom_symbol = atom.GetSymbol()

            # 获取所有邻居（包括可能连接到功能子结构的）
            bonds = []
            for neighbor in atom.GetNeighbors():
                bond = mol.GetBondBetweenAtoms(atom_idx, neighbor.GetIdx())
                bond_type = self._bond_type_to_str(bond.GetBondType())

                # 标记外部连接
                if neighbor.GetIdx() not in atoms:
                    bonds.append(f"ext_{bond_type}")
                else:
                    bonds.append(f"{bond_type}{neighbor.GetIdx()}")

            bond_str = "-".join(bonds) if bonds else ""
            token_parts.append(f"atom[{atom_symbol}_{eq_class}]({bond_str})")

        return " ".join(token_parts)

    def _bond_type_to_str(self, bond_type) -> str:
        """将键类型转换为字符串"""
        from rdkit.Chem.rdchem import BondType

        mapping = {
            BondType.SINGLE: "-",
            BondType.DOUBLE: "=",
            BondType.TRIPLE: "#",
            BondType.AROMATIC: ":",
        }
        return mapping.get(bond_type, "?")

    def _calculate_properties(self, mol: Chem.Mol) -> Dict:
        """计算分子性质"""
        try:
            properties = {
                'molecular_weight': Descriptors.MolWt(mol),
                'logP': Descriptors.MolLogP(mol),
                'TPSA': Descriptors.TPSA(mol),
                'num_h_donors': Lipinski.NumHDonors(mol),
                'num_h_acceptors': Lipinski.NumHAcceptors(mol),
                'num_rotatable_bonds': Lipinski.NumRotatableBonds(mol),
                'num_rings': Descriptors.RingCount(mol),
            }
        except:
            properties = {'error': 'Property calculation failed'}

        return properties


# ============= 使用示例 =============

def main():
    """主函数示例"""
    # 示例 SMILES 列表
    smiles_list = [
        "Oc1ccccc1",  # 苯酚
        "c1ccccc1",    # 苯
        "CC(=O)O",     # 乙酸
        "c1ccncc1",    # 吡啶
        "Nc1ccccc1",   # 苯胺
        "Oc1ccc(O)cc1",  # 对苯二酚
    ]

    # 初始化 tokenizer（默认使用 RDKit 方法，稳定可靠）
    tokenizer = SymmetryAwareTokenizer(use_symmetry=True, method='rdkit')

    # 处理所有分子
    results = tokenizer.process_smiles_list(smiles_list)

    # 输出结果
    print("=" * 80)
    print("分子 Tokenization 结果")
    print("=" * 80)

    for result in results:
        if 'error' in result:
            print(f"\n[{result['index']}] ERROR: {result['smiles']}")
            print(f"  Error: {result['error']}")
            continue

        print(f"\n[{result['index']}] {result['smiles']}")
        print(f"  Canonical: {result['canonical_smiles']}")
        print(f"  Tokens: {result['tokens'][:150]}...")
        print(f"  Symmetry method: {result['symmetry_info'].get('method', 'N/A')}")
        print(f"  Symmetry: {result['symmetry_info']['num_classes']} equivalence classes")

        # 显示部分等价类
        eq_classes = result['symmetry_info'].get('equivalence_classes', {})
        if eq_classes:
            sample_eq = list(eq_classes.items())[:3]
            eq_str = ", ".join([f"{k}:{v}" for k, v in sample_eq])
            print(f"  Equiv classes: {eq_str}...")

        print(f"  Properties: MW={result['properties'].get('molecular_weight', 0):.2f}, "
              f"LogP={result['properties'].get('logP', 0):.2f}")


def test_symmetry_comparison():
    """测试对称性分析的效果"""
    smiles_list = [
        "CS(=O)(=O)C1=CC=C(N2CCN(CC3=CC=C(NS(=O)(=O)C4=CC=NC5=CC=CN=C54)C=C3)CC2)C=C1",      # 苯酚 - 不对称取代
        "c1ccccc1",       # 苯 - 完全对称
        "Oc1ccc(O)cc1",   # 对苯二酚 - 对称取代
        "Nc1cc(N)ccc1",   # 间苯二胺 - 对称取代
    ]

    tokenizer = SymmetryAwareTokenizer(use_symmetry=True, method='rdkit')

    print("\n" + "=" * 80)
    print("对称性比较测试")
    print("=" * 80)

    for smiles in smiles_list:
        result = tokenizer.tokenize_molecule(smiles)

        print(f"\n{smiles}")
        print(f"  Canonical: {result['canonical_smiles']}")

        # 显示原子等价类
        eq_classes = result['symmetry_info']['equivalence_classes']
        print(f"  Equivalence classes ({len(eq_classes)}):")

        # 只显示前几个类
        for eq_class, atoms in list(eq_classes.items())[:5]:
            # 显示原子的元素符号
            mol = Chem.MolFromSmiles(result['canonical_smiles'])
            atom_symbols = [mol.GetAtomWithIdx(a).GetSymbol() for a in atoms]
            print(f"    {eq_class}: {atoms} ({atom_symbols})")

        if len(eq_classes) > 5:
            print(f"    ... and {len(eq_classes) - 5} more classes")


if __name__ == "__main__":
    # 确保已安装所需库
    # pip install rdkit-pypi igraph python-igraph

    main()
    test_symmetry_comparison()