from rdkit import Chem
from rdkit.Chem import rdmolops, QED
from collections import defaultdict
from decompose.Mol2SMILES import VirtualAtomSMILESGenerator
from rdkit.Contrib.SA_Score import sascorer
from rdkit.Chem.Fingerprints.ClusterMols import message

from constant import ele_num


class MolecularDecomposer:
    def __init__(self):
        # 预定义结构
        self.predefined_patterns = {
        }

        # 全局连接计数器
        self.connection_counter = 1
        self.connection_map = {}

    def reset_predefined_patterns(self, smiles: list):
        self.predefined_patterns = {}
        for s in smiles:
            mol = Chem.MolFromSmarts(s)
            Chem.Kekulize(mol, True)
            self.predefined_patterns[s] = mol

    def reset_connection_counter(self):
        """重置连接计数器"""
        self.connection_counter = 1
        self.connection_map = {}

    def find_ring_systems(self, mol):
        """识别分子中的环系统"""
        ring_info = mol.GetRingInfo()
        atom_rings = list(ring_info.AtomRings())

        if not atom_rings:
            return []

        # 构建环系统的连接关系
        ring_connections = defaultdict(set)
        for i, ring1 in enumerate(atom_rings):
            for j, ring2 in enumerate(atom_rings):
                if i < j and set(ring1) & set(ring2):
                    ring_connections[i].add(j)
                    ring_connections[j].add(i)

        # 识别环系统
        visited = set()
        ring_systems = []

        for i in range(len(atom_rings)):
            if i not in visited:
                system = []
                stack = [i]
                visited.add(i)

                while stack:
                    current = stack.pop()
                    system.append(current)
                    for neighbor in ring_connections[current]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            stack.append(neighbor)

                # 获取环系统的原子集合
                system_atoms = set()
                for ring_idx in system:
                    system_atoms.update(atom_rings[ring_idx])
                ring_systems.append(system_atoms)

        return ring_systems

    def find_predefined_structures(self, mol, excluded_atoms):
        """查找预定义结构"""
        patterns_found = []

        for pattern_name, pattern in self.predefined_patterns.items():
            matches = mol.GetSubstructMatches(pattern)
            for match in matches:
                if not any(atom_idx in excluded_atoms for atom_idx in match):
                    patterns_found.append({
                        'name': pattern_name,
                        'atoms': set(match)
                    })
                    excluded_atoms.update(match)

        return patterns_found

    def find_remaining_structures(self, mol, allocated_atoms):
        """查找剩余的未定义结构"""
        all_atoms = set(range(mol.GetNumAtoms()))
        remaining_atoms = all_atoms - allocated_atoms

        if not remaining_atoms:
            return []

        # 将剩余原子分组为连接片段
        remaining_structures = []
        visited = set()

        for start_atom in remaining_atoms:
            if start_atom not in visited:
                fragment_atoms = self._find_connected_fragment(mol, start_atom, remaining_atoms)
                visited.update(fragment_atoms)

                if fragment_atoms:
                    remaining_structures.append({
                        'name': 'undefined_fragment',
                        'atoms': fragment_atoms
                    })

        return remaining_structures

    def _find_connected_fragment(self, mol, start_atom, allowed_atoms):
        """找到连接的原子片段"""
        fragment = set()
        stack = [start_atom]

        while stack:
            atom_idx = stack.pop()
            if atom_idx in allowed_atoms and atom_idx not in fragment:
                fragment.add(atom_idx)
                atom = mol.GetAtomWithIdx(atom_idx)
                for neighbor in atom.GetNeighbors():
                    neighbor_idx = neighbor.GetIdx()
                    if neighbor_idx in allowed_atoms and neighbor_idx not in fragment:
                        stack.append(neighbor_idx)

        return fragment

    def analyze_connections(self, mol, structure_atoms, structure_type, structure_id):
        """分析结构与外部的连接，并分配连接ID"""
        connections = []

        for atom_idx in structure_atoms:
            atom = mol.GetAtomWithIdx(atom_idx)
            for neighbor in atom.GetNeighbors():
                neighbor_idx = neighbor.GetIdx()
                if neighbor_idx not in structure_atoms:
                    bond = mol.GetBondBetweenAtoms(atom_idx, neighbor_idx)

                    # 创建连接键的唯一标识符
                    bond_key = tuple(sorted([atom_idx, neighbor_idx]))

                    # 如果这个连接还没有分配ID，分配一个新的
                    if bond_key not in self.connection_map:
                        self.connection_map[bond_key] = self.connection_counter
                        self.connection_counter += 1

                    connection_id = self.connection_map[bond_key]

                    connection_info = {
                        'internal_atom_idx': atom_idx,
                        'internal_atom_symbol': atom.GetSymbol(),
                        'external_atom_idx': neighbor_idx,
                        'bond_type': bond.GetBondType().name,
                        'connection_id': connection_id
                    }
                    connections.append(connection_info)

        return connections

    def create_fragment_with_dummies(self, mol, atom_indices, connections, fragment_id, link_index):
        """创建带有编号虚拟原子的片段"""
        if not atom_indices:
            return None

        try:
            # 创建分子片段
            frag_mol = Chem.RWMol()
            atom_map = {}

            # 添加真实原子
            for atom_idx in atom_indices:
                atom = mol.GetAtomWithIdx(atom_idx)
                new_atom = Chem.Atom(atom.GetAtomicNum())
                new_atom.SetFormalCharge(atom.GetFormalCharge())
                # new_atom.SetIsAromatic(atom.GetIsAromatic())
                new_atom.SetProp('_IsAromatic', atom.GetProp('_IsAromatic'))
                new_atom.SetProp('_NumEHs', atom.GetProp('_NumEHs'))
                new_idx = frag_mol.AddAtom(new_atom)
                atom_map[atom_idx] = new_idx

            # 添加内部键
            for atom_idx in atom_indices:
                atom = mol.GetAtomWithIdx(atom_idx)
                for neighbor in atom.GetNeighbors():
                    neighbor_idx = neighbor.GetIdx()
                    if neighbor_idx in atom_indices and neighbor_idx > atom_idx:
                        bond = mol.GetBondBetweenAtoms(atom_idx, neighbor_idx)
                        if bond:
                            frag_mol.AddBond(atom_map[atom_idx], atom_map[neighbor_idx], bond.GetBondType())

            dummy_atom_map = {}

            for conn in connections:
                conn['atom_num'] = atom_map[conn['internal_atom_idx']]

            m2s = VirtualAtomSMILESGenerator()
            s = m2s.generate_smiles_with_virtual_atoms(frag_mol.GetMol(), connections)

            return s, Chem.MolToSmiles(frag_mol, canonical=True)

        except Exception as e:
            print(f"创建片段时出错: {e}")
            return 'J'

    def mol_dump(self, mol):
        if mol is None:
            return None
        symbols = {g.GetSymbol() + str(g.GetIdx()) : [f.GetSymbol() + str(f.GetIdx()) + '_' + str(mol.GetBondBetweenAtoms(g.GetIdx(),f.GetIdx()).GetBondType()) for f in g.GetNeighbors()] for g in mol.GetAtoms()}

        return symbols

    def _manual_fragment_smiles(self, mol, atom_indices, connections, link_index):
        """手动构建带连接编号的片段SMILES"""
        try:
            # 创建分子片段但不添加虚拟原子
            frag_mol = Chem.RWMol()
            atom_map = {}
            atom_map_reverse = {}

            # 添加真实原子
            for atom_idx in atom_indices:
                atom = mol.GetAtomWithIdx(atom_idx)
                new_atom = Chem.Atom(atom.GetAtomicNum())
                new_atom.SetFormalCharge(atom.GetFormalCharge())
                # new_atom.SetIsAromatic(atom.GetIsAromatic())
                new_atom.SetProp('_IsAromatic', atom.GetProp('_IsAromatic'))
                new_atom.SetProp('_NumEHs', atom.GetProp('_NumEHs'))

                new_idx = frag_mol.AddAtom(new_atom)
                atom_map[atom_idx] = new_idx
                atom_map_reverse[new_idx] = atom_idx

            # 添加内部键
            for atom_idx in atom_indices:
                atom = mol.GetAtomWithIdx(atom_idx)
                for neighbor in atom.GetNeighbors():
                    neighbor_idx = neighbor.GetIdx()
                    if neighbor_idx in atom_indices and neighbor_idx > atom_idx:
                        bond = mol.GetBondBetweenAtoms(atom_idx, neighbor_idx)
                        if bond:
                            frag_mol.AddBond(atom_map[atom_idx], atom_map[neighbor_idx], bond.GetBondType())

            for conn in connections:
                conn['atom_num'] = atom_map[conn['internal_atom_idx']]
                # atom_new_index = atom_map[conn['internal_atom_idx']]

            m2s = VirtualAtomSMILESGenerator()
            s = m2s.generate_smiles_with_virtual_atoms(frag_mol.GetMol(), connections)

            return s

        except Exception as e:
            print(f"手动构建SMILES时出错: {e}")
            return 'J'

    def decompose_molecule(self, smiles):
        """主分解函数"""
        self.reset_connection_counter()  # 重置连接计数器
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            mol = Chem.MolFromSmarts(smiles)
            if mol is None:
                return None

        for atom in mol.GetAtoms():
            is_aromatic = atom.GetIsAromatic()
            num_explicit_hs = atom.GetNumExplicitHs()
            atom.SetProp('_IsAromatic', str(is_aromatic))
            atom.SetProp('_NumEHs', str(num_explicit_hs))

        Chem.Kekulize(mol, True)

        # 步骤1: 识别环系统
        ring_systems = self.find_ring_systems(mol)
        allocated_atoms = set()
        all_structures = []

        # 处理环系统
        for i, ring_atoms in enumerate(ring_systems):
            connections = self.analyze_connections(mol, ring_atoms, 'ring_system', i)
            all_structures.append({
                'type': 'ring_system',
                'id': i,
                'atoms': ring_atoms,
                'connections': connections,
                'name': f'ring_system_{i + 1}'
            })
            allocated_atoms.update(ring_atoms)

        # 步骤2: 识别预定义结构
        predefined_structures = self.find_predefined_structures(mol, allocated_atoms)

        for i, structure in enumerate(predefined_structures):
            connections = self.analyze_connections(mol, structure['atoms'], 'predefined', i)
            all_structures.append({
                'type': 'predefined',
                'id': i,
                'atoms': structure['atoms'],
                'connections': connections,
                'name': structure['name']
            })
            allocated_atoms.update(structure['atoms'])

        # 步骤3: 识别剩余结构
        remaining_structures = self.find_remaining_structures(mol, allocated_atoms)

        for i, structure in enumerate(remaining_structures):
            connections = self.analyze_connections(mol, structure['atoms'], 'remaining', i)
            all_structures.append({
                'type': 'remaining',
                'id': i,
                'atoms': structure['atoms'],
                'connections': connections,
                'name': structure['name']
            })

        return mol, all_structures

    def generate_fragment_smiles(self, mol, structures, link_index):
        """为所有结构生成带有连接标记的SMILES"""
        fragment_results = []

        for structure in structures:
            smiles, o_smiles = self.create_fragment_with_dummies(
                mol,
                list(structure['atoms']),
                structure['connections'],
                f"{structure['type']}_{structure['id']}",
                link_index=link_index
            )

            if smiles:
                # 收集连接信息
                connection_info = []
                for conn in structure['connections']:
                    connection_info.append({
                        'connection_id': conn['connection_id'],
                        'internal_atom': f"{conn['internal_atom_symbol']}{conn['internal_atom_idx']}",
                        'external_atom_idx': conn['external_atom_idx'],
                        'bond_type': conn['bond_type']
                    })

                fragment_results.append({
                    'type': structure['type'],
                    'name': structure['name'],
                    'smiles': smiles,
                    'smiles_wo_index': o_smiles,
                    'connections': connection_info
                })

        return fragment_results

def get_formula(mol):
    symbols = {}
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol in symbols:
            symbols[symbol] += 1
        else:
            symbols[symbol] = 1

    result = []

    for e in ele_num:
        if e in symbols:
            result.append((e, symbols[e]))

    return result

def _get_prop(mol):
    qed = QED.qed(mol)
    props = QED.properties(mol)
    score = sascorer.calculateScore(mol)
    score = (10 - score) / 9

    return [qed,props.ALOGP,score]

    props = QED.properties(mol)
    return [props.MW, props.ALOGP, props.HBA, props.HBD, props.PSA, props.ROTB, props.AROM, props.ALERTS]

def decompose_smiles(smiles_list, predefined_structures = [], link_index = False, prop_calu = True):
    """主函数：分解多个SMILES字符串"""
    decomposer = MolecularDecomposer()
    if len(predefined_structures) > 0:
        decomposer.reset_predefined_patterns(predefined_structures)

    results = []


    for smiles_t in smiles_list:
        try:
            mol, structures = decomposer.decompose_molecule(smiles_t)
            if mol and structures:
                fragments = decomposer.generate_fragment_smiles(mol, structures, link_index=link_index)
                prop = []
                if prop_calu:
                    prop = _get_prop(mol)

                results.append({
                    'original_smiles': smiles_t,
                    'fragments': fragments,
                    'formula': get_formula(mol),
                    'prop': prop
                })

        except Exception as e:
            print(f"处理 {smiles_t} 时出错: {e}")
            results.append({
                'original_smiles': None,
                'fragments': None,
                'formula': None,
                'prop': None
            })
            continue

    return results


def find_bracketed_content(A, B):
    """
    判断字符串A是否在字符串B中被一对方括号包括，若是则返回这对方括号中的所有内容

    Args:
        A: 要查找的字符串
        B: 被搜索的字符串

    Returns:
        str or None: 如果找到则返回方括号中的完整内容，否则返回None
    """
    # 如果A不在B中，直接返回None
    if A not in B:
        return None

    # 查找A在B中的所有位置
    positions = []
    start = 0
    while True:
        pos = B.find(A, start)
        if pos == -1:
            break
        positions.append(pos)
        start = pos + 1

    # 对每个位置检查是否被方括号包围
    for pos in positions:
        start_pos = pos
        end_pos = pos + len(A) - 1

        # 向左查找开括号 '['
        left_bracket_pos = -1
        for i in range(start_pos - 1, -1, -1):
            if B[i] == '[':
                left_bracket_pos = i
                break
            elif B[i] == ']':  # 遇到闭括号说明不在同一对括号内
                break

        # 向右查找闭括号 ']'
        right_bracket_pos = -1
        for i in range(end_pos + 1, len(B)):
            if B[i] == ']':
                right_bracket_pos = i
                break
            elif B[i] == '[':  # 遇到开括号说明不在同一对括号内
                break

        # 如果找到了匹配的方括号对，且A在它们之间
        if left_bracket_pos != -1 and right_bracket_pos != -1:
            # 返回完整的括号内容
            return B[left_bracket_pos:right_bracket_pos + 1]

    return None


def find_all_bracketed_content(A, B):
    """
    查找字符串A在字符串B中所有被方括号包括的情况

    Args:
        A: 要查找的字符串
        B: 被搜索的字符串

    Returns:
        list: 包含所有匹配的方括号内容的列表
    """
    results = []

    if A not in B:
        return results

    # 查找A在B中的所有位置
    positions = []
    start = 0
    while True:
        pos = B.find(A, start)
        if pos == -1:
            break
        positions.append(pos)
        start = pos + 1

    # 对每个位置检查是否被方括号包围
    for pos in positions:
        start_pos = pos
        end_pos = pos + len(A) - 1

        # 向左查找开括号 '['
        left_bracket_pos = -1
        for i in range(start_pos - 1, -1, -1):
            if B[i] == '[':
                left_bracket_pos = i
                break
            elif B[i] == ']':  # 遇到闭括号说明不在同一对括号内
                break

        # 向右查找闭括号 ']'
        right_bracket_pos = -1
        for i in range(end_pos + 1, len(B)):
            if B[i] == ']':
                right_bracket_pos = i
                break
            elif B[i] == '[':  # 遇到开括号说明不在同一对括号内
                break

        # 如果找到了匹配的方括号对，且A在它们之间
        if left_bracket_pos != -1 and right_bracket_pos != -1:
            content = B[left_bracket_pos:right_bracket_pos + 1]
            if content not in results:  # 避免重复
                results.append(content)

    return results


def find_bracketed_content_with_regex(A, B):
    """
    使用正则表达式方法查找被方括号包括的字符串A

    Args:
        A: 要查找的字符串
        B: 被搜索的字符串

    Returns:
        str or None: 如果找到则返回方括号中的完整内容，否则返回None
    """
    import re

    # 转义A中的特殊字符
    escaped_A = re.escape(A)

    # 构建正则表达式：匹配包含A的方括号对
    pattern = r'\[[^\]]*?' + escaped_A + r'[^\[]*?\]'

    matches = re.findall(pattern, B)

    return matches[0] if matches else None


# 使用示例
if __name__ == "__main__":
    # 测试分子
    test_smiles = [
        'CC12CCC(=O)CC1CCC3C2CCC4(C3CCC4O)C'
    ]

    results = decompose_smiles(test_smiles)

    # 输出结果
    for result in results:
        print(f"\n原始分子: {result['original_smiles']}")
        print("分解片段:")
        for frag in result['fragments']:
            print(f"  {frag['type']:12} {frag['name']:20} {frag['smiles']}")
            if frag['connections']:
                print("    连接位点:")
                for conn in frag['connections']:
                    print(f"      [*:{conn['connection_id']}] -> 原子{conn['external_atom_idx']} ({conn['bond_type']})")