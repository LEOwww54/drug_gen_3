from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from collections import defaultdict
from decompose_1.translator import smiles2token


class VirtualAtomConnectionProcessor:
    def __init__(self):
        self.original_mol = None
        self.cleaned_mol = None
        self.atom_connections = {}
        self.virtual_atoms_info = {}
        self.mol_with_virtual = None
        self.mol_cleaned = None
        self.cleaned_smiles = ''
        self.atom_map = {}

    def process_mol(self, mol, mol_props):
        """
        处理分子对象的主函数

        Args:
            mol: RDKit分子对象，包含虚拟原子（同位素标记的*）
                 原子属性中应包含 '_symmetry' 字段

        Returns:
            dict: 包含处理结果的字典
        """
        if mol_props is not None:
            map = {}
            count = 0
            for i, j in mol_props.items():
                for pname, pvalue in j.items():
                    if pname == '_symmetry':
                        target = str(pvalue)
                        if not target in map:
                            map[target] = count
                            count += 1
                        mol.GetAtomWithIdx(i).SetProp(pname, str(map[target]))
                    else:
                        mol.GetAtomWithIdx(i).SetProp(pname, str(pvalue))

        if mol is None:
            print("错误: 输入的分子对象为空")
            return None

        self.original_mol = mol

        # 分析分子中的连接
        result = self.analyze_connections_from_mol(mol)

        if result:
            connections, cleaned_smiles, tokens = result

            result_dict = {
                'original_mol': mol,
                'cleaned_mol': self.mol_cleaned,
                'cleaned_smiles': cleaned_smiles,
                'connections': connections,
                'virtual_atoms_found': list(self.virtual_atoms_info.keys()),
                'tokens': tokens,
                'atom_mapping': self.atom_map
            }

            return result_dict
        else:
            return None

    def analyze_connections_from_mol(self, mol):
        """
        从分子对象分析连接信息

        Args:
            mol: 包含虚拟原子的分子对象

        Returns:
            tuple: (connections, cleaned_smiles, tokens)
        """
        try:
            # 确保分子被Kekulize
            mol_copy = Chem.Mol(mol)
            Chem.Kekulize(mol_copy, True)

            # 提取虚拟原子信息
            virtual_atoms = self.extract_virtual_atoms_from_mol(mol_copy)

            # 移除虚拟原子
            cleaned_mol = self.remove_virtual_atoms(mol_copy)

            # 为清理后的分子设置原子映射编号
            # 注意：_symmetry属性直接从原始分子复制，不重新计算
            for atom in cleaned_mol.GetAtoms():
                # 获取对应的原始原子索引
                orig_idx = None
                for o_idx, c_idx in self.atom_map.items():
                    if c_idx == atom.GetIdx():
                        orig_idx = o_idx
                        break

                # 从原始分子复制_symmetry属性
                if orig_idx is not None:
                    orig_atom = mol_copy.GetAtomWithIdx(orig_idx)
                    if orig_atom.HasProp("_symmetry"):
                        symmetry_value = orig_atom.GetIntProp("_symmetry")
                        atom.SetIntProp("_symmetry", symmetry_value)
                    else:
                        # 如果原始原子没有_symmetry属性，设置默认值
                        atom.SetIntProp("_symmetry", 0)
                else:
                    atom.SetIntProp("_symmetry", 0)

                # 设置原子映射编号
                if atom.GetAtomicNum() != 1:
                    atom.SetAtomMapNum(atom.GetIdx() + 1)

            # 生成清理后的SMILES
            self.cleaned_smiles = Chem.MolToSmiles(cleaned_mol, allHsExplicit=False)

            # 构建连接信息
            connections = self._build_connections_from_mol(mol_copy, virtual_atoms, cleaned_mol)

            # 获取氢原子信息和形式电荷
            IH = self._get_hydrogen_and_charge_info(cleaned_mol)

            # 生成tokens
            tokens = self._generate_tokens_from_mol(cleaned_mol, connections, IH)

            self.mol_cleaned = cleaned_mol

            return connections, self.cleaned_smiles, tokens

        except Exception as e:
            print(f"分析连接时出错: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None

    def extract_virtual_atoms_from_mol(self, mol):
        """
        从分子对象中提取虚拟原子信息

        Args:
            mol: RDKit分子对象

        Returns:
            list: 虚拟原子信息列表
        """
        virtual_atoms = []
        self.virtual_atoms_info = {}

        # 遍历所有原子
        for atom in mol.GetAtoms():
            # 检查是否为虚拟原子（符号为*且有同位素标记）
            if atom.GetSymbol() == '*' and atom.GetIsotope() > 0:
                virtual_number = atom.GetIsotope()
                neighbors = atom.GetNeighbors()

                virtual_info = {
                    'virtual_number': virtual_number,
                    'atom_idx': atom.GetIdx(),
                    'neighbors': []
                }

                # 获取连接的原子信息
                for neighbor in neighbors:
                    bond = mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
                    bond_symbol = self.get_bond_symbol(bond)

                    virtual_info['neighbors'].append({
                        'neighbor_idx': neighbor.GetIdx(),
                        'neighbor_symbol': neighbor.GetSymbol(),
                        'bond_type': bond.GetBondType(),
                        'bond_symbol': bond_symbol
                    })

                virtual_atoms.append(virtual_info)
                self.virtual_atoms_info[virtual_number] = virtual_info

        return virtual_atoms

    def remove_virtual_atoms(self, mol):
        """
        从分子中移除所有虚拟原子，保持其他原子序号不变

        Args:
            mol: RDKit分子对象

        Returns:
            RDKit.Mol: 移除虚拟原子后的分子
        """
        # 创建可编辑分子
        rw_mol = Chem.RWMol(mol)

        # 收集所有虚拟原子的索引（按降序排列，以便安全删除）
        virtual_indices = []
        for atom in rw_mol.GetAtoms():
            if atom.GetSymbol() == '*' and atom.GetIsotope() > 0:
                virtual_indices.append(atom.GetIdx())

        # 按降序删除虚拟原子
        for idx in sorted(virtual_indices, reverse=True):
            rw_mol.RemoveAtom(idx)

        # 创建清理后的分子
        self.cleaned_mol = rw_mol.GetMol()

        # 建立原子映射：原始索引 -> 清理后索引
        self._build_atom_mapping(mol, self.cleaned_mol)

        return self.cleaned_mol

    def _build_atom_mapping(self, original_mol, cleaned_mol):
        """
        建立原始分子到清理后分子的原子索引映射

        Args:
            original_mol: 原始分子
            cleaned_mol: 清理后的分子
        """
        # 使用子结构匹配找到映射
        matches = original_mol.GetSubstructMatches(cleaned_mol)

        if matches:
            # 取第一个匹配
            match = matches[0]
            # 创建映射：原始索引 -> 清理后索引
            self.atom_map = {orig_idx: clean_idx for clean_idx, orig_idx in enumerate(match)}
        else:
            # 如果子结构匹配失败，尝试直接映射（当分子结构简单时）
            self.atom_map = {}
            orig_idx = 0
            clean_idx = 0
            while orig_idx < original_mol.GetNumAtoms():
                atom = original_mol.GetAtomWithIdx(orig_idx)
                if atom.GetSymbol() == '*' and atom.GetIsotope() > 0:
                    orig_idx += 1
                    continue
                self.atom_map[orig_idx] = clean_idx
                orig_idx += 1
                clean_idx += 1

    def get_cleaned_atom_index(self, original_idx):
        """
        获取清理后分子中对应的原子索引

        Args:
            original_idx: 原始分子中的原子索引

        Returns:
            int: 清理后分子中的原子索引，如果找不到则返回None
        """
        return self.atom_map.get(original_idx)

    def _build_connections_from_mol(self, original_mol, virtual_atoms, cleaned_mol):
        """
        从分子构建连接信息

        Args:
            original_mol: 原始分子
            virtual_atoms: 虚拟原子信息列表
            cleaned_mol: 清理后的分子

        Returns:
            list: 连接信息列表
        """
        connections = []

        for v_info in virtual_atoms:
            virtual_number = v_info['virtual_number']

            for neighbor_info in v_info['neighbors']:
                original_idx = neighbor_info['neighbor_idx']

                # 获取清理后的原子索引
                cleaned_idx = self.get_cleaned_atom_index(original_idx)

                if cleaned_idx is not None:
                    atom = cleaned_mol.GetAtomWithIdx(cleaned_idx)
                    element_symbol = atom.GetSymbol()
                    atom_map_num = atom.GetAtomMapNum()

                    if atom_map_num == 0:
                        atom_map_num = cleaned_idx + 1

                    connection_info = {
                        'element_symbol': element_symbol,
                        'atom_number': atom_map_num,
                        'bond_symbol': neighbor_info['bond_symbol'],
                        'connection_number': virtual_number,
                        'formatted_string': f"{element_symbol} {atom_map_num}\t{neighbor_info['bond_symbol']}{virtual_number}",
                        'formatted_string_1': f'i{atom_map_num}{neighbor_info["bond_symbol"]}',
                        'formatted_string_2': f'i{virtual_number}',
                        'original_atom_idx': original_idx,
                        'cleaned_atom_idx': cleaned_idx
                    }
                    connections.append(connection_info)
                else:
                    print(f"警告: 无法映射原子 {original_idx}")

        return connections

    def _get_hydrogen_and_charge_info(self, mol):
        """
        获取分子的氢原子数量和形式电荷信息

        Args:
            mol: RDKit分子对象

        Returns:
            dict: 原子映射编号 -> 形式电荷
        """
        IH = {}
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() != 1:
                atom_map_num = atom.GetAtomMapNum()
                if atom_map_num == 0:
                    atom_map_num = atom.GetIdx() + 1
                IH[atom_map_num] = atom.GetFormalCharge()
        return IH

    def _generate_tokens_from_mol(self, mol, connections, IH):
        """
        从分子生成自定义token序列

        Args:
            mol: RDKit分子对象
            connections: 连接信息列表
            IH: 氢原子和电荷信息

        Returns:
            list: token序列
        """
        tokens = []
        tokens.append('{')

        # 获取原子的对称性信息（直接从分子属性读取，不重新计算）
        symmetry_map = {}
        for atom in mol.GetAtoms():
            if atom.HasProp("_symmetry"):
                symmetry_map[atom.GetIdx()] = atom.GetIntProp("_symmetry")
            else:
                symmetry_map[atom.GetIdx()] = -1

        text = {}
        for atom in mol.GetAtoms():
            token = []

            # 构建原子符号
            if atom.GetAtomicNum() in [6, 8]:  # C或O
                symbol = f'[{atom.GetSymbol()}]'
            else:
                total_h = atom.GetTotalNumHs()
                if total_h > 1:
                    symbol = f'[{atom.GetSymbol()}H{total_h}]'
                elif total_h == 1:
                    symbol = f'[{atom.GetSymbol()}H]'
                else:
                    symbol = f'[{atom.GetSymbol()}]'

            # 形式电荷
            formal_charge = f"<fc{str(atom.GetFormalCharge())}>"

            # 原子映射编号
            atom_index = atom.GetAtomMapNum()
            if atom_index == 0:
                atom_index = atom.GetIdx() + 1
                atom.SetAtomMapNum(atom_index)

            # 对称性（直接从属性读取）
            atom_sym = f"<sym{symmetry_map.get(atom.GetIdx(), 0)}>"

            # 自由基电子数
            atom_radical = f'<rad{atom.GetNumRadicalElectrons()}>'

            # 连接信息
            conn_info = []
            for connection in connections:
                if connection['atom_number'] == atom_index:
                    conn_info.append(f"<m{connection['bond_symbol']}")
                    conn_info.append(f"{connection['connection_number']}>")

            token.append(symbol)
            token.append(atom_radical)
            token.append(formal_charge)
            token.extend(conn_info)
            token.append(atom_sym)

            if atom.IsInRing():
                token.append('<r>')

            text[atom.GetIdx()] = token

        # 使用翻译器生成token序列
        tokens.extend(smiles2token(mol, text))
        tokens.append('}')

        return tokens

    def get_bond_symbol(self, bond):
        """
        获取化学键的符号表示
        """
        if bond is None:
            return '-'

        if bond.GetIsAromatic():
            return ':'
        elif bond.GetBondType() == Chem.BondType.SINGLE:
            return '-'
        elif bond.GetBondType() == Chem.BondType.DOUBLE:
            return '='
        elif bond.GetBondType() == Chem.BondType.TRIPLE:
            return '#'
        else:
            return '~'

    def format_final_output(self, result):
        """
        格式化最终输出

        Args:
            result: process_mol函数的返回结果字典

        Returns:
            tuple: (output_lines, tokens)
        """
        if not result:
            return None, None

        try:
            output_lines = []

            # 第一行: 处理后的SMILES（带原子编号）
            output_lines.append(result['cleaned_smiles'])

            # 连接信息
            sorted_connections = sorted(result['connections'], key=lambda x: int(x['atom_number']))

            for conn in sorted_connections:
                output_lines.append(conn['formatted_string_1'])
                output_lines.append(conn['formatted_string_2'])

            return output_lines, result['tokens']

        except Exception as e:
            print(f"格式化输出时出错: {e}")
            return None, None

    def get_detailed_atom_info(self, mol):
        """
        获取分子的详细原子信息，用于调试

        Args:
            mol: RDKit分子对象

        Returns:
            list: 原子信息列表
        """
        atom_info = []
        for atom in mol.GetAtoms():
            map_num = atom.GetAtomMapNum()
            is_virtual = atom.GetSymbol() == '*' and atom.GetIsotope() > 0

            # 获取_symmetry属性
            symmetry = 0
            if atom.HasProp("_symmetry"):
                symmetry = atom.GetIntProp("_symmetry")

            atom_info.append({
                'idx': atom.GetIdx(),
                'symbol': atom.GetSymbol(),
                'isotope': atom.GetIsotope(),
                'map_num': map_num,
                'is_virtual': is_virtual,
                'symmetry': symmetry,
                'neighbors': [n.GetIdx() for n in atom.GetNeighbors()],
                'formal_charge': atom.GetFormalCharge(),
                'total_hs': atom.GetTotalNumHs(),
                'is_in_ring': atom.IsInRing()
            })
        return atom_info

    def mol_dump(self, mol):
        """
        调试函数：输出分子结构信息

        Args:
            mol: RDKit分子对象

        Returns:
            dict: 分子结构信息
        """
        if mol is None:
            return None
        symbols = {
            g.GetSymbol() + str(g.GetIdx()): [
                f.GetSymbol() + str(f.GetIdx()) + '_' +
                str(mol.GetBondBetweenAtoms(g.GetIdx(), f.GetIdx()).GetBondType())
                for f in g.GetNeighbors()
            ]
            for g in mol.GetAtoms()
        }
        return symbols


# 使用示例
if __name__ == "__main__":
    # 创建包含虚拟原子的分子
    smiles = '[8*]n1ccc(=[9*])nc1=[10*]'
    mol = Chem.MolFromSmiles(smiles)

    if mol:
        # 注意：在实际使用中，_symmetry属性应该在调用前已经设置
        # 这里为了演示，手动设置一些示例值
        for idx, atom in enumerate(mol.GetAtoms()):
            if atom.GetSymbol() != '*':
                atom.SetIntProp("_symmetry", idx % 3)  # 示例对称性值

        processor = VirtualAtomConnectionProcessor()
        result = processor.process_mol(mol)

        if result:
            print("处理成功！")
            print(f"清理后的SMILES: {result['cleaned_smiles']}")
            print(f"虚拟原子: {result['virtual_atoms_found']}")
            print(f"连接信息数量: {len(result['connections'])}")

            # 格式化输出
            output_lines, tokens = processor.format_final_output(result)
            if output_lines:
                print("\n格式化输出:")
                for line in output_lines:
                    print(line)

            # 调试信息
            print("\n原子映射:")
            for orig_idx, clean_idx in result['atom_mapping'].items():
                print(f"  原始索引 {orig_idx} -> 清理后索引 {clean_idx}")

            print("\n详细原子信息:")
            atom_info = processor.get_detailed_atom_info(mol)
            for info in atom_info:
                print(f"  {info}")