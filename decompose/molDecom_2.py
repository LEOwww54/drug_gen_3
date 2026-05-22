import re
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from collections import defaultdict
from decompose_1.sym_calc import *
from decompose_1.translator import smiles2token
from sympy import false


class VirtualAtomConnectionProcessor:
    def __init__(self):
        self.original_smiles = ""
        self.cleaned_smiles = ""
        self.atom_connections = {}  # 存储原子连接信息
        self.virtual_atoms_info = {}  # 存储虚拟原子信息
        self.mol_with_virtual = None
        self.mol_cleaned = None
        self.cleaned_smiles_1 = ''

    def extract_virtual_atoms(self, smiles):
        """提取虚拟原子信息并生成清理后的SMILES"""
        if smiles == '[8*]n1ccc(=[9*])nc1=[10*]':
            pass
        self.original_smiles = smiles

        all_matches = []
        cleaned_smiles = smiles

        # 模式4: (键[数字*]) (如 (-[1*]), (=[2*]))
        pattern4_matches = re.findall(r'\(([\-=#])\[(\d+)\*\]\)', smiles)
        for bond, number in pattern4_matches:
            all_matches.append({'number': number, 'bond': bond, 'has_parentheses': True})
        cleaned_smiles = re.sub(r'\(([\-=#])\[(\d+)\*\]\)', '', cleaned_smiles)

        # 模式5: 键([数字*]) (如 -([1*]), =([2*]))
        pattern5_matches = re.findall(r'([\-=#]*)\(\[(\d+)\*\]\)', smiles)
        for bond, number in pattern5_matches:
            all_matches.append({'number': number, 'bond': bond, 'has_parentheses': True})
        cleaned_smiles = re.sub(r'([\-=#]*)\(\[(\d+)\*\]\)', '', cleaned_smiles)

        # 模式2: ([数字*])
        pattern2_matches = re.findall(r'\(\[(\d+)\*\]\)', smiles)
        for match in pattern2_matches:
            all_matches.append({'number': match, 'bond': '', 'has_parentheses': True})
        cleaned_smiles = re.sub(r'\(\[(\d+)\*\]\)', '', cleaned_smiles)

        # 模式3: 键[数字*] (如 -[1*], =[2*])
        pattern3_matches = re.findall(r'([\-=#])\[(\d+)\*\]', smiles)
        for bond, number in pattern3_matches:
            all_matches.append({'number': number, 'bond': bond, 'has_parentheses': False})
        cleaned_smiles = re.sub(r'([\-=#])\[(\d+)\*\]', '', cleaned_smiles)

        # 模式6: [数字*]键 (如 [1*]-, [2*]=)
        pattern6_matches = re.findall(r'\[(\d+)\*\](\-=#*)', smiles)
        for number, bond in pattern6_matches:
            all_matches.append({'number': number, 'bond': bond, 'has_parentheses': False})
        cleaned_smiles = re.sub(r'\[(\d+)\*\](\-=#*)', '', cleaned_smiles)

        # 模式1: [数字*]
        pattern1_matches = re.findall(r'\[(\d+)\*\]', smiles)
        for match in pattern1_matches:
            all_matches.append({'number': match, 'bond': '', 'has_parentheses': False})
        cleaned_smiles = re.sub(r'\[(\d+)\*\]', '', cleaned_smiles)

        # 清理多余的括号和键符号
        cleaned_smiles = self.clean_smiles(cleaned_smiles)

        self.cleaned_smiles = cleaned_smiles

        return all_matches

    def clean_smiles(self, smiles):
        """清理SMILES字符串中多余的空括号和键符号"""
        # 移除空的括号对 ()
        while '()' in smiles:
            smiles = smiles.replace('()', '')

        # 移除开头的键符号
        smiles = re.sub(r'^[\-=#]+', '', smiles)

        # 移除结尾的键符号
        smiles = re.sub(r'[\-=#]+$', '', smiles)

        # 移除孤立的键符号（被括号包围的）
        smiles = re.sub(r'\([\-=#]+\)', '', smiles)

        # 处理连续的键符号
        smiles = re.sub(r'[\-=#]{2,}', '-', smiles)  # 多个键符号替换为单键

        # 清理多余的空括号
        while '()' in smiles:
            smiles = smiles.replace('()', '')

        return smiles

    def parse_molecule_with_connections(self, smiles):
        """解析分子并建立连接关系"""
        try:
            # 解析包含虚拟原子的分子
            self.mol_with_virtual = Chem.MolFromSmiles(smiles)
            if self.mol_with_virtual is None:
                self.mol_with_virtual = Chem.MolFromSmarts(smiles)
                if self.mol_with_virtual is None:
                    raise ValueError("无法解析包含虚拟原子的SMILES")

            Chem.Kekulize(self.mol_with_virtual, True)

            # 解析清理后的分子
            self.mol_cleaned = Chem.MolFromSmiles(self.cleaned_smiles)
            if self.mol_cleaned is None:
                self.mol_cleaned = Chem.MolFromSmarts(self.cleaned_smiles)
                if self.mol_cleaned is None:
                    raise ValueError("无法解析清理后的SMILES")

            Chem.Kekulize(self.mol_cleaned, True)

            return self.analyze_connections_advanced(smiles)

        except Exception as e:
            print(f"解析分子时出错: {e}")
            return None

    def mol_dump(self, mol):
        if mol is None:
            return None
        symbols = {g.GetSymbol() + str(g.GetIdx()) : [f.GetSymbol() + str(f.GetIdx()) + '_' + str(mol.GetBondBetweenAtoms(g.GetIdx(),f.GetIdx()).GetBondType()) for f in g.GetNeighbors()] for g in mol.GetAtoms()}

        return symbols

    def analyze_connections_advanced(self, original_smiles):
        """使用更高级的方法分析连接信息"""
        try:
            # 解析原始分子
            mol_original = Chem.MolFromSmiles(original_smiles)
            if mol_original is None:
                mol_original = Chem.MolFromSmarts(original_smiles)
                if mol_original is None:
                    return None, None

            Chem.Kekulize(mol_original, True)

            _, symmetry = get_symmetry_equivalent_atoms(mol_original)

            # 创建不包含虚拟原子的分子副本
            mol_cleaned = Chem.RWMol(mol_original)

            # 标记要删除的虚拟原子
            atoms_to_remove = []
            virtual_connections = []

            for atom in mol_cleaned.GetAtoms():
                if not atom.GetSymbol() == '*':
                    atom.SetIntProp("_symmetry", symmetry[atom.GetIdx()])

                if atom.GetSymbol() == '*' and atom.GetIsotope() > 0:
                    virtual_number = atom.GetIsotope()
                    neighbors = atom.GetNeighbors()

                    if len(neighbors) == 1:
                        neighbor_idx = neighbors[0].GetIdx()
                        bond = mol_cleaned.GetBondBetweenAtoms(atom.GetIdx(), neighbor_idx)

                        bond_symbol = self.get_bond_symbol(bond)
                        virtual_connections.append({
                            'virtual_number': virtual_number,
                            'connected_atom_idx': neighbor_idx,
                            'bond_type': bond.GetBondType(),
                            'bond_symbol': bond_symbol,
                            'original_atom_idx': atom.GetIdx()
                        })

                    atoms_to_remove.append(atom.GetIdx())

            # 按索引降序删除虚拟原子（避免索引变化）
            for atom_idx in sorted(atoms_to_remove, reverse=True):
                mol_cleaned.RemoveAtom(atom_idx)
            # 获取清理后的分子
            self.mol_cleaned = mol_cleaned.GetMol()

            IH = {}

            # 为清理后的分子设置原子映射编号
            for atom in self.mol_cleaned.GetAtoms():
                if atom.GetAtomicNum() != 1:
                    atom.SetAtomMapNum(atom.GetIdx() + 1)  # 从1开始编号
                    IH[atom.GetAtomMapNum()] = atom.GetFormalCharge()


            # 生成带原子编号的SMILES
            self.cleaned_smiles = Chem.MolToSmiles(self.mol_cleaned, allHsExplicit=False)

            # 处理连接信息 - 使用更精确的映射
            connections = self.map_connections_precisely(mol_original, virtual_connections)

            tokens = self._smiles_number_process_mol(self.mol_cleaned, connections, IH)

            return connections, self.cleaned_smiles, tokens

        except Exception as e:
            print(f"高级分析出错: {e}")
            return None, None, None

    def _smiles_number_process_mol(self, mol : Chem.Mol, connections, IH):
        ssss = Chem.MolToSmiles(mol)
        tokens = []
        tokens.append('{')
        text = {}
        for atom in mol.GetAtoms():
            token = ['^atom^']

            if atom.GetAtomicNum() in [6, 8]:
                symbol = f'[{atom.GetSymbol()}]'
            else:
                total_h = atom.GetTotalNumHs()
                if total_h > 1:
                    symbol = f'[{atom.GetSymbol()}H{total_h}]'
                elif total_h == 1:
                    symbol = f'[{atom.GetSymbol()}H]'
                else:
                    symbol = f'[{atom.GetSymbol()}]'


            formal_charge = f"<fc{str(atom.GetFormalCharge())}>"
            atom_index = atom.GetAtomMapNum()
            atom_sym = f"<sym{atom.GetProp('_symmetry')}>"

            atom_radical = f'<rad{atom.GetNumRadicalElectrons()}>'

            conn_info = []
            for connection in connections:
                if connection['atom_number'] == atom_index:
                    tmp = f"<m{connection['bond_symbol']}"
                    conn_info.append(tmp)
                    tmp = f"{connection['connection_number']}>"
                    conn_info.append(tmp)

            token.append(symbol)
            token.append(atom_radical)
            token.append(formal_charge)
            token.extend(conn_info)
            token.append(atom_sym)
            if(atom.IsInRing()):
                token.append('<r>')

            text[atom.GetIdx()] = token

        tokens.extend(smiles2token(mol, text))

        tokens.append('}')
        return tokens

    def _smiles_number_process(self, smiles, connections, IH):
        tokens = []
        token = ''
        in_atom = False
        symbol = False
        Aromatic = False
        index = False
        extra_info = []
        ring = False
        multiring = False
        i = 0
        atom_count = 0
        tokens.append('{')
        while i < len(smiles):
            s = smiles[i]
            token = token + s

            if s == '[':
                tokens.append('^atom^')
                token = s
                if in_atom:
                    raise 'error in bracket'
                in_atom = True
                symbol  = True
                i = i + 1
                continue
            else:
                if s== ']' and in_atom:
                    in_atom = False
                    tokens.append( f"[{token[1].upper()}" + token[2:])
                    token = ''
                    for e in extra_info:
                        tokens.append(e)

                    extra_info = []

                    i = i + 1
                    continue

                if symbol:
                    symbol = False
                    if s.islower():
                        Aromatic = True
                    else:
                        Aromatic = False
                    i = i + 1
                    continue
                if s == ':' and in_atom:
                    index = True
                    symbol = False
                    i = i + 1
                    continue
                if index:
                    ts = ""
                    while not s == ']':
                        ts = ts + s
                        i = i + 1
                        if i >= len(smiles):
                            break
                        s = smiles[i]

                    atom_count = int(ts)

                    fc = IH[atom_count]

                    for connection in connections:
                        if connection['atom_number'] == atom_count:
                            tmp = f"<m{connection['bond_symbol']}"
                            extra_info.append(tmp)
                            tmp = f"{connection['connection_number']}>"
                            extra_info.append(tmp)

                    if Aromatic:
                        extra_info.append('<A>')

                    if not fc == 0:
                        extra_info.append(f'<c{fc}>')

                    index = False
                    # i = i + 1
                    continue
                if s == '%' and not in_atom:
                    ring = True## ring
                    multiring = True
                    i = i + 1
                    continue
                if s.isdigit() and not in_atom: ## ring
                    ring = True

                    if multiring:
                        ts = ''
                        while s.isdigit():
                            ts = ts + s
                            i = i + 1
                            if i >= len(smiles):
                                break
                            s = smiles[i]

                        tokens.append(f'<r%{int(ts)}>')
                        multiring = False
                    else:
                        tokens.append(f'<r{int(s)}>')
                        i = i + 1

                    ring = False
                    # i = i + 1
                    continue
                if (s == '(' or s == ')' or s == '=' or s == '#' or s == '-' or s=='.') and not in_atom:
                    tokens.append(s)

                i = i + 1

        tokens.append('}')
        return tokens


    def map_connections_precisely(self, original_mol, virtual_connections):
        """精确映射连接信息，使用处理后分子的原子编号"""
        connections = []

        # 使用子结构匹配找到原子映射
        matches = original_mol.GetSubstructMatches(self.mol_cleaned)

        if matches:
            # 取第一个匹配
            match = matches[0]
            # 创建从原始分子到清理后分子的映射
            original_to_cleaned = {orig_idx: clean_idx for clean_idx, orig_idx in enumerate(match)}

            for vc in virtual_connections:
                original_atom_idx = vc['connected_atom_idx']

                if original_atom_idx in original_to_cleaned:
                    cleaned_atom_idx = original_to_cleaned[original_atom_idx]

                    # 获取原子元素符号
                    atom = self.mol_cleaned.GetAtomWithIdx(cleaned_atom_idx)
                    element_symbol = atom.GetSymbol()

                    # 获取原子的映射编号（应该与cleaned_atom_idx + 1一致）
                    atom_map_num = atom.GetAtomMapNum()
                    if atom_map_num == 0:
                        # 如果没有设置映射编号，使用索引+1
                        atom_map_num = cleaned_atom_idx + 1

                    # 格式化连接信息 - 使用清理后分子的原子编号
                    connection_info = {
                        'element_symbol': element_symbol,
                        'atom_number': atom_map_num,  # 使用映射编号而不是索引
                        'bond_symbol': vc['bond_symbol'],
                        'connection_number': vc['virtual_number'],
                        'formatted_string': f"{element_symbol} {atom_map_num}\t{vc['bond_symbol']}{vc['virtual_number']}",
                        'formatted_string_1': f'i{atom_map_num}{vc["bond_symbol"]}',
                        'formatted_string_2': f'i{vc["virtual_number"]}',
                        'original_atom_idx': original_atom_idx,
                        'cleaned_atom_idx': cleaned_atom_idx
                    }
                    connections.append(connection_info)
                else:
                    print(f"警告: 无法映射原子 {original_atom_idx}")

        return connections

    def get_bond_symbol(self, bond):
        """获取化学键的符号表示"""
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
        """格式化最终输出"""
        if not result:
            return "none"
        try:
            tokens = []
            output_lines = []

            # 第一行: 处理后的SMILES（带原子编号）
            output_lines.append(result['cleaned_smiles'])

            # 第二行开始: 连接信息
            # 对连接信息按原子编号排序
            sorted_connections = sorted(result['connections'], key=lambda x: int(x['atom_number']))

            for conn in sorted_connections:
                output_lines.append(conn['formatted_string_1'])
                output_lines.append(conn['formatted_string_2'])
        except Exception as e:
            return None

        return output_lines, result['tokens']

    def get_detailed_atom_info(self, mol):
        """获取分子的详细原子信息，用于调试"""
        atom_info = []
        for atom in mol.GetAtoms():
            map_num = atom.GetAtomMapNum()
            atom_info.append(f"原子{atom.GetIdx()}: {atom.GetSymbol()} (映射编号: {map_num})")
        return atom_info

    def process_smiles(self, smiles):
        """处理SMILES字符串的主函数"""
        # 提取虚拟原子
        virtual_atoms = self.extract_virtual_atoms(smiles)

        # 解析分子并分析连接
        result = self.parse_molecule_with_connections(smiles)

        if result:
            connections, cleaned_smiles, tokens = result

            result_dict = {
                'original_smiles': smiles,
                'cleaned_smiles': cleaned_smiles,
                'connections': connections,
                'virtual_atoms_found': virtual_atoms,
                'tokens': tokens
            }

            return result_dict
        else:
            return None
