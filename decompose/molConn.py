from rdkit import Chem
from rdkit.Chem import AllChem
from collections import defaultdict
import re


def auto_connect_fragments_by_dummy_atoms(smiles_list, default_bond_type='-'):
    """
    根据相同编号的虚拟原子自动连接多个分子片段

    参数:
    smiles_list: 包含SMILES字符串的列表，其中虚拟原子用[*:N]表示
    default_bond_type: 默认键类型，当无法确定时使用

    返回:
    连接后的分子对象和SMILES字符串
    """

    # 1. 解析所有片段并提取虚拟原子信息
    mols = []
    all_dummy_info = []  # 存储每个片段的虚拟原子信息
    dummy_number_to_fragments = defaultdict(list)  # 虚拟原子编号到片段的映射

    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"无法解析SMILES: {smiles}")
        mol = Chem.AddHs(mol)
        mols.append(mol)

        # 提取虚拟原子信息
        fragment_dummy_info = extract_dummy_atoms_with_bonds(mol)
        all_dummy_info.append(fragment_dummy_info)

        # 建立虚拟原子编号到片段的映射
        for dummy_num, dummy_data in fragment_dummy_info.items():
            dummy_number_to_fragments[dummy_num].append((i, dummy_data))

    # 2. 自动确定连接关系
    connections = auto_generate_connections(dummy_number_to_fragments, default_bond_type)

    if not connections:
        print("警告: 未找到任何连接关系")


    # 3. 创建组合分子
    combined_mol = Chem.RWMol()
    atom_mapping = []  # 记录每个片段原子在新分子中的索引映射

    # 添加所有原子（除了虚拟原子）
    for i, mol in enumerate(mols):
        fragment_atom_map = {}

        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() != 0:  # 不是虚拟原子
                # Preserve isotope, explicit-H policy, charge and radical state.
                new_atom = Chem.Atom(atom)
                new_idx = combined_mol.AddAtom(new_atom)
                fragment_atom_map[atom.GetIdx()] = new_idx

        atom_mapping.append(fragment_atom_map)

    # 4. 添加片段内部的键（不涉及虚拟原子）
    for i, mol in enumerate(mols):
        for bond in mol.GetBonds():
            begin_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()
            begin_atom = mol.GetAtomWithIdx(begin_idx)
            end_atom = mol.GetAtomWithIdx(end_idx)

            # 只有当两个原子都不是虚拟原子时才添加键
            if begin_atom.GetAtomicNum() != 0 and end_atom.GetAtomicNum() != 0:
                if begin_idx in atom_mapping[i] and end_idx in atom_mapping[i]:
                    combined_mol.AddBond(
                        atom_mapping[i][begin_idx],
                        atom_mapping[i][end_idx],
                        bond.GetBondType()
                    )

    # 5. 处理片段间的连接
    used_pairs = set()  # 避免重复连接

    for conn in connections:
        frag1_idx, dummy_num1, frag2_idx, dummy_num2, bond_type = conn

        # 检查是否已经连接过这个片段对
        pair_key = tuple(sorted([frag1_idx, frag2_idx]))
        if pair_key in used_pairs:
            continue
        used_pairs.add(pair_key)

        # 获取虚拟原子在原片段中的邻居（真实原子）
        real_atom1 = all_dummy_info[frag1_idx][dummy_num1]['neighbor_idx']
        real_atom2 = all_dummy_info[frag2_idx][dummy_num2]['neighbor_idx']

        if real_atom1 is None or real_atom2 is None:
            print(f"警告: 无法找到虚拟原子 {dummy_num1} 或 {dummy_num2} 的连接原子")
            continue

        # 获取在新分子中的原子索引
        new_idx1 = atom_mapping[frag1_idx][real_atom1]
        new_idx2 = atom_mapping[frag2_idx][real_atom2]

        # 添加连接键
        bond_type_obj = get_bond_type(bond_type)
        combined_mol.AddBond(new_idx1, new_idx2, bond_type_obj)
        # print(f"连接: 片段{frag1_idx}的原子{new_idx1} --{bond_type}-- 片段{frag2_idx}的原子{new_idx2}")

    # 6. 清理并返回结果
    final_mol = combined_mol.GetMol()
    final_mol = Chem.RemoveHs(final_mol)
    try:
        Chem.Kekulize(final_mol,True)
        ## Chem.SanitizeMol(final_mol)
        final_smiles = Chem.MolToSmiles(final_mol)
    except Exception as e:
        print(f"Sanitization 警告: {e}")
        final_smiles = Chem.MolToSmiles(final_mol)

    return final_mol, final_smiles, connections


def extract_dummy_atoms_with_bonds(mol):
    """
    提取分子中的虚拟原子信息及其连接信息
    返回: {虚拟原子编号: {'idx': 原子索引, 'neighbor_idx': 邻居原子索引, 'bond_type': 键类型}}
    """
    dummy_atoms = {}
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:  # 虚拟原子
            isotope = atom.GetIsotope()
            if isotope > 0:
                # 获取虚拟原子的连接信息
                neighbor_idx = None
                bond_type = '-'

                for bond in atom.GetBonds():
                    neighbor = bond.GetOtherAtom(atom)
                    if neighbor.GetAtomicNum() != 0:  # 真实原子
                        neighbor_idx = neighbor.GetIdx()
                        # 获取键类型
                        if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
                            bond_type = '='
                        elif bond.GetBondType() == Chem.rdchem.BondType.TRIPLE:
                            bond_type = '#'
                        elif bond.GetBondType() == Chem.rdchem.BondType.AROMATIC:
                            bond_type = ':'
                        break

                dummy_atoms[isotope] = {
                    'idx': atom.GetIdx(),
                    'neighbor_idx': neighbor_idx,
                    'bond_type': bond_type
                }
    return dummy_atoms


def auto_generate_connections(dummy_number_to_fragments, default_bond_type):
    """
    根据虚拟原子编号自动生成连接关系
    """
    connections = []
    processed_pairs = set()

    for dummy_num, fragments in dummy_number_to_fragments.items():
        # 每个编号的虚拟原子应该连接两个片段
        if len(fragments) == 2:
            frag1_idx, dummy_info1 = fragments[0]
            frag2_idx, dummy_info2 = fragments[1]

            # 避免重复连接
            pair_key = tuple(sorted([frag1_idx, frag2_idx]))
            if pair_key in processed_pairs:
                continue

            # 确定键类型（优先使用第一个片段的键类型）
            bond_type = dummy_info1.get('bond_type', default_bond_type)

            connections.append((frag1_idx, dummy_num, frag2_idx, dummy_num, bond_type))
            processed_pairs.add(pair_key)

        elif len(fragments) > 2:
            print(f"警告: 虚拟原子编号 {dummy_num} 出现在 {len(fragments)} 个片段中，只连接前两个")
            frag1_idx, dummy_info1 = fragments[0]
            frag2_idx, dummy_info2 = fragments[1]

            pair_key = tuple(sorted([frag1_idx, frag2_idx]))
            if pair_key not in processed_pairs:
                bond_type = dummy_info1.get('bond_type', default_bond_type)
                connections.append((frag1_idx, dummy_num, frag2_idx, dummy_num, bond_type))
                processed_pairs.add(pair_key)

    return connections


def get_bond_type(bond_char):
    """将字符转换为RDKit键类型"""
    bond_types = {
        '-': Chem.rdchem.BondType.SINGLE,
        '=': Chem.rdchem.BondType.DOUBLE,
        '#': Chem.rdchem.BondType.TRIPLE,
        ':': Chem.rdchem.BondType.AROMATIC,
        '~': Chem.rdchem.BondType.UNSPECIFIED
    }
    return bond_types.get(bond_char, Chem.rdchem.BondType.SINGLE)


def print_dummy_atom_info(smiles_list):
    """打印所有片段的虚拟原子信息，用于调试"""
    print("虚拟原子信息分析:")
    print("-" * 50)

    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        dummy_info = extract_dummy_atoms_with_bonds(mol)

        print(f"片段 {i}: {smiles}")
        if dummy_info:
            for dummy_num, info in dummy_info.items():
                neighbor_atom = mol.GetAtomWithIdx(info['neighbor_idx']) if info['neighbor_idx'] is not None else None
                neighbor_symbol = neighbor_atom.GetSymbol() if neighbor_atom else "None"
                print(f"  虚拟原子 [*:{dummy_num}] 连接到 {neighbor_symbol} (键类型: {info['bond_type']})")
        else:
            print("   无虚拟原子")
        print()

ORGANIC_ELEMENTS_1 = [
        'B', 'C', 'N', 'O', 'F', 'P', 'S',
'I',
    ]

ORGANIC_ELEMENTS_2 = [
        'Cl',
        'Br',
    ]

def _smiles_atom(token, *, legacy=False, charge=None, radical=0, aromatic=False):
    """Preserve SMILES bracket atoms; adapt only explicitly marked legacy SISR."""
    if not (token.startswith('[') and token.endswith(']')):
        raise ValueError(f'Invalid atom token: {token!r}')
    atom = Chem.AtomFromSmiles(token)
    if atom is None:
        raise ValueError(f'Invalid SMILES atom token: {token!r}')
    body = token[1:-1]
    if aromatic:
        body = re.sub(r'^(\d*)([A-Z][a-z]?)',
                      lambda m: m[1] + m[2].lower(), body)
    if charge is not None:
        if atom.GetFormalCharge() and atom.GetFormalCharge() != charge:
            raise ValueError(f'Conflicting charges for {token!r}')
        # Charge precedes an optional atom map; SMILES uses +2/-2, not 2+/-2-.
        body, separator, mapping = body.partition(':')
        body = re.sub(r'(?:\+{2,}|-{2,}|[+-]\d*)$', '', body)
        if charge:
            body += '+' if charge > 0 else '-'
            if abs(charge) > 1:
                body += str(abs(charge))
        if separator:
            body += ':' + mapping
    if legacy and not radical and not (charge or atom.GetFormalCharge()):
        # Old [C]/[O] etc. represented atoms with inferred hydrogen counts.
        # Keep explicit H, isotope, chirality and maps inside their brackets.
        if body in {'B', 'C', 'N', 'O', 'P', 'S', 'F', 'Cl', 'Br', 'I',
                    'b', 'c', 'n', 'o', 'p', 's'}:
            return body
    result = '[' + body + ']'
    if Chem.AtomFromSmiles(result) is None:
        raise ValueError(f'Invalid SMILES atom token: {result!r}')
    return result


def _translate_fragment(tokens):
    legacy = any(t.startswith(('<fc', '<rad')) or t == '^atom^' for t in tokens)
    output, links, suffix = [], [], []
    atom_token = None
    charge, radical, aromatic = None, 0, False
    pending_bond = None

    def flush_atom():
        if pending_bond is not None:
            raise ValueError('Attachment bond is missing its connection number')
        if atom_token is not None:
            output.append(_smiles_atom(atom_token, legacy=legacy, charge=charge,
                                       radical=radical, aromatic=aromatic))
            output.extend(links)
            output.extend(suffix)

    for token in tokens:
        if token.startswith('['):
            flush_atom()
            atom_token = token
            links, suffix = [], []
            charge, radical, aromatic = None, 0, False
            continue
        if token == '^atom^' or token == '^':
            continue
        if atom_token is None:
            raise ValueError(f'Expected an atom before {token!r}')
        if pending_bond is not None:
            match = re.fullmatch(r'([-=#:~]?)([1-9]\d*)>', token)
            if match is None:
                raise ValueError(f'Invalid attachment number: {token!r}')
            token_bond, connection_number = match.groups()
            if pending_bond and token_bond and pending_bond != token_bond:
                raise ValueError(f'Conflicting attachment bonds: {pending_bond!r} and {token_bond!r}')
            bond = token_bond or pending_bond or '-'
            links.append(f'({bond}[{connection_number}*])')
            pending_bond = None
        elif token == '<m':
            # The marker carries no bond; an optional bond prefixes the number token.
            pending_bond = ''
        elif re.fullmatch(r'<m[-=#:~]', token):
            # Continue accepting the former layout (<m= 1>) when decoding old data.
            pending_bond = token[2:]
        elif re.fullmatch(r'<fc[+-]?\d+>', token):
            charge = int(token[3:-1])
        elif re.fullmatch(r'<rad\d+>', token):
            radical = int(token[4:-1])
        elif token == '<A>':
            aromatic = True
        elif token == '<r>' or re.fullmatch(r'<sym-?\d+>', token):
            continue
        elif re.fullmatch(r'<r%?\d+>', token):
            number = int(token[2:-1].lstrip('%'))
            suffix.append(str(number) if number < 10 else
                          (f'%{number}' if number < 100 else f'%({number})'))
        elif token in {'-', '=', '#', ':', '~', '/', '\\', '(', ')', '.'}:
            suffix.append(token)
        else:
            raise ValueError(f'Unsupported SISR token: {token!r}')
    flush_atom()
    smiles = ''.join(output)
    if not smiles or Chem.MolFromSmiles(smiles) is None:
        raise ValueError(f'Fragment is not valid SMILES: {smiles!r}')
    return smiles


def mol_translate(text):
    """Decode SISR fragments, preserving explicit H and embedded SMILES charges.

    Legacy fc/rad metadata is accepted for existing datasets. New bracket atom
    tokens are used literally, including isotope, chirality, H count and charge.
    Kekulized fragments omit single-bond tokens; ``<m N>`` denotes a single-bond
    attachment and ``<m =N>`` denotes a double-bond attachment. The former
    ``<m= N>`` layout remains supported when decoding older datasets.
    """
    if '<sep>' in text:
        text = text.split('<sep>', 1)[1]
    results = []
    fragment = None
    for token in text.split():
        if token == '{':
            if fragment is not None:
                raise ValueError('Nested fragment boundary')
            fragment = []
        elif token == '}':
            if fragment is None:
                raise ValueError('Unmatched fragment boundary')
            results.append(_translate_fragment(fragment))
            fragment = None
        elif fragment is not None:
            fragment.append(token)
    if fragment is not None:
        raise ValueError('Unclosed fragment boundary')
    return results


def gen2mol(texts):
    result = []

    for line in texts:
        if isinstance(line, tuple):
            line = line[0]
        try:
            processed_line = mol_translate(line)
            result.append(auto_connect_fragments_by_dummy_atoms(processed_line))
        except:
            continue


    return result

if __name__ == '__main__':
    s = '<start> { [C] <rad0> <fc0> - [C] <rad0> <fc0> <r> ( - [N] <rad0> <fc0> <r> = <r2> ) ( = [C] <rad0> <fc0> <r> ( - [C] <rad0> <fc0> <r> = [C] <rad0> <fc0> <r> - [N] <rad0> <fc0> <r> = <r2> ) ( - [C] <rad0> <fc0> ( = [O] <rad0> <fc0> ) ( - [NH] <rad0> <fc0> - [C] <rad0> <fc0> ( - [C] <rad0> <fc0> ) ( - [C] <rad0> <fc0> <r> ( = [C] <rad0> <fc0> <r> - <r1> ) ( - [S] <rad0> <fc0> <r> - [C] <rad0> <fc0> <r> ( = [C] <rad0> <fc0> <r> - <r1> ) ( - [Cl] <rad0> <fc0> ) ) ) ) ) ) } </s>'
    r = gen2mol([s])
    d = Chem.MolFromSmiles('CC(C)(C)c1ccc2occ(CC(=O)Nc3ccccc3F)c2c1')
    Chem.Kekulize(d, True)
    r = Chem.MolToSmiles(d)
    pass
