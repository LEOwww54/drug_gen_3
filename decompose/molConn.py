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
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            mol = Chem.MolFromSmarts(smiles, sanitize=False)
            if mol is None:
                raise ValueError(f"无法解析SMILES: {smiles}")
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
                new_atom = Chem.Atom(atom.GetAtomicNum())
                new_atom.SetFormalCharge(atom.GetFormalCharge())
                new_atom.SetIsAromatic(atom.GetIsAromatic())
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
    try:
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

def mol_translate(text : str):
    text = text.replace('\t', ' ')
    if '<sep>' in text:
        text = text.split('<sep>')
        text = text[1]
    text = text.strip()

    text = text.split(' ')
    results = []

    frags = []
    frag = []

    for token in text:
        if token == '{':
            frag = []
            continue
        if token == '}':
            frag.append('^atom^')
            frags.append(frag)
            continue

        frag.append(token)

    atoms = ''
    for frag in frags:
        first_check = True
        atoms = ''
        atom = ''
        symbol = ''
        aromatic = False
        rings = []

        link_bond = ''
        link_index = ''
        links = []
        b = False

        functional_token = ''

        for token in frag:
            if token == '^atom^' and not first_check:
                if aromatic:
                    if len(symbol) > 1:
                        symbol = symbol[0].lower() + symbol[1:]
                    else:
                        symbol = symbol.lower()
                aromatic = False
                if b:
                    symbol = f"[{symbol}]"
                b = False

                for link in links:
                    symbol += link
                links = []

                symbol += functional_token
                functional_token = ''

                for ring in rings:
                    symbol += ring
                rings = []

                atom = symbol
                atoms = atoms + atom
                continue
            else:
                first_check = False
            if token[0] == '[':
                ts = token[1:-1]

                if '+' in ts or '-' in ts or ts[0].isdigit() or '@' in ts:
                    b = True
                elif len(ts) == 2 and not ts in ORGANIC_ELEMENTS_2:
                    b = True
                elif len(ts) == 1 and not ts in ORGANIC_ELEMENTS_1:
                    b = True
                elif len(ts) > 2:
                    b = True

                symbol = ts

                continue

            if token[0] == '<':
                if token[1] == 'm':
                    link_bond = token[2]
                    continue
                if token[1] == 'r':
                    functional_token += token[2:-1]
                    continue
                if token[1] == 'A':
                    aromatic = True
                    continue
                if token[1] == 'c':
                    FormalCharge = token[2:-1]
                    FormalCharge = FormalCharge[1:] + FormalCharge[0]
                    continue

            if token[-1] == '>':
                link_index = int(token[:-1])
                links.append(f"({link_bond}[{link_index}*])")
                link_bond = ''
                link_index = ''
                continue
            if not token == '^atom^':
                functional_token += token
        results.append(atoms)

    return results

def gen2mol(texts):
    result = []

    for line in texts:
        try:
            processed_line = mol_translate(line)
            result.append(auto_connect_fragments_by_dummy_atoms(processed_line))
        except Exception as e:
            print(e)
            continue

    return result

if __name__ == '__main__':
    x = "CC(=O)O[C@H]1[C@H]2[C@@]([C@H]3[C@@]([C@]4(C[C@@H]5[C@]6(C[C@@H](C(=C([C@@H](O6)C(=O)[C@]5(C4=C(C3=O)C)OC(=O)C)OC(=O)c7ccccc7)O)C)OC(=O)C)O2)OC(=O)c8ccccc8)(C1(C)C)OC(=O)C"
    x = Chem.MolFromSmiles(x)
    s = ['[1*]C1OC(=[10*])C([2*])C([11*])OC1=[12*]', '[13*]C1=CC=C([14*])C=C1', '[3*]C1NC(=[16*])C([4*])NC(=[17*])C2NC(=[18*])C(NC(=[19*])C([14*])NC1=[15*])C1=C(C([7*])=C([6*])C([5*])=C1)C1=C([8*])C=C([9*])C=C12', '[17*]=O', '[18*]=O', '[10*]=O', '[19*]=O', '[11*]O[13*]', '[15*]=O', '[16*]=O', '[12*]=O', '[1*]CC', '[2*]C', '[5*]O', '[6*]O', '[7*]Cl', '[3*]C(C)C', '[8*]O', '[9*]O', '[4*]CC(N)=O']
    r = auto_connect_fragments_by_dummy_atoms(s)
    r = gen2mol(s)
    for i in r:
        print(i[1])

