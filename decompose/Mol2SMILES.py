from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import re


class VirtualAtomSMILESGenerator:
    def __init__(self):
        self.bond_type_map = {
            'SINGLE': Chem.BondType.SINGLE,
            'DOUBLE': Chem.BondType.DOUBLE,
            'TRIPLE': Chem.BondType.TRIPLE,
            'AROMATIC': Chem.BondType.AROMATIC,
            '-': Chem.BondType.SINGLE,
            '=': Chem.BondType.DOUBLE,
            '#': Chem.BondType.TRIPLE,
            ':': Chem.BondType.AROMATIC
        }

    def add_virtual_atoms_to_mol(self, mol, connections):
        """
        向分子添加虚拟原子

        Args:
            mol: RDKit分子对象
            connections: 连接信息列表，每个连接是字典格式
                [{
                    'atom_num': 原子序号(从1开始),
                    'bond_type': 化学键类型,
                    'connection_id': 连接编号
                }]

        Returns:
            Chem.Mol: 添加了虚拟原子的分子
        """
        # 创建可编辑的分子副本
        rw_mol = Chem.RWMol(mol)

        added_virtual_atoms = []

        for conn in connections:
            # 获取连接原子（从1-based转换为0-based）
            atom_idx = conn['atom_num']
            if atom_idx >= rw_mol.GetNumAtoms():
                print(f"警告: 原子序号 {conn['atom_num']} 超出范围")
                continue

            # 创建虚拟原子
            virtual_atom = Chem.Atom(0)  # 原子序号0表示虚拟原子
            virtual_atom.SetIsotope(conn['connection_id'])  # 使用同位素存储连接编号
            virtual_atom.SetProp("_MolFileComment", f"VirtualAtom_{conn['connection_id']}")

            # 添加虚拟原子到分子
            virtual_idx = rw_mol.AddAtom(virtual_atom)
            added_virtual_atoms.append(virtual_idx)

            # 获取化学键类型
            bond_type = self.get_bond_type(conn['bond_type'])

            # 添加化学键
            rw_mol.AddBond(atom_idx, virtual_idx, bond_type)

        return rw_mol.GetMol()

    def get_bond_type(self, bond_spec):
        """将化学键描述转换为RDKit的BondType"""
        if isinstance(bond_spec, Chem.BondType):
            return bond_spec

        bond_spec_str = str(bond_spec).upper().strip()

        if bond_spec_str in self.bond_type_map:
            return self.bond_type_map[bond_spec_str]
        elif bond_spec_str in ['SINGLE', '-', '1']:
            return Chem.BondType.SINGLE
        elif bond_spec_str in ['DOUBLE', '=', '2']:
            return Chem.BondType.DOUBLE
        elif bond_spec_str in ['TRIPLE', '#', '3']:
            return Chem.BondType.TRIPLE
        elif bond_spec_str in ['AROMATIC', ':', '4']:
            return Chem.BondType.AROMATIC
        else:
            print(f"警告: 未知的化学键类型 '{bond_spec}'，使用单键")
            return Chem.BondType.SINGLE

    def generate_smiles_with_virtual_atoms(self, mol, connections, canonical=True):
        """
        生成带有编号虚拟原子的SMILES

        Args:
            mol: RDKit分子对象
            connections: 连接信息列表
            canonical: 是否生成规范SMILES

        Returns:
            str: 带有虚拟原子的SMILES字符串
        """
        # 添加虚拟原子
        mol_with_virtual = self.add_virtual_atoms_to_mol(mol, connections)

        # 生成SMILES
        if canonical:
            smiles = Chem.MolToSmiles(mol_with_virtual, canonical=True)
        else:
            smiles = Chem.MolToSmiles(mol_with_virtual, canonical=False)

        return smiles

    def create_connection_info(self, atom_nums, bond_types, connection_ids):
        """
        创建连接信息列表的便捷方法

        Args:
            atom_nums: 原子序号列表（从1开始）
            bond_types: 化学键类型列表
            connection_ids: 连接编号列表

        Returns:
            list: 连接信息字典列表
        """
        connections = []
        for atom_num, bond_type, conn_num in zip(atom_nums, bond_types, connection_ids):
            connections.append({
                'atom_num': atom_num,
                'bond_type': bond_type,
                'connection_id': conn_num
            })
        return connections


def get_bond_symbol(self, bond_type):
    """将BondType转换为符号表示"""
    if bond_type == Chem.BondType.SINGLE:
        return '-'
    elif bond_type == Chem.BondType.DOUBLE:
        return '='
    elif bond_type == Chem.BondType.TRIPLE:
        return '#'
    elif bond_type == Chem.BondType.AROMATIC:
        return ':'
    else:
        return '~'
# 添加到类中
VirtualAtomSMILESGenerator.get_bond_symbol = get_bond_symbol
