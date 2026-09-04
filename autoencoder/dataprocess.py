import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from tqdm import tqdm
import json


class MoleculeGraph:
    """分子图数据结构 - 只包含原子类型和键类型"""

    def __init__(self, smiles, max_nodes=30):
        self.smiles = smiles
        self.max_nodes = max_nodes
        self.mol = Chem.MolFromSmiles(smiles)
        if self.mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        Chem.SanitizeMol(self.mol)

        # 原子类型和键类型
        self.atom_types = self._get_atom_types()
        self.bond_types, self.bond_pairs = self._get_bond_types()
        self.n_nodes = len(self.atom_types)

        # 原子类型词汇表
        self.atom_vocab = {
            'C': 1, 'N': 2, 'O': 3, 'S': 4, 'P': 5,
            'F': 6, 'Cl': 7, 'Br': 8, 'I': 9, 'B': 10,
            'Si': 11, 'Se': 12, 'Sn': 13, 'Pb': 14, 'As': 15,
        }

        # 键类型词汇表
        self.bond_vocab = {
            Chem.BondType.SINGLE: 1,
            Chem.BondType.DOUBLE: 2,
            Chem.BondType.TRIPLE: 3,
            Chem.BondType.AROMATIC: 4,
        }

    def _get_atom_types(self):
        """提取原子类型（只保留符号）"""
        atom_types = []
        for atom in self.mol.GetAtoms():
            symbol = atom.GetSymbol()
            # 处理特殊情况
            if symbol == 'C' and atom.GetIsAromatic():
                symbol = 'c'
            atom_types.append(symbol)
        return atom_types

    def _get_bond_types(self):
        """提取键类型"""
        bond_types = []
        bond_pairs = []

        for bond in self.mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            bond_type = bond.GetBondType()

            # 存储键对和类型
            bond_pairs.append((i, j))
            bond_pairs.append((j, i))  # 双向
            bond_types.append(bond_type)
            bond_types.append(bond_type)

        return bond_types, bond_pairs

    def to_dense(self, max_nodes):
        """转换为稠密矩阵"""
        # 原子类型：使用one-hot编码
        atom_feat = torch.zeros(max_nodes, len(self.atom_vocab) + 1)  # +1 for padding
        for i, symbol in enumerate(self.atom_types):
            if symbol in self.atom_vocab:
                atom_feat[i, self.atom_vocab[symbol]] = 1
            else:
                # 未知原子类型
                atom_feat[i, 0] = 1

        # 键类型：使用one-hot编码
        bond_feat = torch.zeros(max_nodes, max_nodes, len(self.bond_vocab) + 1)  # +1 for no bond
        for (i, j), bond_type in zip(self.bond_pairs, self.bond_types):
            if i < max_nodes and j < max_nodes:
                if bond_type in self.bond_vocab:
                    bond_feat[i, j, self.bond_vocab[bond_type]] = 1
                else:
                    bond_feat[i, j, 0] = 1  # 未知键类型

        mask = torch.zeros(max_nodes, dtype=torch.bool)
        mask[:self.n_nodes] = True

        return atom_feat, bond_feat, mask, self.n_nodes


class SubstructureDataset(Dataset):
    """子结构数据集"""

    def __init__(self, smiles_list, max_nodes=30):
        self.smiles_list = smiles_list
        self.max_nodes = max_nodes
        self.graphs = []

        print(f"Processing {len(smiles_list)} molecules...")
        for smiles in tqdm(smiles_list, desc="Processing"):
            try:
                graph = MoleculeGraph(smiles, max_nodes)
                if 1 <= graph.n_nodes <= max_nodes:
                    self.graphs.append(graph)
            except Exception as e:
                continue

        print(f"Successfully processed {len(self.graphs)} molecules")

        # 获取词汇表大小
        self.atom_vocab_size = len(self.graphs[0].atom_vocab) + 1 if self.graphs else 2
        self.bond_vocab_size = len(self.graphs[0].bond_vocab) + 1 if self.graphs else 2

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        graph = self.graphs[idx]
        atom_feat, bond_feat, mask, n_nodes = graph.to_dense(self.max_nodes)
        return {
            'atom_feat': atom_feat,  # one-hot编码的原子类型
            'bond_feat': bond_feat,  # one-hot编码的键类型
            'mask': mask,
            'n_nodes': n_nodes,
            'smiles': graph.smiles
        }


def collate_graphs(batch):
    """批处理函数"""
    max_nodes = max([item['atom_feat'].shape[0] for item in batch])

    atom_feats = []
    bond_feats = []
    masks = []
    n_nodes = []
    smiles_list = []

    for item in batch:
        # 原子特征填充
        atom_feat = item['atom_feat']
        padded_atom = torch.zeros(max_nodes, atom_feat.shape[-1])
        padded_atom[:len(atom_feat)] = atom_feat
        atom_feats.append(padded_atom)

        # 键特征填充
        bond_feat = item['bond_feat']
        padded_bond = torch.zeros(max_nodes, max_nodes, bond_feat.shape[-1])
        padded_bond[:len(bond_feat), :len(bond_feat)] = bond_feat
        bond_feats.append(padded_bond)

        # 掩码
        mask = item['mask']
        padded_mask = torch.zeros(max_nodes, dtype=torch.bool)
        padded_mask[:len(mask)] = mask
        masks.append(padded_mask)

        n_nodes.append(item['n_nodes'])
        smiles_list.append(item['smiles'])

    return {
        'atom_feat': torch.stack(atom_feats),
        'bond_feat': torch.stack(bond_feats),
        'mask': torch.stack(masks),
        'n_nodes': torch.tensor(n_nodes),
        'smiles': smiles_list
    }

def from_json(json_file, filter = 50):
    with open(json_file, 'r') as f:
        json_data = json.load(f)

    smiles_list = {}
    for dataset, category in json_data.items():
        for type, smiles_pair in category.items():
            for smiles, number in smiles_pair.items():
                if smiles not in smiles_list:
                    smiles_list[smiles] = int(number)
                else:
                    smiles_list[smiles] = smiles_list[smiles] + int(number)
    if filter > 0:
        result = { k:v for k,v in smiles_list.items() if v > filter}
    else:
        result = smiles_list

    return result, result.keys()

if __name__ == '__main__':
    x = from_json('../data/stru_data.json')
    subdata = SubstructureDataset(x[1])
    pass