import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from tqdm import tqdm
import json


class MoleculeGraph:
    """分子图数据结构"""

    def __init__(self, smiles):
        self.smiles = smiles
        self.mol = Chem.MolFromSmiles(smiles)
        if self.mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        Chem.SanitizeMol(self.mol)

        # 构建图
        self.node_features = self._get_node_features()
        self.edge_features, self.edge_index = self._get_edge_features()
        self.n_nodes = len(self.node_features)

    def _get_node_features(self):
        """提取原子特征"""
        features = []
        for atom in self.mol.GetAtoms():
            # 原子类型 (one-hot)
            atom_type = atom.GetAtomicNum()
            # 度的one-hot
            degree = atom.GetDegree()
            # 形式电荷
            charge = atom.GetFormalCharge()
            # 杂化方式
            hybrid = atom.GetHybridization()
            # 是否在环中
            in_ring = int(atom.IsInRing())
            # 芳香性
            aromatic = int(atom.GetIsAromatic())

            feat = [
                atom_type / 100.0,  # 归一化
                degree / 6.0,
                charge,
                hybrid / 8.0,
                in_ring,
                aromatic
            ]
            features.append(feat)
        return torch.tensor(features, dtype=torch.float)

    def _get_edge_features(self):
        """提取边特征"""
        edge_features = []
        edge_index = [[], []]

        for bond in self.mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            # 键类型
            bond_type = bond.GetBondType()
            bond_type_onehot = {
                Chem.BondType.SINGLE: [1, 0, 0, 0],
                Chem.BondType.DOUBLE: [0, 1, 0, 0],
                Chem.BondType.TRIPLE: [0, 0, 1, 0],
                Chem.BondType.AROMATIC: [0, 0, 0, 1]
            }.get(bond_type, [0, 0, 0, 0])

            # 是否在环中
            in_ring = int(bond.IsInRing())

            feat = bond_type_onehot + [in_ring]
            edge_features.append(feat)
            edge_features.append(feat)  # 双向

            edge_index[0].append(i)
            edge_index[1].append(j)
            edge_index[0].append(j)
            edge_index[1].append(i)

        if len(edge_features) == 0:
            # 处理孤立原子
            return torch.tensor([], dtype=torch.float), torch.tensor([[], []], dtype=torch.long)

        return torch.tensor(edge_features, dtype=torch.float), torch.tensor(edge_index, dtype=torch.long)

    def to_dense(self, max_nodes):
        """转换为稠密矩阵（用于批处理）"""
        node_feat = F.pad(self.node_features, (0, 0, 0, max_nodes - self.n_nodes), value=0)

        edge_feat = torch.zeros(max_nodes, max_nodes, self.edge_features.shape[-1])
        if len(self.edge_index) > 0 and len(self.edge_index[0]) > 0:
            for k, (i, j) in enumerate(zip(self.edge_index[0], self.edge_index[1])):
                if i < max_nodes and j < max_nodes:
                    edge_feat[i, j] = self.edge_features[k]

        mask = torch.zeros(max_nodes, dtype=torch.bool)
        mask[:self.n_nodes] = True

        return node_feat, edge_feat, mask, self.n_nodes


class SubstructureDataset(Dataset):
    """子结构数据集"""

    def __init__(self, smiles_list, max_nodes=50):
        self.smiles_list = smiles_list
        self.max_nodes = max_nodes
        self.graphs = []

        for smiles in tqdm(smiles_list, desc="Processing molecules"):
            try:
                graph = MoleculeGraph(smiles)
                if graph.n_nodes <= max_nodes:
                    self.graphs.append(graph)
            except Exception as e:
                print(e)
                continue

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        graph = self.graphs[idx]
        node_feat, edge_feat, mask, n_nodes = graph.to_dense(self.max_nodes)
        return {
            'node_feat': node_feat,
            'edge_feat': edge_feat,
            'mask': mask,
            'n_nodes': n_nodes,
            'smiles': graph.smiles
        }


def collate_graphs(batch):
    """批处理函数"""
    node_feats = torch.stack([item['node_feat'] for item in batch])
    edge_feats = torch.stack([item['edge_feat'] for item in batch])
    masks = torch.stack([item['mask'] for item in batch])
    n_nodes = torch.tensor([item['n_nodes'] for item in batch])
    smiles = [item['smiles'] for item in batch]

    return {
        'node_feat': node_feats,
        'edge_feat': edge_feats,
        'mask': masks,
        'n_nodes': n_nodes,
        'smiles': smiles
    }

def from_json(json_file):
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

    return smiles_list, smiles_list.keys()

if __name__ == '__main__':
    x = from_json('../data/stru_data.json')
    subdata = SubstructureDataset(x[1])
    pass