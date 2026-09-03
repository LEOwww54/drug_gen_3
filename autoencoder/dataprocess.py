import torch
import torch.nn.functional as F
from pytorch_lightning.trainer.connectors.logger_connector import result
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from tqdm import tqdm
import json


class MoleculeGraph:
    """分子图数据结构（修复版）"""

    def __init__(self, smiles, max_nodes=30):
        self.smiles = smiles
        self.max_nodes = max_nodes
        self.mol = Chem.MolFromSmiles(smiles)
        if self.mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        Chem.SanitizeMol(self.mol)

        # 构建图
        self.node_features = self._get_node_features()
        self.edge_features, self.edge_index = self._get_edge_features()
        self.n_nodes = len(self.node_features)

        # 确保边特征维度一致
        self.edge_dim = 5  # 固定边特征维度

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
        """提取边特征（修复版）"""
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

        # 修复：如果没有边，返回空的边特征但保持维度一致
        if len(edge_features) == 0:
            # 返回空的边特征，但维度为 [0, 5]
            return torch.tensor([], dtype=torch.float), torch.tensor([[], []], dtype=torch.long)

        return torch.tensor(edge_features, dtype=torch.float), torch.tensor(edge_index, dtype=torch.long)

    def to_dense(self, max_nodes):
        """转换为稠密矩阵（修复版）"""
        # 节点特征填充
        node_feat = torch.zeros(max_nodes, self.node_features.shape[-1])
        node_feat[:self.n_nodes] = self.node_features

        # 边特征填充 - 修复：确保维度一致
        edge_dim = 5  # 固定边特征维度
        edge_feat = torch.zeros(max_nodes, max_nodes, edge_dim)

        if len(self.edge_index) > 0 and len(self.edge_index[0]) > 0:
            # 有边的情况
            for k, (i, j) in enumerate(zip(self.edge_index[0], self.edge_index[1])):
                if i < max_nodes and j < max_nodes and k < len(self.edge_features):
                    # 确保边特征维度正确
                    feat = self.edge_features[k]
                    if len(feat) == edge_dim:
                        edge_feat[i, j] = feat
                    else:
                        # 如果维度不对，填充零
                        edge_feat[i, j] = torch.zeros(edge_dim)

        mask = torch.zeros(max_nodes, dtype=torch.bool)
        mask[:self.n_nodes] = True

        return node_feat, edge_feat, mask, self.n_nodes


class SubstructureDataset(Dataset):
    """子结构数据集（修复版）"""

    def __init__(self, smiles_list, max_nodes=30):
        self.smiles_list = smiles_list
        self.max_nodes = max_nodes
        self.graphs = []
        self.edge_dim = 5  # 固定边特征维度

        print(f"Processing {len(smiles_list)} molecules...")
        for smiles in tqdm(smiles_list, desc="Processing molecules"):
            try:
                graph = MoleculeGraph(smiles, max_nodes)
                if graph.n_nodes <= max_nodes and graph.n_nodes > 0:
                    self.graphs.append(graph)
            except Exception as e:
                print(f"Warning: Failed to process {smiles}: {e}")
                continue

        print(f"Successfully processed {len(self.graphs)} molecules")

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        graph = self.graphs[idx]
        node_feat, edge_feat, mask, n_nodes = graph.to_dense(self.max_nodes)

        # 验证边特征维度
        if edge_feat.shape[-1] != self.edge_dim:
            # 修复维度
            new_edge_feat = torch.zeros(self.max_nodes, self.max_nodes, self.edge_dim)
            copy_dim = min(edge_feat.shape[-1], self.edge_dim)
            new_edge_feat[:, :, :copy_dim] = edge_feat[:, :, :copy_dim]
            edge_feat = new_edge_feat

        return {
            'node_feat': node_feat,
            'edge_feat': edge_feat,
            'mask': mask,
            'n_nodes': n_nodes,
            'smiles': graph.smiles
        }


def collate_graphs(batch):
    """批处理函数（修复版）"""
    # 获取批次中最大节点数
    max_nodes = max([item['node_feat'].shape[0] for item in batch])
    edge_dim = 5  # 固定边特征维度

    # 统一填充
    node_feats = []
    edge_feats = []
    masks = []
    n_nodes = []

    for item in batch:
        node_feat = item['node_feat']
        edge_feat = item['edge_feat']
        mask = item['mask']
        n = item['n_nodes']

        # 填充节点特征
        padded_node = torch.zeros(max_nodes, node_feat.shape[-1])
        padded_node[:n] = node_feat[:n]
        node_feats.append(padded_node)

        # 填充边特征 - 修复：确保维度一致
        padded_edge = torch.zeros(max_nodes, max_nodes, edge_dim)
        # 只复制有效的边特征
        if edge_feat.shape[0] > 0 and edge_feat.shape[1] > 0:
            valid_n = min(n, edge_feat.shape[0], edge_feat.shape[1])
            if edge_feat.shape[-1] == edge_dim:
                padded_edge[:valid_n, :valid_n] = edge_feat[:valid_n, :valid_n]
            else:
                # 如果边特征维度不对，只复制能复制的部分
                copy_dim = min(edge_feat.shape[-1], edge_dim)
                padded_edge[:valid_n, :valid_n, :copy_dim] = edge_feat[:valid_n, :valid_n, :copy_dim]
        edge_feats.append(padded_edge)

        # 填充掩码
        padded_mask = torch.zeros(max_nodes, dtype=torch.bool)
        padded_mask[:n] = mask[:n]
        masks.append(padded_mask)

        n_nodes.append(n)

    return {
        'node_feat': torch.stack(node_feats),
        'edge_feat': torch.stack(edge_feats),
        'mask': torch.stack(masks),
        'n_nodes': torch.tensor(n_nodes),
        'smiles': [item['smiles'] for item in batch]
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