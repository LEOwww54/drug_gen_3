import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from autoencoder.dataprocess import MoleculeGraph
from constant import device


class PositionalEncoding(nn.Module):
    """位置编码"""

    def __init__(self, d_model, max_nodes=50):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(max_nodes, d_model) * 0.1)

    def forward(self, x, mask):
        B, N, _ = x.shape
        pos = self.pos_embed[:N, :].unsqueeze(0).expand(B, -1, -1)
        return x + pos


class GraphMultiHeadAttention(nn.Module):
    """多头图注意力 - 简化版"""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_out = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, node_feat, mask=None):
        B, N, _ = node_feat.shape

        # 计算Q, K, V
        Q = self.w_q(node_feat).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(node_feat).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(node_feat).view(B, N, self.n_heads, self.d_k).transpose(1, 2)

        # 注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)

        # 应用掩码
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(~mask, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, V)
        output = output.transpose(1, 2).contiguous().view(B, N, self.d_model)

        return self.w_out(output)


class GraphTransformerLayer(nn.Module):
    """Graph Transformer层"""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.attention = GraphMultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, node_feat, mask=None):
        # 自注意力
        attn_out = self.attention(node_feat, mask)
        node_feat = self.norm1(node_feat + attn_out)

        # 前馈网络
        ffn_out = self.ffn(node_feat)
        node_feat = self.norm2(node_feat + ffn_out)

        return node_feat


class GraphTransformerAutoencoder(nn.Module):
    """
    简化的Graph Transformer自编码器
    只关注原子类型和键类型的重建
    """

    def __init__(self,
                 atom_vocab_size=100,  # 原子类型词汇表大小
                 bond_vocab_size=5,  # 键类型词汇表大小
                 d_model=256,
                 latent_dim=128,
                 n_heads=8,
                 n_layers=4,
                 max_nodes=30,
                 dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.latent_dim = latent_dim
        self.max_nodes = max_nodes
        self.atom_vocab_size = atom_vocab_size
        self.bond_vocab_size = bond_vocab_size

        # 1. 原子类型嵌入
        self.atom_embed = nn.Linear(atom_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_nodes)

        # 2. 虚拟节点（用于图池化）
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # 3. 编码器
        self.encoder_layers = nn.ModuleList([
            GraphTransformerLayer(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])

        # 4. 潜在空间 (VAE)
        self.mu = nn.Linear(d_model, latent_dim)
        self.logvar = nn.Linear(d_model, latent_dim)

        # 5. 解码器
        self.decoder_init = nn.Linear(latent_dim, d_model)
        self.decoder_layers = nn.ModuleList([
            GraphTransformerLayer(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])

        # 6. 输出头 - 主要任务
        self.atom_predictor = nn.Linear(d_model, atom_vocab_size)
        self.bond_predictor = nn.Linear(d_model * 2, bond_vocab_size)

        # 7. 长度预测 - 改为分类任务 (one-hot编码)
        # 预测每个可能长度的概率
        self.length_predictor = nn.Sequential(
            nn.Linear(latent_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, max_nodes + 1)  # +1 用于padding/无效长度
        )

        self.dropout = nn.Dropout(dropout)

    def reparameterize(self, mu, logvar):
        """重参数化"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, atom_feat, mask):
        """编码器"""
        B, N, _ = atom_feat.shape

        # 1. 嵌入
        node_emb = self.atom_embed(atom_feat)  # B × N × d_model

        # 2. 位置编码
        node_emb = self.pos_encoding(node_emb, mask)

        # 3. 添加虚拟节点
        cls_token = self.cls_token.expand(B, -1, -1)
        node_emb = torch.cat([cls_token, node_emb], dim=1)  # B × (N+1) × d_model

        # 更新掩码
        cls_mask = torch.ones(B, 1, dtype=torch.bool, device=mask.device)
        mask_ext = torch.cat([cls_mask, mask], dim=1)

        # 4. Graph Transformer编码
        for layer in self.encoder_layers:
            node_emb = layer(node_emb, mask_ext)

        # 5. 提取图表示
        graph_emb = node_emb[:, 0, :]  # B × d_model

        # 6. VAE参数
        mu = self.mu(graph_emb)
        logvar = self.logvar(graph_emb)
        z = self.reparameterize(mu, logvar)

        return z, mu, logvar

    def decode(self, z, target_length=None):
        """解码器"""
        B = z.shape[0]

        # 1. 预测长度 (分类任务)
        length_logits = self.length_predictor(z)  # B × (max_nodes + 1)

        if target_length is None:
            # 使用argmax获取预测的长度
            target_length = length_logits.argmax(dim=-1)  # B
            # 确保长度在有效范围内
            target_length = target_length.clamp(1, self.max_nodes)
        else:
            target_length = target_length.long().clamp(1, self.max_nodes)

        # 2. 初始化节点特征
        node_feat = self.decoder_init(z).unsqueeze(1)  # B × 1 × d_model
        node_feat = node_feat.expand(-1, self.max_nodes, -1)

        # 3. 位置编码
        mask = torch.arange(self.max_nodes, device=z.device).unsqueeze(0) < target_length.unsqueeze(1)
        node_feat = self.pos_encoding(node_feat, mask)

        # 4. Graph Transformer解码
        for layer in self.decoder_layers:
            node_feat = layer(node_feat, mask)

        # 5. 预测原子类型
        atom_logits = self.atom_predictor(node_feat)  # B × max_nodes × atom_vocab_size

        # 6. 预测键类型（只预测上三角）
        edge_logits = []
        edge_indices = []

        for i in range(self.max_nodes):
            for j in range(i + 1, self.max_nodes):
                pair_feat = torch.cat([node_feat[:, i, :], node_feat[:, j, :]], dim=-1)
                logit = self.bond_predictor(pair_feat)  # B × bond_vocab_size
                edge_logits.append(logit)
                edge_indices.append((i, j))

        if edge_logits:
            edge_logits = torch.stack(edge_logits, dim=1)  # B × num_edges × bond_vocab_size
        else:
            edge_logits = torch.zeros(B, 0, self.bond_vocab_size, device=z.device)

        return atom_logits, edge_logits, target_length, length_logits

    def forward(self, atom_feat, mask, target_length=None):
        """前向传播"""
        # 编码
        z, mu, logvar = self.encode(atom_feat, mask)

        # 解码
        atom_logits, edge_logits, pred_length, length_logits = self.decode(z, target_length)

        return {
            'z': z,
            'mu': mu,
            'logvar': logvar,
            'atom_logits': atom_logits,
            'edge_logits': edge_logits,
            'pred_length': pred_length,
            'length_logits': length_logits  # 添加长度logits输出
        }

    def encode_smiles(self, smiles, max_nodes=30):
        """编码SMILES为潜在向量"""
        graph = MoleculeGraph(smiles, max_nodes)
        atom_feat, _, mask, _ = graph.to_dense(max_nodes)
        atom_feat = atom_feat.unsqueeze(0)
        mask = mask.unsqueeze(0)

        with torch.no_grad():
            z, _, _ = self.encode(atom_feat, mask)
        return z.squeeze(0)

    def decode_to_graph(self, z, max_nodes=30):
        """从潜在向量解码为图"""
        with torch.no_grad():
            # 确保z是2D
            if z.dim() == 1:
                z = z.unsqueeze(0)

            atom_logits, edge_logits, pred_length, _ = self.decode(z)

            # 获取预测
            n = pred_length.item()

            # 原子类型
            atom_probs = F.softmax(atom_logits[0, :n], dim=-1)
            atom_indices = atom_probs.argmax(dim=-1).cpu().numpy()

            # 键类型
            edge_probs = F.softmax(edge_logits[0], dim=-1)

            # 构建分子
            mol = Chem.RWMol()
            atom_map = {}

            # 添加原子
            atom_symbols = ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'B']
            for i, idx in enumerate(atom_indices):
                if idx == 0:  # padding
                    continue
                if idx <= len(atom_symbols):
                    atom = Chem.Atom(atom_symbols[idx - 1])
                    mol.AddAtom(atom)
                    atom_map[i] = mol.GetNumAtoms() - 1

            # 添加键
            edge_idx = 0
            for i in range(n):
                for j in range(i + 1, n):
                    if edge_idx < len(edge_probs):
                        bond_type_idx = edge_probs[edge_idx].argmax().item()
                        if bond_type_idx > 0:  # 有键
                            bond_map = {
                                1: Chem.BondType.SINGLE,
                                2: Chem.BondType.DOUBLE,
                                3: Chem.BondType.TRIPLE,
                                4: Chem.BondType.AROMATIC,
                            }
                            if i in atom_map and j in atom_map:
                                mol.AddBond(atom_map[i], atom_map[j], bond_map.get(bond_type_idx, Chem.BondType.SINGLE))
                    edge_idx += 1

            try:
                mol = mol.GetMol()
                Chem.SanitizeMol(mol)
                smiles = Chem.MolToSmiles(mol)
                return smiles
            except:
                return None


class GraphAutoencoderLoss(nn.Module):
    """
    自编码器损失函数
    只计算原子类型和键类型的重建损失
    """

    def __init__(self, atom_weight=1.0, edge_weight=1.0, length_weight=0.1, kl_weight=0.001, max_nodes=30):
        super().__init__()
        self.atom_weight = atom_weight
        self.edge_weight = edge_weight
        self.length_weight = length_weight
        self.kl_weight = kl_weight
        self.max_nodes = max_nodes

    def forward(self, pred, target):
        # target: {'atom_feat': atom_feat, 'bond_feat': bond_feat, 'mask': mask, 'n_nodes': n_nodes}

        B, N, _ = target['atom_feat'].shape

        # 1. 原子类型损失 (交叉熵)
        target_atoms = target['atom_feat']  # one-hot
        target_atom_indices = target_atoms.argmax(dim=-1).to(device)  # B × N

        # 只计算有效位置
        mask = target['mask']
        atom_loss = F.cross_entropy(
            pred['atom_logits'].permute(0, 2, 1),  # B × vocab × N
            target_atom_indices,
            ignore_index=0  # padding
        )

        # 2. 键类型损失 (交叉熵)
        # 构建目标键类型
        target_bonds = target['bond_feat']  # B × N × N × vocab
        target_bond_indices = target_bonds.argmax(dim=-1).to(device)  # B × N × N

        # 只考虑上三角
        edge_indices = []
        target_edge_types = []

        for i in range(N):
            for j in range(i + 1, N):
                edge_indices.append((i, j))
                target_edge_types.append(target_bond_indices[:, i, j])

        if edge_indices:
            target_edge_types = torch.stack(target_edge_types, dim=1)  # B × num_edges

            edge_loss = F.cross_entropy(
                pred['edge_logits'].permute(0, 2, 1),  # B × vocab × num_edges
                target_edge_types,
                ignore_index=0
            )
        else:
            edge_loss = torch.tensor(0.0, device=pred['edge_logits'].device)

        # 3. 长度预测损失 - 改为交叉熵 (分类任务)
        target_length = target['n_nodes'].to(device)  # B
        # 确保目标长度在有效范围内
        target_length = target_length.clamp(1, self.max_nodes)

        # 创建one-hot目标
        # length_logits 的形状是 B × (max_nodes + 1)
        length_loss = F.cross_entropy(
            pred['length_logits'],  # B × (max_nodes + 1)
            target_length,
            ignore_index=0  # 忽略无效长度 (如果有的话)
        )

        # 4. KL散度
        kl_loss = -0.5 * torch.sum(1 + pred['logvar'] - pred['mu'].pow(2) - pred['logvar'].exp())
        kl_loss = kl_loss / pred['mu'].shape[0]

        # 总损失
        total_loss = (self.atom_weight * atom_loss +
                      self.edge_weight * edge_loss +
                      self.length_weight * length_loss +
                      self.kl_weight * kl_loss)

        return {
            'total_loss': total_loss,
            'atom_loss': atom_loss,
            'edge_loss': edge_loss,
            'length_loss': length_loss,
            'kl_loss': kl_loss
        }


class PropertyPredictionHead(nn.Module):
    """
    可选的属性预测头
    可以添加各种性质预测，不影响主要自编码器
    """

    def __init__(self, latent_dim, hidden_dim=128):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)  # 单值预测
        )

    def forward(self, z):
        return self.predictor(z)


class MultiPropertyPredictor(nn.Module):
    """
    多属性预测头
    可以预测多个性质
    """

    def __init__(self, latent_dim, num_properties=5, hidden_dim=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.property_heads = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(num_properties)
        ])

    def forward(self, z):
        shared = self.shared(z)
        return [head(shared).squeeze(-1) for head in self.property_heads]


class ExtendedAutoencoder(nn.Module):
    """
    扩展的自编码器：主要自编码器 + 辅助预测头
    """

    def __init__(self, autoencoder, property_predictor=None):
        super().__init__()
        self.autoencoder = autoencoder
        self.property_predictor = property_predictor

    def forward(self, atom_feat, mask):
        # 自编码器
        ae_output = self.autoencoder(atom_feat, mask)

        # 属性预测（可选）
        if self.property_predictor is not None:
            properties = self.property_predictor(ae_output['z'])
            ae_output['properties'] = properties

        return ae_output

    def compute_loss(self, pred, target, property_targets=None):
        """计算总损失：自编码器损失 + 属性预测损失"""
        # 自编码器损失
        ae_loss_fn = GraphAutoencoderLoss()
        ae_loss = ae_loss_fn(pred, target)

        total_loss = ae_loss['total_loss']

        # 属性预测损失（如果有）
        if property_targets is not None and 'properties' in pred:
            prop_loss = 0
            for pred_prop, target_prop in zip(pred['properties'], property_targets):
                prop_loss += F.mse_loss(pred_prop, target_prop)
            total_loss += prop_loss
            ae_loss['property_loss'] = prop_loss

        ae_loss['total_loss'] = total_loss
        return ae_loss