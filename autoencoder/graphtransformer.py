import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from autoencoder.dataprocess import MoleculeGraph


class PositionalEncoding(nn.Module):
    """拉普拉斯位置编码"""

    def __init__(self, d_model, max_nodes=50):
        super().__init__()
        self.d_model = d_model
        self.max_nodes = max_nodes
        self.pos_embed = nn.Parameter(torch.randn(max_nodes, d_model) * 0.1)

    def forward(self, x, mask):
        # x: B × N × d_model
        B, N, _ = x.shape
        pos = self.pos_embed[:N, :].unsqueeze(0).expand(B, -1, -1)
        return x + pos


class GraphMultiHeadAttention(nn.Module):
    """多头图注意力（修复版）"""

    def __init__(self, d_model, n_heads, d_edge, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        # 修复：确保边特征处理正确
        self.w_edge = nn.Linear(d_edge, n_heads)
        self.w_out = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, node_feat, edge_feat, mask=None):
        # node_feat: B × N × d_model
        # edge_feat: B × N × N × d_edge
        # mask: B × N

        B, N, _ = node_feat.shape

        # 1. 计算Q, K, V
        Q = self.w_q(node_feat).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(node_feat).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(node_feat).view(B, N, self.n_heads, self.d_k).transpose(1, 2)

        # 2. 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)  # B × n_heads × N × N

        # 3. 添加边偏置（修复版）
        # 确保edge_feat的维度正确
        edge_bias = self.w_edge(edge_feat)  # B × N × N × n_heads
        edge_bias = edge_bias.permute(0, 3, 1, 2)  # B × n_heads × N × N

        # 修复：确保尺寸一致
        if edge_bias.shape != scores.shape:
            # 如果尺寸不匹配，进行裁剪或填充
            min_N = min(edge_bias.shape[-1], scores.shape[-1])
            edge_bias = edge_bias[:, :, :min_N, :min_N]
            scores = scores[:, :, :min_N, :min_N]
            N = min_N

        scores = scores + edge_bias

        # 4. 应用掩码
        if mask is not None:
            # mask: B × N
            mask = mask.unsqueeze(1).unsqueeze(2)  # B × 1 × 1 × N
            mask = mask[:, :, :, :N]  # 确保尺寸匹配
            scores = scores.masked_fill(~mask, -1e9)

        # 5. Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 6. 加权聚合
        output = torch.matmul(attn_weights, V)  # B × n_heads × N × d_k
        output = output.transpose(1, 2).contiguous().view(B, N, self.d_model)

        return self.w_out(output)


class GraphTransformerLayer(nn.Module):
    """Graph Transformer层（修复版）"""

    def __init__(self, d_model, n_heads, d_edge, dropout=0.1):
        super().__init__()
        self.attention = GraphMultiHeadAttention(d_model, n_heads, d_edge, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, node_feat, edge_feat, mask=None):
        # node_feat: B × N × d_model
        # edge_feat: B × N × N × d_edge

        # 修复：确保node_feat和edge_feat的N维度一致
        if node_feat.shape[1] != edge_feat.shape[1]:
            # 如果尺寸不一致，调整到较小值
            min_N = min(node_feat.shape[1], edge_feat.shape[1])
            node_feat = node_feat[:, :min_N, :]
            edge_feat = edge_feat[:, :min_N, :min_N, :]
            if mask is not None:
                mask = mask[:, :min_N]

        # 自注意力
        attn_out = self.attention(node_feat, edge_feat, mask)
        node_feat = self.norm1(node_feat + attn_out)

        # 前馈网络
        ffn_out = self.ffn(node_feat)
        node_feat = self.norm2(node_feat + ffn_out)

        return node_feat


class GraphTransformerAutoencoder(nn.Module):
    """基于Graph Transformer的自编码器（修复版）"""

    def __init__(self,
                 node_dim=6,
                 edge_dim=5,
                 d_model=256,
                 latent_dim=128,
                 n_heads=8,
                 n_layers=6,
                 max_nodes=50,
                 dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.latent_dim = latent_dim
        self.max_nodes = max_nodes

        # 1. 输入嵌入
        self.node_embed = nn.Linear(node_dim, d_model)
        self.edge_embed = nn.Linear(edge_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_nodes)

        # 2. 虚拟节点
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # 3. 编码器
        self.encoder_layers = nn.ModuleList([
            GraphTransformerLayer(d_model, n_heads, d_model, dropout)
            for _ in range(n_layers)
        ])

        # 4. 池化层
        self.pool_proj = nn.Linear(d_model, d_model)

        # 5. 潜在空间 (VAE)
        self.mu = nn.Linear(d_model, latent_dim)
        self.logvar = nn.Linear(d_model, latent_dim)

        # 6. 解码器初始嵌入
        self.decoder_init = nn.Linear(latent_dim, d_model)
        self.decoder_pos = PositionalEncoding(d_model, max_nodes)

        # 7. 解码器
        self.decoder_layers = nn.ModuleList([
            GraphTransformerLayer(d_model, n_heads, d_model, dropout)
            for _ in range(n_layers)
        ])

        # 8. 输出头
        self.atom_predictor = nn.Linear(d_model, 100)  # 原子类型数
        self.edge_exist_predictor = nn.Linear(d_model * 2, 1)
        self.edge_type_predictor = nn.Linear(d_model * 2, 4)  # 4种键类型

        # 9. 长度预测
        self.length_predictor = nn.Sequential(
            nn.Linear(latent_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )

        self.dropout = nn.Dropout(dropout)

    def reparameterize(self, mu, logvar):
        """重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, node_feat, edge_feat, mask):
        """编码器（修复版）"""
        B, N, _ = node_feat.shape

        # 修复：确保edge_feat与node_feat的N一致
        if edge_feat.shape[1] != N:
            # 如果边特征维度不对，进行截断或填充
            min_N = min(edge_feat.shape[1], N)
            node_feat = node_feat[:, :min_N, :]
            edge_feat = edge_feat[:, :min_N, :min_N, :]
            if mask is not None:
                mask = mask[:, :min_N]
            N = min_N

        # 1. 嵌入
        node_emb = self.node_embed(node_feat)  # B × N × d_model
        edge_emb = self.edge_embed(edge_feat)  # B × N × N × d_model

        # 2. 位置编码
        node_emb = self.pos_encoding(node_emb, mask)

        # 3. 添加虚拟节点
        cls_token = self.cls_token.expand(B, -1, -1)
        node_emb = torch.cat([cls_token, node_emb], dim=1)  # B × (N+1) × d_model

        # 更新mask
        cls_mask = torch.ones(B, 1, dtype=torch.bool, device=mask.device)
        mask_ext = torch.cat([cls_mask, mask], dim=1)

        # 修复：扩展边特征以匹配新的节点数
        edge_emb_ext = torch.zeros(B, N + 1, N + 1, edge_emb.shape[-1], device=edge_emb.device)
        edge_emb_ext[:, 1:, 1:, :] = edge_emb

        # 4. Graph Transformer编码
        for layer in self.encoder_layers:
            node_emb = layer(node_emb, edge_emb_ext, mask_ext)

        # 5. 提取图表示 (使用虚拟节点)
        graph_emb = node_emb[:, 0, :]  # B × d_model

        # 6. VAE参数
        mu = self.mu(graph_emb)
        logvar = self.logvar(graph_emb)
        z = self.reparameterize(mu, logvar)

        return z, mu, logvar

    def decode(self, z, target_length=None):
        """解码器（修复版）"""
        B = z.shape[0]

        # 1. 预测长度
        if target_length is None:
            length_pred = self.length_predictor(z).squeeze(-1)  # B
            target_length = torch.round(torch.exp(length_pred)).clamp(1, self.max_nodes)
            target_length = target_length.long()
        else:
            target_length = target_length.long().clamp(1, self.max_nodes)

        # 2. 初始化节点特征
        node_feat = self.decoder_init(z).unsqueeze(1)  # B × 1 × d_model
        node_feat = node_feat.expand(-1, self.max_nodes, -1)  # B × max_nodes × d_model

        # 3. 位置编码
        mask = torch.arange(self.max_nodes, device=z.device).unsqueeze(0) < target_length.unsqueeze(1)
        node_feat = self.decoder_pos(node_feat, mask)

        # 4. 初始化边特征 (全零)
        edge_feat = torch.zeros(B, self.max_nodes, self.max_nodes, self.d_model, device=z.device)

        # 5. Graph Transformer解码
        for layer in self.decoder_layers:
            node_feat = layer(node_feat, edge_feat, mask)

        # 6. 预测原子类型
        atom_logits = self.atom_predictor(node_feat)  # B × max_nodes × 100

        # 7. 预测边（修复版 - 只预测有效位置）
        edge_exist_logits = []
        edge_type_logits = []

        max_edges = self.max_nodes * (self.max_nodes - 1) // 2
        for i in range(self.max_nodes):
            for j in range(i + 1, self.max_nodes):
                pair_feat = torch.cat([node_feat[:, i, :], node_feat[:, j, :]], dim=-1)
                exist_logit = self.edge_exist_predictor(pair_feat)
                type_logit = self.edge_type_predictor(pair_feat)
                edge_exist_logits.append(exist_logit)
                edge_type_logits.append(type_logit)

        if edge_exist_logits:
            edge_exist_logits = torch.stack(edge_exist_logits, dim=1)  # B × num_edges × 1
            edge_type_logits = torch.stack(edge_type_logits, dim=1)  # B × num_edges × 4
        else:
            # 如果没有边，创建空的tensor
            edge_exist_logits = torch.zeros(B, 0, 1, device=z.device)
            edge_type_logits = torch.zeros(B, 0, 4, device=z.device)

        return atom_logits, edge_exist_logits, edge_type_logits, target_length

    def forward(self, node_feat, edge_feat, mask):
        """前向传播"""
        # 编码
        z, mu, logvar = self.encode(node_feat, edge_feat, mask)

        # 解码
        atom_logits, edge_exist_logits, edge_type_logits, pred_length = self.decode(z)

        return {
            'z': z,
            'mu': mu,
            'logvar': logvar,
            'atom_logits': atom_logits,
            'edge_exist_logits': edge_exist_logits,
            'edge_type_logits': edge_type_logits,
            'pred_length': pred_length
        }


class GraphAutoencoderLoss(nn.Module):
    """自编码器损失函数（修复版）"""

    def __init__(self, atom_weight=1.0, edge_weight=1.0, length_weight=0.1, kl_weight=0.001):
        super().__init__()
        self.atom_weight = atom_weight
        self.edge_weight = edge_weight
        self.length_weight = length_weight
        self.kl_weight = kl_weight

    def forward(self, pred, target):
        """计算损失（修复版）"""
        # 1. 原子类型损失
        B, N, _ = target['node_feat'].shape
        target_atoms = target['node_feat'][:, :, 0]  # 使用第一个特征作为原子类型
        target_atoms = (target_atoms * 100).long()  # 还原原子序数

        # 修复：确保预测和目标尺寸一致
        pred_atoms = pred['atom_logits'][:, :N, :]  # 只取有效长度

        atom_loss = F.cross_entropy(
            pred_atoms.permute(0, 2, 1),
            target_atoms,
            ignore_index=0
        )

        # 2. 边损失（修复版）
        target_edges = target['edge_feat'][:, :, :, 0]  # 边存在性

        # 构建目标边标签
        edge_labels = []
        for b in range(B):
            for i in range(N):
                for j in range(i + 1, N):
                    edge_labels.append(target_edges[b, i, j] > 0)

        if edge_labels:
            edge_labels = torch.tensor(edge_labels, dtype=torch.float,
                                       device=pred['edge_exist_logits'].device)
            edge_labels = edge_labels.view(B, -1)

            # 修复：确保预测和标签的边数量一致
            pred_edges = pred['edge_exist_logits'][:, :edge_labels.shape[1], :]

            # 边存在性损失
            edge_exist_loss = F.binary_cross_entropy_with_logits(
                pred_edges.squeeze(-1),
                edge_labels
            )
        else:
            edge_exist_loss = torch.tensor(0.0, device=pred['edge_exist_logits'].device)

        # 3. 长度损失
        target_length = target['n_nodes'].float()
        length_loss = F.mse_loss(pred['pred_length'].float(), target_length)

        # 4. KL散度
        kl_loss = -0.5 * torch.sum(1 + pred['logvar'] - pred['mu'].pow(2) - pred['logvar'].exp())
        kl_loss = kl_loss / pred['mu'].shape[0]

        # 总损失
        total_loss = (self.atom_weight * atom_loss +
                      self.edge_weight * edge_exist_loss +
                      self.length_weight * length_loss +
                      self.kl_weight * kl_loss)

        return {
            'total_loss': total_loss,
            'atom_loss': atom_loss,
            'edge_exist_loss': edge_exist_loss,
            'length_loss': length_loss,
            'kl_loss': kl_loss
        }