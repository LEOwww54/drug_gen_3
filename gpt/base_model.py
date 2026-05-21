import torch
from torch import nn, optim
import numpy as np
import math
import constant

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, attn_mask):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(
            self.d_k)  # scores : [batch_size, n_heads, len_q, len_k]
        if attn_mask is not None:
            scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is True.

        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
        return context, attn


import torch
import torch.nn as nn
import numpy as np


class RiemannianScaledDotProductAttention(nn.Module):
    def __init__(self, learnable_G=True, eps=1e-8, lambda_max=1.0):
        '''
        d_k: 每个头的维度
        n_heads: 注意力头数
        learnable_G: 是否学习每头的度量张量
        eps: 数值稳定项
        lambda_max: lambda的最大值（通过sigmoid缩放实现）
        '''
        super().__init__()
        self.d_k = constant.d_k
        self.n_heads = constant.n_heads
        self.eps = eps
        self.lambda_max = lambda_max

        # 1. 可学习的每头独立度量 G（推荐）
        if learnable_G:
            init_G = torch.eye(constant.d_k).unsqueeze(0).repeat(constant.n_heads, 1, 1)
            self.G = nn.Parameter(init_G)  # [n_heads, d_k, d_k]
        else:
            self.register_buffer('G', torch.eye(constant.d_k).unsqueeze(0).repeat(constant.n_heads, 1, 1))

        # 2. Lambda学习网络：将 v 映射到标量 lambda
        # v 的维度是 [batch_size, n_heads, d_k]
        # 设计一个轻量级MLP，每个头共享或独立
        self.lambda_mlp = nn.Sequential(
            nn.Linear(constant.d_k, constant.d_k // 2),
            nn.ReLU(),
            nn.Linear(constant.d_k // 2, 1),
            nn.Sigmoid()  # 输出范围 [0, 1]
        )

    def forward(self, Q, K, V, v_cond, attn_mask=None):
        '''
        Q, K, V: [batch_size, n_heads, len_q, d_k]
        v_cond: [batch_size, n_heads, d_k] 或 [batch_size, n_heads, 1, d_k]
                条件切向量，由条件编码器生成
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        batch_size, n_heads, len_q, d_k = Q.shape

        # ========== 1. 处理 v_cond 形状 ==========
        if v_cond.dim() == 3:
            # [B, H, d_k]
            v = v_cond
        elif v_cond.dim() == 4:
            # [B, H, 1, d_k] -> squeeze
            v = v_cond.squeeze(2)
        else:
            raise ValueError(f"v_cond shape must be [B, H, d_k] or [B, H, 1, d_k], got {v_cond.shape}")

        # ========== 2. 从 v 学习 lambda ==========
        # 每个 batch 和每个头独立计算 lambda
        # v: [B, H, d_k] -> lambda: [B, H, 1, 1]
        lambda_val = self.lambda_mlp(v)  # [B, H, 1]
        lambda_val = lambda_val.unsqueeze(-1)  # [B, H, 1, 1]
        # 缩放至 [0, lambda_max]
        lambda_val = lambda_val * self.lambda_max

        # ========== 3. 获取当前batch的度量 ==========
        G = self.G.unsqueeze(0).expand(batch_size, -1, -1, -1)  # [B, H, d_k, d_k]

        # ========== 4. 构造条件度量 G_cond = G + lambda * (v v^T)/||v||^2 ==========
        # 计算 v 的范数平方 [B, H, 1, 1]
        v_norm_sq = torch.sum(v ** 2, dim=-1, keepdim=True).clamp(min=self.eps)  # [B, H, 1]
        v_norm_sq = v_norm_sq.unsqueeze(-1)  # [B, H, 1, 1]

        # 构造 v v^T [B, H, d_k, d_k]
        v_expanded = v.unsqueeze(-1)  # [B, H, d_k, 1]
        vvT = torch.matmul(v_expanded, v_expanded.transpose(-1, -2))  # [B, H, d_k, d_k]

        # 条件更新量
        delta_G = lambda_val * vvT / (v_norm_sq + self.eps)  # [B, H, d_k, d_k]

        # 最终度量 [B, H, d_k, d_k]
        G_cond = G + delta_G

        # ========== 5. 扩展到每个query位置 ==========
        G_expanded = G_cond.unsqueeze(2)  # [B, H, 1, d_k, d_k]

        # ========== 6. 计算位移矩阵 D_ij = q_i - k_j ==========
        D = Q.unsqueeze(3) - K.unsqueeze(2)  # [B, H, len_q, len_k, d_k]

        # ========== 7. 黎曼距离平方 D^T @ G_cond @ D ==========
        # 方法：先计算 D @ G_cond，再与 D 点积
        DG = torch.einsum('b h q k d, b h e d -> b h q k e', D, G_expanded)
        riemann_dist_sq = torch.einsum('b h q k d, b h q k d -> b h q k', DG, D)

        # ========== 8. 计算注意力分数 ==========
        scores = -0.5 * riemann_dist_sq / np.sqrt(self.d_k)

        if attn_mask is not None:
            scores.masked_fill_(attn_mask, -1e9)

        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)  # [B, H, len_q, d_v]

        # 可选：返回 lambda_val 用于监控
        return context, attn, lambda_val

class DualChannelCrossAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads, p_type='lora_1', conditional=['protein', 'protein_pocket']):
        super().__init__()
        self.global_attn_ln = nn.LayerNorm(d_model)
        self.local_attn_ln = nn.LayerNorm(d_model)
        self.global_attention = MultiHeadAttention(d_model=d_model, d_k=d_k, d_v=d_v, n_heads=n_heads, p_type=p_type, conditional=conditional)
        self.local_attention = MultiHeadAttention(d_model=d_model, d_k=d_k, d_v=d_v, n_heads=n_heads, p_type=p_type, conditional=conditional)
        self.fusion_gate = nn.Linear(d_model * 2, d_model)

    def forward(self, query, key, value, attn_mask, focus_mask, y=None):
        """
        focus_mask: [batch, kv_len] 重点区域为1，其他为0
        """
        # 1. 全局注意力（看整个序列）
        global_out, _ = self.global_attention(query, key, value, attn_mask, y)
        global_out = self.global_attn_ln(global_out)

        # 2. 局部注意力（只看重点区域）
        # 将非重点区域的key/value置零或屏蔽
        local_out, _ = self.local_attention(query, key, value, focus_mask, y)
        local_out = self.local_attn_ln(local_out)

        # 3. 自适应融合
        combined = torch.cat([global_out, local_out], dim=-1)
        gate = torch.sigmoid(self.fusion_gate(combined))
        output = gate * global_out + (1 - gate) * local_out

        return output, None

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads, p_type='lora_1', conditional=['unconditional']):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.layernorm = nn.LayerNorm(d_model)
        self.p_type = p_type
        self.conditional = conditional

        if 'prop' in conditional:
            if p_type == 'lora_1':
                self.prop_encoder_lora_1 = nn.Sequential(
                    nn.Linear(constant.prop_len, 4),
                    nn.GELU(),
                    nn.Linear(4, 1),
                )

                self.lora_1_q = nn.Parameter(torch.randn(d_model, d_model) * 0.01)
                self.lora_1_k = nn.Parameter(torch.randn(d_model, d_model) * 0.01)
                self.lora_1_v = nn.Parameter(torch.randn(d_model, d_model) * 0.01)
                self.lora_1_o = nn.Parameter(torch.randn(d_model, d_model) * 0.01)
            elif p_type == 'Riemannian':
                self.Riemannian_encoder = nn.Sequential(
                    nn.Linear(constant.prop_len, self.d_k //2),
                    nn.GELU(),
                    nn.Linear(self.d_k //2, self.d_k),
                    nn.GELU(),
                    nn.Linear(self.d_k, self.d_k * n_heads),
                )
                self.Riemannian_attention = RiemannianScaledDotProductAttention()

        self.dropout = nn.Dropout(0.1)

    def forward(self, input_Q, input_K, input_V, attn_mask, y=None):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        p_type = self.p_type
        residual, batch_size = input_Q, input_Q.size(0)

        Q = self.W_Q(input_Q)
        K = self.W_K(input_K)
        V = self.W_V(input_V)

        if 'prop' in self.conditional:
            if y is not None:
                if p_type == 'lora_1':
                    y = self.prop_encoder_lora_1(y)
                    y = y.unsqueeze(-1)

                    lora_q = input_Q @ self.lora_1_q
                    lora_k = input_K @ self.lora_1_k
                    lora_v = input_V @ self.lora_1_v

                    lora_q = lora_q * y
                    lora_k = lora_k * y
                    lora_v = lora_v * y

                    Q = Q + lora_q
                    K = K + lora_k
                    V = V + lora_v


        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = Q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = K.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # K: [batch_size, n_heads, len_k, d_k]
        V = V.view(batch_size, -1, self.n_heads, self.d_v).transpose(1,
                                                                           2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1,
                                                  1)  # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        if p_type == 'Riemannian':
            y_t = self.Riemannian_encoder(y)
            y_t = y_t.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
            context, attn, _ = self.Riemannian_attention(Q, K, V, attn_mask, y_t)
        else:
            context, attn = ScaledDotProductAttention(self.d_k)(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1,
                                                  self.n_heads * self.d_v)  # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context)  # [batch_size, len_q, d_model]

        if 'prop' in  self.conditional:
            if y is not None:
                if p_type=='lora_1':
                    lora_o = context @ self.lora_1_o
                    lora_o = lora_o * y
                    output = output + lora_o

        # return self.layernorm(output + residual), attn
        return output, attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff, p_type='lora_0', conditional=['unconditional']):
        super(PoswiseFeedForwardNet, self).__init__()
        self.W1 = nn.Linear(d_model, d_ff)
        self.W2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(0.1)
        self.activation = nn.GELU()

        self.p_type = p_type
        self.conditional = conditional

        if 'prop' in self.conditional:
            if p_type=='lora_1':
                self.prop_encoder_lora_1 = nn.Sequential(
                    nn.Linear(constant.prop_len, 4),
                    nn.GELU(),
                    nn.Linear(4, 1),
                )

                self.lora_1_W1 = nn.Parameter(torch.randn(d_model, d_ff) * 0.01)
                self.lora_1_W2 = nn.Parameter(torch.zeros(d_ff, d_model) * 0.01)

        # self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'lora_0' in name:
                continue  # LoRA 已特殊初始化
            if len(param.shape) >= 2:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.zeros_(param)

    def forward(self, inputs, y=None):
        batch_size, seq_len, _ = inputs.shape

        # === 基础 FFN 路径 ===
        h_base = self.W1(inputs)  # [batch, seq_len, d_ff]

        # === LoRA 路径 ===
        # W1 的 LoRA 贡献
        if 'prop' in self.conditional:
            if self.p_type=='lora_1':
                y = self.prop_encoder_lora_1(y)
                y = y.unsqueeze(-1)

                lora_W1 = inputs @ self.lora_1_W1  # [batch, seq_len, d_ff]
                lora_W1 = lora_W1 * y
                # 合并基础路径和 LoRA 路径
                h = h_base + lora_W1
            else:
                h = h_base
        else:
            h = h_base

        # 激活
        h = self.activation(h)
        h = self.dropout(h)

        # === 输出层 ===
        out_base = self.W2(h)  # [batch, seq_len, d_model]

        if 'prop' in self.conditional:
            if self.p_type=='lora_1':
                lora_W2 = h @ self.lora_1_W2
                lora_W2 = lora_W2 * y
                # 合并输出
                output = out_base + lora_W2
            else:
                output = out_base
        else:
            output = out_base

        return output


class PositionalEncoding(nn.Module):
    """实现Positional Encoding功能"""

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        位置编码器的初始化函数
        :param d_model: 词向量的维度，与输入序列的特征维度相同，512
        :param dropout: 置零比率
        :param max_len: 句子最大长度,5000
        """
        super(PositionalEncoding, self).__init__()
        # 初始化一个nn.Dropout层，设置给定的dropout比例
        self.dropout = nn.Dropout(p=dropout)

        # 初始化一个位置编码矩阵
        # (5000,512)矩阵，保持每个位置的位置编码，一共5000个位置，每个位置用一个512维度向量来表示其位置编码
        pe = torch.zeros(max_len, d_model)
        # 偶数和奇数在公式上有一个共同部分，使用log函数把次方拿下来，方便计算
        # position表示的是字词在句子中的索引，如max_len是128，那么索引就是从0，1，2，...,127
        # 论文中d_model是512，2i符号中i从0取到255，那么2i对应取值就是0,2,4...510
        # (5000) -> (5000,1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 计算用于控制正余弦的系数，确保不同频率成分在d_model维空间内均匀分布
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # 根据位置和div_term计算正弦和余弦值，分别赋值给pe的偶数列和奇数列
        pe[:, 0::2] = torch.sin(position * div_term)  # 从0开始到最后面，补长为2，其实代表的就是偶数位置
        pe[:, 1::2] = torch.cos(position * div_term)  # 从1开始到最后面，补长为2，其实代表的就是奇数位置
        # 上面代码获取之后得到的pe:[max_len * d_model]
        # 下面这个代码之后得到的pe形状是：[1 * max_len * d_model]
        # 多增加1维，是为了适应batch_size
        # (5000, 512) -> (1, 5000, 512)
        pe = pe.unsqueeze(0)
        # 将计算好的位置编码矩阵注册为模块缓冲区（buffer），这意味着它将成为模块的一部分并随模型保存与加载，但不会被视为模型参数参与反向传播
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [seq_len, batch_size, d_model]  经过词向量的输入
        """
        c = self.pe[:, :x.size(1)].clone().detach()  # 经过词向量的输入与位置编码相加
        # Dropout层会按照设定的比例随机“丢弃”（置零）一部分位置编码与词向量相加后的元素，
        # 以此引入正则化效果，防止模型过拟合
        return c