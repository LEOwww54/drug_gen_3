import math
import numpy as np
import torch
import torch.nn as nn
import constant


class ScaledDotProductAttention(nn.Module):
    """Standard scaled dot-product attention used for heterogeneous cross-attention."""

    def __init__(self, d_k, dropout=0.0):
        super().__init__()
        self.d_k = d_k
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, attn_mask=None):
        # Q: [B, H, Lq, Dk], K: [B, H, Lk, Dk], V: [B, H, Lk, Dv]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.d_k)
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask.bool(), torch.finfo(scores.dtype).min)

        attn = torch.softmax(scores, dim=-1)
        if attn_mask is not None:
            # Fully masked rows must contribute zero, including in low precision.
            attn = attn.masked_fill(attn_mask.bool(), 0.0)
        attn = self.dropout(attn)
        context = torch.matmul(attn, V)
        return context, attn


class PropertyEncoder(nn.Module):
    """
    Shared continuous-property encoder.

    Input:  [B, prop_len]
    Output: [B, d_model]

    Important: SA/QED/LogP and other continuous properties should preferably be
    standardized in the data pipeline using training-set statistics.  We do not
    apply per-sample LayerNorm to raw properties because that would erase useful
    absolute-value information.
    """

    def __init__(self, prop_len, d_model, hidden_dim=None, dropout=0.1):
        super().__init__()
        hidden_dim = hidden_dim or max(64, min(d_model, 256))
        self.net = nn.Sequential(
            nn.Linear(prop_len, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, prop):
        if prop is None:
            return None
        prop = torch.nan_to_num(prop.float(), nan=0.0, posinf=0.0, neginf=0.0)
        return self.net(prop)


class PropertyAdaptiveScaledDotProductAttention(nn.Module):
    """
    Lightweight property-adaptive metric attention (PAMA).

    Instead of constructing a full dynamic metric G in R^{Dk x Dk}, the method
    learns one positive diagonal metric for every attention head:

        D_h(c) = diag(1 + alpha * tanh(f_h(c)))
        score_ij^h = q_i^T D_h(c) k_j / sqrt(Dk)

    This keeps the dynamic condition tensor at [B, H, Dk], avoids the expensive
    [B, H, Dk, Dk] metric, and never materializes [B, H, L, L, Dk] pairwise
    displacement tensors.
    """

    def __init__(
        self,
        d_k=constant.d_k,
        n_heads=constant.n_heads,
        prop_context_dim=None,
        alpha=0.5,
        dropout=0.0,
    ):
        super().__init__()
        self.d_k = d_k
        self.n_heads = n_heads
        self.alpha = float(alpha)
        if not 0.0 <= self.alpha < 1.0:
            raise ValueError("alpha must satisfy 0 <= alpha < 1")
        prop_context_dim = prop_context_dim or getattr(constant, "d_model", d_k * n_heads)

        hidden = max(64, min(prop_context_dim, 256))
        self.metric_generator = nn.Sequential(
            nn.Linear(prop_context_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_heads * d_k),
        )
        # Zero initialization makes D(c) ~= I at the start of conditional fine-tuning.
        nn.init.zeros_(self.metric_generator[-1].weight)
        nn.init.zeros_(self.metric_generator[-1].bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, attn_mask=None, prop_context=None):
        if prop_context is None:
            scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.d_k)
            metric_diag = None
        else:
            batch_size = Q.size(0)
            raw = self.metric_generator(prop_context)
            raw = raw.view(batch_size, self.n_heads, 1, self.d_k)
            metric_diag = 1.0 + self.alpha * torch.tanh(raw)
            # q^T D(c) k.  D(c) is strictly positive for alpha < 1.
            Q_metric = Q * metric_diag
            scores = torch.matmul(Q_metric, K.transpose(-1, -2)) / math.sqrt(self.d_k)

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask.bool(), torch.finfo(scores.dtype).min)

        attn = torch.softmax(scores, dim=-1)
        if attn_mask is not None:
            # Fully masked rows must contribute zero, including in low precision.
            attn = attn.masked_fill(attn_mask.bool(), 0.0)
        attn = self.dropout(attn)
        context = torch.matmul(attn, V)
        return context, attn, metric_diag


class RiemannianScaledDotProductAttention(PropertyAdaptiveScaledDotProductAttention):
    """
    Backward-compatible class name.

    The old full-matrix implementation was memory intensive and mixed metric
    similarity with distance terminology.  This class now implements the
    lightweight diagonal property-adaptive metric attention used by the upgraded
    model.  Existing code using p_type='Riemannian' therefore remains runnable.
    """

    def __init__(self, d_k=constant.d_k, n_heads=constant.n_heads, **kwargs):
        super().__init__(d_k=d_k, n_heads=n_heads, **kwargs)

    def forward(self, Q, K, V, attn_mask, v_cond):
        return super().forward(Q, K, V, attn_mask, v_cond)


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention with a strict separation between:

    1) molecular self-attention: optional property-adaptive diagonal metric;
    2) heterogeneous cross-attention: standard dot-product compatibility.

    The input/output signature is kept compatible with the original code.
    """

    def __init__(
        self,
        d_model,
        d_k,
        d_v,
        n_heads,
        p_type="",
        conditional=("unconditional",),
        attention_mode="self",
    ):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.p_type = p_type
        self.conditional = list(conditional)
        self.attention_mode = attention_mode
        self.standard_attention = ScaledDotProductAttention(d_k, dropout=0.1)

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.dropout = nn.Dropout(0.1)

        # p_type='Riemannian' is preserved as a compatibility alias for PAMA.
        metric_aliases = {"", "Riemannian", "riemannian", "PAMA", "pama", "metric", "property_metric"}
        self.use_property_metric = (
            self.attention_mode == "self"
            and "prop" in self.conditional
            and self.p_type in metric_aliases
        )

        if self.use_property_metric:
            self.property_attention = PropertyAdaptiveScaledDotProductAttention(
                d_k=d_k,
                n_heads=n_heads,
                prop_context_dim=d_model,
                alpha=0.5,
                dropout=0.1,
            )

        # Optional low-rank property adaptation kept for old p_type='lora_1' experiments.
        self.use_low_rank_adapter = (
            self.attention_mode == "self"
            and "prop" in self.conditional
            and self.p_type == "lora_1"
        )
        if self.use_low_rank_adapter:
            rank = min(16, max(4, d_model // 64))
            self.prop_gate = nn.Sequential(nn.Linear(d_model, n_heads), nn.Tanh())
            self.q_down = nn.Linear(d_model, rank, bias=False)
            self.q_up = nn.Linear(rank, d_k * n_heads, bias=False)
            self.k_down = nn.Linear(d_model, rank, bias=False)
            self.k_up = nn.Linear(rank, d_k * n_heads, bias=False)
            self.v_down = nn.Linear(d_model, rank, bias=False)
            self.v_up = nn.Linear(rank, d_v * n_heads, bias=False)
            nn.init.zeros_(self.q_up.weight)
            nn.init.zeros_(self.k_up.weight)
            nn.init.zeros_(self.v_up.weight)

    def _expand_mask(self, attn_mask):
        if attn_mask is None:
            return None
        if attn_mask.dim() == 3:
            return attn_mask.unsqueeze(1).expand(-1, self.n_heads, -1, -1)
        if attn_mask.dim() == 4:
            return attn_mask
        raise ValueError(f"Unsupported attention mask shape: {attn_mask.shape}")

    def forward(self, input_Q, input_K, input_V, attn_mask, y=None):
        batch_size = input_Q.size(0)

        Q = self.W_Q(input_Q)
        K = self.W_K(input_K)
        V = self.W_V(input_V)

        if self.use_low_rank_adapter and y is not None:
            # y is the shared [B, d_model] property context.
            gate = self.prop_gate(y).unsqueeze(-1).unsqueeze(2)  # [B,H,1,1]
            q_delta = self.q_up(self.q_down(input_Q)).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
            k_delta = self.k_up(self.k_down(input_K)).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
            v_delta = self.v_up(self.v_down(input_V)).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)
        else:
            gate = None
            q_delta = k_delta = v_delta = None

        Q = Q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        if gate is not None:
            Q = Q + gate * q_delta
            K = K + gate * k_delta
            V = V + gate * v_delta

        attn_mask = self._expand_mask(attn_mask)

        if self.use_property_metric and y is not None:
            context, attn, _ = self.property_attention(Q, K, V, attn_mask, y)
        else:
            # Cross-attention always comes through this path, even when property
            # conditioning is enabled elsewhere in the model.
            context, attn = self.standard_attention(Q, K, V, attn_mask)

        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v)
        output = self.fc(context)
        output = self.dropout(output)
        return output, attn


class DualChannelCrossAttention(nn.Module):
    """Global protein attention + pocket-focused attention with gated fusion."""

    def __init__(self, d_model, d_k, d_v, n_heads, p_type="", conditional=("protein", "protein_pocket")):
        super().__init__()
        self.global_attn_ln = nn.LayerNorm(d_model)
        self.local_attn_ln = nn.LayerNorm(d_model)

        # Cross-space attention deliberately does NOT use the molecular property metric.
        self.global_attention = MultiHeadAttention(
            d_model=d_model,
            d_k=d_k,
            d_v=d_v,
            n_heads=n_heads,
            p_type=p_type,
            conditional=conditional,
            attention_mode="cross",
        )
        self.local_attention = MultiHeadAttention(
            d_model=d_model,
            d_k=d_k,
            d_v=d_v,
            n_heads=n_heads,
            p_type=p_type,
            conditional=conditional,
            attention_mode="cross",
        )
        self.fusion_gate = nn.Linear(d_model * 2, d_model)

    def forward(self, query, key, value, attn_mask, focus_mask, y=None):
        global_out, _ = self.global_attention(query, key, value, attn_mask, None)
        global_out = self.global_attn_ln(global_out)

        local_out, _ = self.local_attention(query, key, value, focus_mask, None)
        local_out = self.local_attn_ln(local_out)

        combined = torch.cat([global_out, local_out], dim=-1)
        gate = torch.sigmoid(self.fusion_gate(combined))
        output = gate * global_out + (1.0 - gate) * local_out
        return output, None


class PoswiseFeedForwardNet(nn.Module):
    """
    Standard FFN plus lightweight property FiLM modulation.

    This replaces the previous full d_model x d_ff condition matrices and therefore
    greatly reduces conditional-branch parameters and memory use.
    """

    def __init__(self, d_model, d_ff, p_type="", conditional=("unconditional",)):
        super().__init__()
        self.W1 = nn.Linear(d_model, d_ff)
        self.W2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(0.1)
        self.activation = nn.GELU()
        self.conditional = list(conditional)
        self.use_prop = "prop" in self.conditional

        if self.use_prop:
            hidden = max(64, min(d_model, 256))
            self.film = nn.Sequential(
                nn.Linear(d_model, hidden),
                nn.GELU(),
                nn.Linear(hidden, d_model * 2),
            )
            # Start from the original unconditional FFN.
            nn.init.zeros_(self.film[-1].weight)
            nn.init.zeros_(self.film[-1].bias)

    def forward(self, inputs, y=None):
        h = self.W1(inputs)
        h = self.activation(h)
        h = self.dropout(h)
        output = self.W2(h)
        output = self.dropout(output)

        if self.use_prop and y is not None:
            gamma, beta = self.film(y).chunk(2, dim=-1)
            gamma = 0.1 * torch.tanh(gamma)
            beta = 0.1 * torch.tanh(beta)
            output = output * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)

        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # Input convention is [B, L, D].  Keep original behavior: return only PE.
        if x.size(1) > self.pe.size(1):
            raise ValueError(f"Sequence length {x.size(1)} exceeds positional capacity {self.pe.size(1)}")
        return self.pe[:, : x.size(1)].to(dtype=x.dtype, device=x.device)
