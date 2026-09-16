import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torch.nn.init as init

import gpt.base_model as base_model
from gpt.base_model import *
from gpt.dataset import *


def _uses_protein(conditional):
    conditional = set(conditional)
    return bool({"protein", "pocket", "protein_pocket"} & conditional)


def get_attn_pad_mask(seq_q, seq_k):
    """Padding mask: [B,Lq,Lk], True means masked."""
    batch_size, len_q = seq_q.size()
    _, len_k = seq_k.size()
    pad_attn_mask = seq_k.eq(constant.PAD_TOKEN_ID).unsqueeze(1)
    return pad_attn_mask.expand(batch_size, len_q, len_k)


def get_attn_subsequence_mask(seq):
    """Causal mask: [B,L,L], True means masked."""
    seq_len = seq.size(1)
    mask = torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=seq.device),
        diagonal=1,
    )
    return mask.unsqueeze(0).expand(seq.size(0), -1, -1)


def get_key_padding_mask(protein_length, smiles, focus_seq=None):
    """
    Build molecule-query x protein-key mask.

    focus_seq follows the original code convention: True means the residue is
    masked in the pocket-focused channel, False means it is retained.
    """
    if protein_length is None:
        return None, None

    if not torch.is_tensor(protein_length):
        protein_length = torch.as_tensor(protein_length, device=smiles.device)
    protein_length = protein_length.to(smiles.device).long()

    batch_size = protein_length.size(0)
    max_protein_length = int(protein_length.max().item())
    len_q = smiles.size(1)

    residue_index = torch.arange(max_protein_length, device=smiles.device).view(1, 1, -1)
    length_mask = residue_index >= protein_length.view(batch_size, 1, 1)
    length_mask = length_mask.expand(batch_size, len_q, max_protein_length)

    query_pad = smiles.eq(constant.PAD_TOKEN_ID).unsqueeze(-1)
    key_padding_mask = length_mask | query_pad

    focus_mask = None
    if focus_seq is not None:
        if not torch.is_tensor(focus_seq):
            focus_seq = torch.as_tensor(focus_seq, device=smiles.device)
        focus_seq = focus_seq.to(smiles.device).bool()
        focus_mask = focus_seq.unsqueeze(1).expand(batch_size, len_q, -1)
        focus_mask = focus_mask | key_padding_mask

    return key_padding_mask, focus_mask


class DecoderLayer(nn.Module):
    def __init__(self, prop_len, p_type, conditional):
        super().__init__()
        self.p_type = p_type
        self.conditional = list(conditional)

        # Intrinsic molecular properties act only on molecular self-attention.
        self.dec_self_attn = MultiHeadAttention(
            d_model=d_model,
            d_k=d_k,
            d_v=d_v,
            n_heads=n_heads,
            p_type=p_type,
            conditional=self.conditional,
            attention_mode="self",
        )
        self.pos_ffn = PoswiseFeedForwardNet(
            d_model=d_model,
            d_ff=d_ff,
            p_type=p_type,
            conditional=self.conditional,
        )

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)

        # Extrinsic biological conditions use heterogeneous cross-attention.
        if "protein" in self.conditional:
            self.protein_encoder_conditional = nn.Sequential(
                nn.Linear(protein_emb_size, emb_size * 2),
                nn.GELU(),
                nn.Linear(emb_size * 2, emb_size),
            )
            self.protein_cross_attn_ln_conditional = nn.LayerNorm(emb_size)
            self.protein_cross_attn_conditional = MultiHeadAttention(
                d_model=d_model,
                d_k=d_k,
                d_v=d_v,
                n_heads=n_heads,
                p_type=p_type,
                conditional=self.conditional,
                attention_mode="cross",
            )
            self.protein_ffn_conditional = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(d_ff, emb_size),
            )

        elif "pocket" in self.conditional:
            self.pocket_encoder_conditional = nn.Sequential(
                nn.Linear(protein_emb_size, emb_size * 2),
                nn.GELU(),
                nn.Linear(emb_size * 2, emb_size),
            )
            self.pocket_cross_attn_ln_conditional = nn.LayerNorm(emb_size)
            self.pocket_cross_attn_conditional = MultiHeadAttention(
                d_model=d_model,
                d_k=d_k,
                d_v=d_v,
                n_heads=n_heads,
                p_type=p_type,
                conditional=self.conditional,
                attention_mode="cross",
            )
            self.pocket_ffn_conditional = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(d_ff, emb_size),
            )

        elif "protein_pocket" in self.conditional:
            self.protein_pocket_encoder_conditional = nn.Sequential(
                nn.Linear(protein_emb_size, emb_size * 2),
                nn.GELU(),
                nn.Linear(emb_size * 2, emb_size),
            )
            self.protein_pocket_cross_attn_conditional = DualChannelCrossAttention(
                d_model=d_model,
                d_k=d_k,
                d_v=d_v,
                n_heads=n_heads,
                p_type=p_type,
                conditional=self.conditional,
            )

    def forward(self, dec_inputs, dec_self_attn_mask, protein, protein_padding_mask, pocket_mask, prop):
        # Pre-LN molecular self-attention.
        x = dec_inputs
        x_norm = self.ln1(x)
        self_out, dec_self_attn = self.dec_self_attn(
            x_norm, x_norm, x_norm, dec_self_attn_mask, prop
        )
        x = x + self_out

        # Pre-LN property-conditioned FFN (FiLM modulation).
        x_norm = self.ln2(x)
        x = x + self.pos_ffn(x_norm, prop)
        x = self.ln3(x)

        # Cross-attention never uses a Q-K metric across heterogeneous spaces.
        if protein is not None and "protein" in self.conditional:
            protein_encoded = self.protein_encoder_conditional(protein)
            residual = x
            q = self.protein_cross_attn_ln_conditional(x)
            cross_output, _ = self.protein_cross_attn_conditional(
                q, protein_encoded, protein_encoded, protein_padding_mask, None
            )
            x = residual + cross_output
            x = x + self.protein_ffn_conditional(x)

        elif protein is not None and "pocket" in self.conditional:
            protein_encoded = self.pocket_encoder_conditional(protein)
            residual = x
            q = self.pocket_cross_attn_ln_conditional(x)
            # In pocket-only mode, use pocket_mask if available, otherwise the
            # normal protein padding mask.
            cross_mask = pocket_mask if pocket_mask is not None else protein_padding_mask
            cross_output, _ = self.pocket_cross_attn_conditional(
                q, protein_encoded, protein_encoded, cross_mask, None
            )
            x = residual + cross_output
            x = x + self.pocket_ffn_conditional(x)

        elif protein is not None and pocket_mask is not None and "protein_pocket" in self.conditional:
            protein_encoded = self.protein_pocket_encoder_conditional(protein)
            residual = x
            cross_output, _ = self.protein_pocket_cross_attn_conditional(
                x,
                protein_encoded,
                protein_encoded,
                protein_padding_mask,
                pocket_mask,
                None,
            )
            x = residual + cross_output

        return x, dec_self_attn


class Decoder(nn.Module):
    def __init__(self, vocab_size, prop_len, p_type, conditional):
        super().__init__()
        self.p_type = p_type
        self.conditional = list(conditional)
        self.tgt_emb = nn.Embedding(vocab_size, d_model).to(device)
        self.pos_emb = base_model.PositionalEncoding(d_model=d_model, max_len=max_pos).to(device)
        self.layers = nn.ModuleList(
            [
                DecoderLayer(prop_len, p_type=self.p_type, conditional=self.conditional).to(device)
                for _ in range(n_layers)
            ]
        )
        self.dropout = nn.Dropout(p=0.1)
        self.prop_len = prop_len

        # A single shared encoder is used by all transformer blocks.
        if "prop" in self.conditional:
            self.prop_encoder_conditional = base_model.PropertyEncoder(
                prop_len=prop_len,
                d_model=d_model,
                hidden_dim=max(64, min(d_model, 256)),
                dropout=0.1,
            )
        else:
            self.prop_encoder_conditional = None

    def forward(self, dec_inputs, protein, protein_length, pocket, prop):
        dec_outputs = self.tgt_emb(dec_inputs)
        dec_outputs = dec_outputs + self.pos_emb(dec_outputs)
        dec_outputs = self.dropout(dec_outputs)

        pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)
        causal_mask = get_attn_subsequence_mask(dec_inputs)
        final_mask = pad_mask | causal_mask

        prop_context = None
        if self.prop_encoder_conditional is not None and prop is not None:
            prop_context = self.prop_encoder_conditional(prop)

        if _uses_protein(self.conditional) and protein is not None:
            enc_key_padding_mask, pocket_mask = get_key_padding_mask(
                protein_length, dec_inputs, pocket
            )
        else:
            enc_key_padding_mask = None
            pocket_mask = None

        dec_self_attns = []
        for layer in self.layers:
            dec_outputs, dec_self_attn = layer(
                dec_outputs,
                final_mask,
                protein,
                enc_key_padding_mask,
                pocket_mask,
                prop_context,
            )
            dec_self_attns.append(dec_self_attn)

        return dec_outputs, dec_self_attns


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size=constant.vocab_size,
        prop_len=constant.prop_len,
        p_type="",
        conditional=("unconditional",),
    ):
        super().__init__()
        self.conditional = list(conditional)
        self.prop_len = prop_len
        self.register_buffer("property_mean", torch.zeros(prop_len))
        self.register_buffer("property_std", torch.ones(prop_len))
        self.register_buffer("property_stats_fitted", torch.tensor(False))
        self.decoder = Decoder(vocab_size, prop_len, p_type=p_type, conditional=self.conditional)
        self.projection = nn.Linear(d_model, vocab_size).to(device)

        # Auxiliary head strengthens continuous-property awareness without
        # changing GPT.forward's four return values.
        if "prop" in self.conditional:
            self.property_head = nn.Sequential(
                nn.Linear(d_model, max(64, d_model // 2)),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(max(64, d_model // 2), prop_len),
            )
        else:
            self.property_head = None

        self.property_loss_weight = float(getattr(constant, "property_loss_weight", 0.1))

        self.apply(self._init_weights)
        self._reset_identity_conditioning()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            init.kaiming_uniform_(m.weight, a=0, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def _reset_identity_conditioning(self):
        # The generic initialization above would otherwise overwrite the special
        # zero initialization required for smooth fine-tuning from an old GPT.
        for module in self.modules():
            if isinstance(module, base_model.PropertyAdaptiveScaledDotProductAttention):
                nn.init.zeros_(module.metric_generator[-1].weight)
                nn.init.zeros_(module.metric_generator[-1].bias)
            if isinstance(module, base_model.PoswiseFeedForwardNet) and getattr(module, "use_prop", False):
                nn.init.zeros_(module.film[-1].weight)
                nn.init.zeros_(module.film[-1].bias)
            if isinstance(module, base_model.MultiHeadAttention) and getattr(module, "use_low_rank_adapter", False):
                nn.init.zeros_(module.q_up.weight)
                nn.init.zeros_(module.k_up.weight)
                nn.init.zeros_(module.v_up.weight)

    def normalize_properties(self, prop):
        return (prop - self.property_mean) / self.property_std

    def forward(self, dec_inputs, protein, protein_length, pocket, prop):
        if prop is not None:
            prop = self.normalize_properties(prop)
        # Output signature is unchanged.
        dec_outputs, dec_self_attns = self.decoder(
            dec_inputs, protein, protein_length, pocket, prop
        )
        dec_logits = self.projection(dec_outputs)
        return (
            dec_logits.view(-1, dec_logits.size(-1)),
            dec_self_attns,
            dec_logits,
            dec_outputs,
        )

    def predict_properties(self, hidden_states, token_ids):
        """Predict molecule-level properties from masked mean-pooled hidden states."""
        if self.property_head is None:
            return None
        valid = token_ids.ne(constant.PAD_TOKEN_ID).unsqueeze(-1).to(hidden_states.dtype)
        denom = valid.sum(dim=1).clamp_min(1.0)
        pooled = (hidden_states * valid).sum(dim=1) / denom
        return self.property_head(pooled)

    def _generate(self, input_ids, protein, protein_length, pocket, prop,
                  method, arg, max_length):
        """Generate up to a total length, preserving the original batch order."""
        if input_ids.ndim != 2 or input_ids.size(1) == 0:
            raise ValueError("input_ids must be a nonempty [batch, length] tensor")
        capacity = self.decoder.pos_emb.pe.size(1)
        if input_ids.size(1) > capacity:
            raise ValueError("Prompt exceeds positional encoding capacity")
        if isinstance(max_length, bool) or int(max_length) != max_length or max_length < 1:
            raise ValueError("max_length must be a positive integer")
        limit = min(int(max_length), capacity)
        if method == "temperature" and (not math.isfinite(float(arg)) or float(arg) <= 0):
            raise ValueError("temperature must be finite and positive")
        if method == "top_k" and (isinstance(arg, bool) or int(arg) != arg or int(arg) < 1):
            raise ValueError("k must be a positive integer")
        result = [None] * input_ids.size(0)
        indices = torch.arange(input_ids.size(0), device=input_ids.device)
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                # An already terminated prompt must not receive extra tokens.
                active = ~input_ids[:, -1].eq(constant.EOS_TOKEN_ID)
                for idx in (~active).nonzero(as_tuple=True)[0].tolist():
                    result[idx] = input_ids[idx].cpu().tolist()
                conditions = [protein, protein_length, pocket, prop]
                conditions = [v[active] if v is not None else None for v in conditions]
                input_ids, indices = input_ids[active], indices[active]
                for _ in range(max(0, limit - input_ids.size(1))):
                    if not input_ids.size(0):
                        break
                    logits = self(input_ids, *conditions)[2][:, -1, :]
                    if method == "greedy":
                        token = logits.argmax(dim=-1, keepdim=True)
                    elif method == "temperature":
                        token = torch.multinomial(F.softmax(logits / float(arg), dim=-1), 1)
                    else:
                        values, choices = logits.topk(min(int(arg), logits.size(-1)), dim=-1)
                        token = choices.gather(-1, torch.multinomial(F.softmax(values, dim=-1), 1))
                    input_ids = torch.cat([input_ids, token], dim=-1)
                    finished = token[:, 0].eq(constant.EOS_TOKEN_ID)
                    for idx in finished.nonzero(as_tuple=True)[0].tolist():
                        result[indices[idx].item()] = input_ids[idx].cpu().tolist()
                    keep = ~finished
                    input_ids, indices = input_ids[keep], indices[keep]
                    conditions = [v[keep] if v is not None else None for v in conditions]
                for idx, row in zip(indices.tolist(), input_ids):
                    result[idx] = row.cpu().tolist()
        finally:
            self.train(was_training)
        return result

    def temperature_sampling(model, tokenizer, input_ids, protein, protein_length,
                             pocket, prop, temperature=0.8, max_length=constant.max_pos):
        return [tokenizer.decode(row) for row in model._generate(
            input_ids, protein, protein_length, pocket, prop,
            "temperature", temperature, max_length)]

    def top_k_sampling(model, tokenizer, input_ids, protein, protein_length,
                       prop, k=40, max_length=constant.max_pos, pocket=None):
        return [tokenizer.decode(row) for row in model._generate(
            input_ids, protein, protein_length, pocket, prop, "top_k", k, max_length)]

    def greedy_decoder(self, dec_input, protein, protein_length, pocket=None, prop=None):
        rows = self._generate(dec_input, protein, protein_length, pocket, prop,
                              "greedy", None, min(dec_input.size(1) + 101,
                                                  self.decoder.pos_emb.pe.size(1)))
        tensors = [torch.tensor(row, dtype=dec_input.dtype, device=dec_input.device) for row in rows]
        if not tensors:
            return dec_input
        return nn.utils.rnn.pad_sequence(tensors, batch_first=True,
                                         padding_value=constant.PAD_TOKEN_ID)

    def answer(
        self,
        sentence,
        protein,
        tokenizer,
        prop=None,
        method="greedy",
        batch=1,
        model="t12_35M",
        pocket=None,
        arg=None,
    ):
        dec_input = tokenizer.encode(sentence).ids
        dec_input = torch.tensor(dec_input, dtype=torch.long, device=device).repeat(batch, 1)

        if protein is not None:
            if model == "t12_35M":
                param = ESM.load_model(model)
                protein_rep = ESM.protein2vector([(protein, "")], param=param)
                protein = protein_rep[2]
                protein_length = protein_rep[1]
            elif model == "rACSF":
                from gpt.rACSF import cal_rACSF

                protein_rep = cal_rACSF(protein)[0]
                protein = protein_rep
                protein_length = len(protein_rep)
            else:
                protein_length = len(protein)

            # Keep the original external pocket convention: pocket contains
            # 1-based residue indices to retain in the local channel.
            if pocket is not None:
                pocket_mask = [i + 1 not in pocket for i in range(protein_length)]
            else:
                pocket_mask = [False for _ in range(protein_length)]

            protein = torch.tensor(protein, device=device, dtype=torch.float)
            protein = protein.unsqueeze(0).repeat(batch, 1, 1)
            protein_length = torch.tensor(protein_length, dtype=torch.long, device=device).repeat(batch)
            pocket = torch.tensor(pocket_mask, device=device, dtype=torch.bool).unsqueeze(0).repeat(batch, 1)
        else:
            protein_length = None
            pocket = None

        if prop is not None:
            prop = torch.as_tensor(prop, dtype=torch.float, device=device)
            if prop.dim() == 1:
                prop = prop.unsqueeze(0)
            if prop.size(0) == 1 and batch > 1:
                prop = prop.repeat(batch, 1)

        if method == "greedy":
            # Preserve the original single-string greedy return convention.
            output = self.greedy_decoder(
                dec_input[:1],
                protein=protein[:1] if protein is not None else None,
                protein_length=protein_length[:1] if protein_length is not None else None,
                pocket=pocket[:1] if pocket is not None else None,
                prop=prop[:1] if prop is not None else None,
            ).squeeze(0).cpu().tolist()
            return tokenizer.decode(output)
        elif method == "temperature":
            return self.temperature_sampling(
                input_ids=dec_input,
                tokenizer=tokenizer,
                protein=protein,
                protein_length=protein_length,
                prop=prop,
                temperature=0.8 if arg is None else arg,
                pocket=pocket,
            )
        elif method == "top_k":
            return self.top_k_sampling(
                k=40 if arg is None else arg,
                input_ids=dec_input,
                tokenizer=tokenizer,
                protein=protein,
                protein_length=protein_length,
                prop=prop,
                pocket=pocket,
            )
        return ""


def _prepare_batch_conditionals(protein, protein_len, pocket, props, conditional):
    if not _uses_protein(conditional):
        protein = None
        protein_len = None
        pocket = None
    if "prop" not in conditional:
        props = None

    if protein is not None:
        protein = torch.as_tensor(protein, dtype=torch.float, device=device)
    if protein_len is not None:
        protein_len = torch.as_tensor(protein_len, dtype=torch.long, device=device)
    if pocket is not None:
        pocket = torch.as_tensor(pocket, dtype=torch.bool, device=device)
    if props is not None:
        props = torch.as_tensor(props, dtype=torch.float, device=device)
    return protein, protein_len, pocket, props


def _compute_loss(model, criterion, outputs, dec_outputs, dec_inputs, props):
    token_loss = criterion(outputs[0], dec_outputs.view(-1))
    if props is None or model.property_head is None or model.property_loss_weight == 0:
        return token_loss, None

    # Structure-only pass: target properties cannot be copied through conditioning.
    hidden, _ = model.decoder(dec_inputs, None, None, None, None)
    pred_prop = model.predict_properties(hidden, dec_inputs)
    prop_loss = F.mse_loss(pred_prop, model.normalize_properties(props))
    total_loss = token_loss + model.property_loss_weight * prop_loss
    return total_loss, prop_loss


def _fit_property_statistics(model, loaders):
    """Fit only on training datasets, without padding or changing stored samples."""
    if model.property_head is None or bool(model.property_stats_fitted):
        return
    count = 0
    mean = torch.zeros(model.prop_len, dtype=torch.float64)
    m2 = torch.zeros_like(mean)
    for loader in loaders:
        if loader is None:
            continue
        for sample in loader.dataset:
            values = sample['props']
            if values is None:
                raise ValueError('Property-conditioned training requires properties for every sample')
            values = torch.as_tensor(values, dtype=torch.float64)
            if values.shape != mean.shape or not torch.isfinite(values).all():
                raise ValueError('Invalid training properties')
            count += 1
            delta = values - mean
            mean += delta / count
            m2 += delta * (values - mean)
    if not count:
        raise ValueError('No training properties available')
    std = (m2 / count).clamp_min(0).sqrt()
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    model.property_mean.copy_(mean)
    model.property_std.copy_(std)
    model.property_stats_fitted.fill_(True)


class _LossTotals:
    def __init__(self, weight):
        self.weight = weight
        self.token_sum = self.prop_sum = 0.0
        self.tokens = self.elements = 0

    def add(self, loss, prop_loss, targets, props):
        prop_value = 0.0 if prop_loss is None else prop_loss.item()
        token_value = loss.item() - self.weight * prop_value
        count = targets.ne(constant.PAD_TOKEN_ID).sum().item()
        self.token_sum += token_value * count
        self.tokens += count
        if prop_loss is not None:
            self.prop_sum += prop_value * props.numel()
            self.elements += props.numel()

    def result(self):
        token = self.token_sum / max(self.tokens, 1)
        prop = self.prop_sum / max(self.elements, 1)
        return dict(loss=token + self.weight * prop, token_loss=token, prop_loss=prop)


def train_step(model, data_loader, optimizer, criterion, clip=1, print_every=None, vs=0, conditional=("unconditional",)):
    if data_loader is None:
        return -1
    model.train()
    totals = _LossTotals(model.property_loss_weight)

    progress = tqdm(data_loader, desc="Train")
    for i, (dec_inputs, dec_outputs, protein, protein_len, pocket, props) in enumerate(progress):
        optimizer.zero_grad(set_to_none=True)
        dec_inputs = torch.as_tensor(dec_inputs, dtype=torch.long, device=device)
        dec_outputs = torch.as_tensor(dec_outputs, dtype=torch.long, device=device)
        protein, protein_len, pocket, props = _prepare_batch_conditionals(
            protein, protein_len, pocket, props, conditional
        )

        model_outputs = model(dec_inputs, protein, protein_len, pocket, props)
        loss, prop_loss = _compute_loss(model, criterion, model_outputs, dec_outputs, dec_inputs, props)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        totals.add(loss, prop_loss, dec_outputs, props)
        aggregate = totals.result()
        metrics = {"loss": f"{loss.item():.4f}", "avg_loss": f"{aggregate['loss']:.4f}",
                   "token_loss": f"{aggregate['token_loss']:.4f}"}
        if prop_loss is not None:
            metrics["prop_loss"] = f"{aggregate['prop_loss']:.4f}"
        progress.set_postfix(metrics)

    return totals.result()['loss']


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - elapsed_mins * 60)
    return elapsed_mins, elapsed_secs


def _evaluate(model, data_iterable, criterion, conditional, return_metrics=False):
    if data_iterable is None:
        return None
    model.eval()
    totals = _LossTotals(model.property_loss_weight)
    with torch.no_grad():
        for datas in data_iterable:
            if datas is None:
                continue
            for dec_inputs, dec_outputs, protein, protein_len, pocket, props in tqdm(datas):
                dec_inputs = torch.as_tensor(dec_inputs, dtype=torch.long, device=device)
                dec_outputs = torch.as_tensor(dec_outputs, dtype=torch.long, device=device)
                protein, protein_len, pocket, props = _prepare_batch_conditionals(
                    protein, protein_len, pocket, props, conditional
                )
                model_outputs = model(dec_inputs, protein, protein_len, pocket, props)
                loss, prop_loss = _compute_loss(model, criterion, model_outputs, dec_outputs, dec_inputs, props)
                totals.add(loss, prop_loss, dec_outputs, props)
    if not totals.tokens:
        return None
    result = totals.result()
    return result if return_metrics else result["loss"]


def evaluate_property_condition_usage(model, data_iterable, conditional=("prop",)):
    """Compare matched vs cyclically shuffled properties; positive delta indicates use.

    This diagnostic measures teacher-forced sensitivity, not generated-molecule accuracy.
    Singleton batches are excluded because they cannot be shuffled.
    """
    if "prop" not in conditional:
        raise ValueError("Property conditioning is required")
    was_training = model.training
    model.eval()
    matched = shuffled = 0.0
    tokens = 0
    try:
        with torch.no_grad():
            for loader in data_iterable:
                if loader is None:
                    continue
                for inputs, targets, protein, lengths, pocket, props in loader:
                    inputs = torch.as_tensor(inputs, dtype=torch.long, device=device)
                    targets = torch.as_tensor(targets, dtype=torch.long, device=device)
                    if inputs.size(0) < 2:
                        continue
                    protein, lengths, pocket, props = _prepare_batch_conditionals(
                        protein, lengths, pocket, props, conditional)
                    if props is None:
                        raise ValueError("Properties are missing")
                    for is_shuffled, values in [(False, props), (True, props.roll(1, 0))]:
                        logits = model(inputs, protein, lengths, pocket, values)[0]
                        loss = F.cross_entropy(logits, targets.reshape(-1),
                                               ignore_index=constant.PAD_TOKEN_ID, reduction="sum").item()
                        if is_shuffled:
                            shuffled += loss
                        else:
                            matched += loss
                    tokens += targets.ne(constant.PAD_TOKEN_ID).sum().item()
    finally:
        model.train(was_training)
    if not tokens:
        raise ValueError("No usable multi-sample batches")
    return dict(matched_token_loss=matched / tokens, shuffled_token_loss=shuffled / tokens,
                property_shuffle_delta=(shuffled - matched) / tokens)


def train(data_loader, epochs, vs, lr, model=None, p_type="", conditional=("unconditional",), save_name="GPT.pt",
          output_dir=None):
    from pathlib import Path
    output_dir = Path(output_dir) if output_dir is not None else Path('checkpoints/fragGPT')
    output_dir.mkdir(parents=True, exist_ok=True)
    if model is None:
        model = GPT(vocab_size=vs, prop_len=prop_len, p_type=p_type, conditional=conditional)
        try:
            state = torch.load("checkpoints/fragGPT/GPT.pt", map_location=device)
            model.load_state_dict(state, strict=False)
        except Exception:
            pass

    params = list(model.parameters())
    total_params = sum(p.numel() for p in params)
    trainable_params = sum(p.numel() for p in params if p.requires_grad)
    print(f"总参数数量: {total_params}")
    print(f"可训练参数数量: {trainable_params}")
    print(model)

    # Preserve the original unconditional behavior: conditional-only modules do
    # not contribute when no condition is requested.  No manual freezing is
    # required because they are not on the forward path.

    from datetime import datetime

    model.to(device)
    _fit_property_statistics(model, data_loader[0])
    criterion = nn.CrossEntropyLoss(ignore_index=constant.PAD_TOKEN_ID).to(device)
    optimizer = optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr)
    X, Y = [], []
    last_loss = math.inf
    saved_best = False

    for train_data in data_loader[0]:
        for epoch in range(epochs):
            X.append(epoch)
            start_time = time.time()
            train_loss = train_step(
                model,
                train_data,
                optimizer,
                criterion,
                CLIP,
                conditional=conditional,
            )
            Y.append(train_loss)
            end_time = time.time()

            valid_metrics = _evaluate(model, data_loader[1], criterion, conditional, return_metrics=True)
            selection_loss = train_loss if valid_metrics is None else valid_metrics['loss']
            if math.isfinite(selection_loss) and selection_loss < last_loss:
                torch.save(model.state_dict(), output_dir / save_name)
                last_loss = selection_loss
                saved_best = True
            if valid_metrics is not None:
                print(f"valid metrics:{valid_metrics}")

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            print(f"Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s")
            print(f"\tTrain Loss: {train_loss:.3f}")

    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    np.save(output_dir / (now + "_X.npy"), X)
    np.save(output_dir / (now + "_Y.npy"), Y)

    if not saved_best:
        raise RuntimeError("Training did not produce a finite best checkpoint")
    model.load_state_dict(torch.load(output_dir / save_name, map_location=device, weights_only=True))
    test_loss = _evaluate(model, data_loader[2], criterion, conditional, return_metrics=True)
    if test_loss is not None:
        print(f"test loss:{test_loss}")

    return model


def fine_tune(model_path, data_loader, epochs, vs, lr, p_type="", conditional=("prop",), save_name="GPT.pt"):
    model = GPT(vocab_size=vs, prop_len=prop_len, p_type=p_type, conditional=conditional)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    return train(
        data_loader,
        epochs,
        vs,
        lr,
        model,
        p_type=p_type,
        conditional=conditional,
        save_name=save_name,
    )


def fine_tune_fragGPT(epoch, model_path, plk_path, tokenizer_path):
    data_loaders, tokenizer__ = get_PL_dataloader(plk_path=plk_path, tokenizer_path=tokenizer_path)
    lr = 5e-5
    model = GPT(vocab_size=tokenizer__.get_vocab_size())
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    return train(((data_loaders,), None, data_loaders), epoch, tokenizer__.get_vocab_size(), lr, model=model)


def fine_tune_fragGPT_molonly(epoch, model_path, tokenizer_path, train_path="gpt/finetune.txt"):
    tokenizer__ = tokenizer.tokenizer_from_file(file_path=tokenizer_path)
    smiles = []
    with open(train_path, "r") as file_:
        line = file_.readline().strip("\n")
        while line:
            smiles.append(line)
            line = file_.readline().strip("\n")

    # The user indicated that the decomposition/SISR construction pipeline has
    # already been upgraded externally.  This call is intentionally left in its
    # original place so the model keeps the same data input contract.
    dataprocess.mol_decomp_mp_(smiles, n_core=60)
    data_loaders = get_frag_dataloader_without_split(
        frag_token_fun_1,
        tokenizer__,
        train_file="gpt/frag_decom_test_other.txt",
        test_file="",
        batch_size=30,
        multiset=1,
    )

    lr = 5e-5
    model = GPT(vocab_size=tokenizer__.get_vocab_size())
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)

    params = list(model.parameters())
    total_params = sum(p.numel() for p in params)
    print(f"总参数数量: {total_params}")
    return train(data_loaders, epoch, tokenizer__.get_vocab_size(), lr, model=model)


def train_PL_fragGPT(epoch, plk_path):
    data_loaders, tokenizer__ = get_PL_dataloader(plk_path=plk_path)
    lr = 4e-4
    return train(((data_loaders,), None, data_loaders), epoch, tokenizer__.get_vocab_size(), lr)


if __name__ == "__main__":
    print()
