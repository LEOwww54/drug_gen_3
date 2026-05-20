import json
import math

import torch
import torch.utils.data as Data
# from adapters.methods import lora
from torch import nn, optim
import numpy as np
import time
from tqdm import tqdm
import torch.nn.functional as F

import base_model
import constant
import dataprocess
# import token_process
from base_model import *
from constant import  *

from gpt.dataset import *
import gpt.ESM as ESM

def get_attn_pad_mask(seq_q, seq_k):
    '''
    seq_q: [batch_size, seq_len]
    seq_k: [batch_size, seq_len]
    seq_len could be src_len or it could be tgt_len
    seq_len in seq_q and seq_len in seq_k maybe not equal
    '''
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(constant.PAD_TOKEN_ID).unsqueeze(1)  # [batch_size, 1, len_k], True is masked
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]


def get_attn_subsequence_mask(seq):
    '''
    seq: [batch_size, tgt_len]
    '''
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)  # Upper triangular matrix
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    subsequence_mask = subsequence_mask.to(device)
    return subsequence_mask  # [batch_size, tgt_len, tgt_len]

def get_key_padding_mask(protein_length, smiles, focus_seq = None):
    if protein_length is None:
        return None
    batch_size = protein_length.size(0)
    max_protein_length = protein_length.max()

    key_padding_mask = smiles.eq(constant.PAD_TOKEN_ID)
    key_padding_mask = key_padding_mask.unsqueeze(-1).repeat(1, 1, max_protein_length)

    if focus_seq is not None:
        focus_mask = focus_seq.unsqueeze(1).repeat(1, key_padding_mask.size(1), 1).to(device)
        focus_mask = focus_mask | key_padding_mask
        focus_mask.to(device)
    else:
        focus_mask = None

    for i in range(batch_size):
        key_padding_mask[i,:, protein_length[i]:] = True

    return key_padding_mask.to(device), focus_mask


class DecoderLayer(nn.Module):
    def __init__(self, prop_len, p_type, conditional):
        super(DecoderLayer, self).__init__()
        self.p_type = p_type
        self.conditional = conditional
        self.dec_self_attn = MultiHeadAttention(d_model=d_model, d_k=d_k, d_v=d_v, n_heads=n_heads, p_type=p_type, conditional = conditional)
        # self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet(d_model=d_model,d_ff=d_ff,p_type=p_type, conditional = conditional)

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)

        if 'protein' in conditional and ('protein_pocket' not in conditional):
            self.protein_encoder_conditional = nn.Sequential(
                nn.Linear(protein_emb_size, emb_size * 2),
                nn.GELU(),
                nn.Linear(emb_size * 2, emb_size)
            )
            self.protein_cross_attn_ln_conditional = nn.LayerNorm(emb_size)
            self.protein_cross_attn_conditional = MultiHeadAttention(d_model=d_model, d_k=d_k, d_v=d_v, n_heads=n_heads, p_type=p_type, conditional = conditional)
            self.protein_ffn_conditional = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Linear(d_ff, emb_size)
            )

        elif 'protein_pocket' in conditional:
            self.protein_pocket_encoder_conditional = nn.Sequential(
                nn.Linear(protein_emb_size, emb_size * 2),
                nn.GELU(),
                nn.Linear(emb_size * 2, emb_size)
            )

            self.protein_pocket_cross_attn_conditional = DualChannelCrossAttention(d_model=d_model, d_k=d_k, d_v=d_v, n_heads=n_heads,
                                                                     p_type=p_type, conditional=conditional)


        # self.prop_condition_gate = nn.Linear(emb_size * 2, emb_size)
        return

    def forward(self, dec_inputs, dec_self_attn_mask, protein, protein_padding_mask, pocket_mask, prop):
        '''
        dec_inputs: [batch_size, tgt_len, d_model]
        dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
        '''
        # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]

        dec_inputs = self.ln1(dec_inputs)
        residual_cross = dec_inputs
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask, prop)
        dec_outputs = dec_outputs + residual_cross

        dec_outputs = self.ln2(dec_outputs)
        residual_ffn = dec_outputs
        dec_outputs = self.pos_ffn(dec_outputs, prop)  # [batch_size, tgt_len, d_model]
        dec_outputs = dec_outputs + residual_ffn
        dec_outputs = self.ln3(dec_outputs)

        if protein is not None and 'protein' in self.conditional and ('protein_pocket' not in self.conditional):
            protein = self.protein_encoder_conditional(protein)
            # residual_cross = dec_outputs
            residual_cross = dec_outputs
            #x_cross = self.protein_cross_attn_ln(dec_outputs)
            x_cross = self.protein_cross_attn_ln_conditional(dec_outputs)

            # 应用Cross-Attention
            cross_output, _ = self.protein_cross_attn_conditional(
                x_cross,
                protein,
                protein,
                protein_padding_mask
            )

            # gate_input = torch.cat([x_cross, cross_output], dim=-1)
            # gate = torch.sigmoid(self.protein_condition_gate(gate_input))
            # conditioned = residual_cross + gate * cross_output
            conditioned = residual_cross + cross_output
            dec_outputs = conditioned

            residual = dec_outputs
            dec_outputs = self.protein_ffn_conditional(dec_outputs)
            dec_outputs = dec_outputs + residual
        elif 'protein_pocket' in self.conditional:
            protein = self.protein_pocket_encoder_conditional(protein)
            residual_cross = dec_outputs

            cross_output, _ = self.protein_pocket_cross_attn_conditional(
                dec_outputs,
                protein,
                protein,
                protein_padding_mask,
                pocket_mask
            )

            conditioned = residual_cross + cross_output
            dec_outputs = conditioned

        return dec_outputs, dec_self_attn

        if protein is not None:
            protein = self.protein_encoder(protein)
            # residual_cross = dec_outputs
            residual_cross = dec_outputs
            #x_cross = self.protein_cross_attn_ln(dec_outputs)
            x_cross = self.protein_cross_attn_ln(dec_outputs)

            # 应用Cross-Attention
            cross_output, _ = self.protein_cross_attn(
                x_cross,
                protein,
                protein,
                protein_padding_mask
            )

            # gate_input = torch.cat([x_cross, cross_output], dim=-1)
            # gate = torch.sigmoid(self.protein_condition_gate(gate_input))
            # conditioned = residual_cross + gate * cross_output
            conditioned = residual_cross + cross_output
            dec_outputs = conditioned

        if prop is not None:
            prop = self.prop_encoder(prop)
            # residual_cross = dec_outputs
            residual_cross = dec_outputs
            # x_cross = self.prop_cross_attn_ln(dec_outputs)
            x_cross = self.prop_cross_attn_ln(dec_outputs)

            cross_output, _ = self.prop_cross_attn(
                prop,
                x_cross,
                x_cross,
                None
            )

            conditioned = residual_cross + cross_output
            dec_outputs = conditioned

        dec_outputs = self.final_proj(dec_outputs)

        return dec_outputs, dec_self_attn


class Decoder(nn.Module):
    def __init__(self, vocab_size, prop_len, p_type, conditional):
        super(Decoder, self).__init__()
        self.p_type = p_type
        self.conditional = conditional
        self.tgt_emb = nn.Embedding(vocab_size, d_model).to(device)
        self.pos_emb = base_model.PositionalEncoding(d_model=d_model, max_len=max_pos).to(device)
        # self.pos_emb = nn.Embedding(max_pos, d_model).to(device)
        self.layers = nn.ModuleList([DecoderLayer(prop_len, p_type=self.p_type, conditional=self.conditional).to(device) for _ in range(n_layers)])
        self.dropout = nn.Dropout(p=0.1)
        self.prop_len = prop_len


        if 'prop' in conditional:
            self.prop_emb_conditional = nn.Sequential(
                    nn.Linear(prop_len, prop_len),
                )

    def forward(self, dec_inputs, protein, protein_length, pocket, prop):
        '''
        dec_inputs: [batch_size, tgt_len]
        '''
        dec_outputs = self.tgt_emb(dec_inputs)  # [batch_size, tgt_len, d_model]
        dec_outputs = dec_outputs + self.pos_emb(dec_outputs)
        dec_outputs = self.dropout(dec_outputs)

        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)  # [batch_size, tgt_len, tgt_len]
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs)  # [batch_size, tgt_len, tgt_len]
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequence_mask),
                                      0)  # [batch_size, tgt_len, tgt_len]

        # K = dec_outputs.clone()
        final_mask = dec_self_attn_mask.clone()

        if 'prop' in self.conditional and prop is not None:
            prop = self.prop_emb_conditional(prop)

        if 'protein' in self.conditional and protein is not None:
            enc_key_padding_mask, pocket_mask = get_key_padding_mask(protein_length, dec_inputs, pocket)
        else:
            enc_key_padding_mask = None
            pocket_mask = None

        dec_self_attns = []
        for layer in self.layers:
            # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
            dec_outputs, dec_self_attn = layer(dec_outputs, final_mask, protein, enc_key_padding_mask, pocket_mask, prop)
            dec_self_attns.append(dec_self_attn)

        return dec_outputs, dec_self_attns


class GPT(nn.Module):
    def __init__(self, vocab_size=constant.vocab_size, prop_len=constant.prop_len, p_type = '', conditional=['unconditional']):
        super(GPT, self).__init__()
        self.decoder = Decoder(vocab_size, prop_len, p_type=p_type, conditional=conditional)
        self.projection = nn.Linear(d_model, vocab_size).to(device)

    def forward(self, dec_inputs, protein, protein_length, pocket, prop):
        """
        dec_inputs: [batch_size, tgt_len]
        """

        # dec_outpus: [batch_size, tgt_len, d_model], dec_self_attns: [n_layers, batch_size, n_heads, tgt_len, tgt_len]
        dec_outputs, dec_self_attns = self.decoder(dec_inputs, protein, protein_length, pocket, prop)
        # dec_logits: [batch_size, tgt_len, tgt_vocab_size]
        dec_logits = self.projection(dec_outputs)
        return dec_logits.view(-1, dec_logits.size(-1)), dec_self_attns, dec_logits, dec_outputs

    def temperature_sampling(model, tokenizer, input_ids, protein, protein_length, pocket, prop, temperature=0.8, max_length=constant.max_pos):
        with torch.no_grad():
            result = []
            for _ in range(max_length):
                outputs = model(input_ids, protein, protein_length, pocket, prop)
                next_token_logits = outputs[2][:, -1, :]

                # 应用温度参数
                next_token_logits = next_token_logits / temperature

                # 从调整后的分布中采样
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                input_ids = torch.cat([input_ids, next_token], dim=-1)

                end_sign = True
                remove_id = []

                for i in range(next_token.size(0)):
                    if not next_token[i].item() == constant.EOS_TOKEN_ID:
                        end_sign = False
                    else:
                        result.append(input_ids[i].cpu().detach().numpy().tolist())
                        remove_id.append(i)
                if end_sign:
                    break
                else:
                    remove_id = sorted(remove_id, reverse=True)
                    for i in remove_id:
                        if prop is not None:
                            prop = prop[1:]
                        if protein is not None:
                            protein = protein[1:]
                            protein_length = protein_length[1:]
                        if pocket is not None:
                            pocket = pocket[1:]
                        if i < input_ids.size(0) - 1:
                            input_ids = torch.cat((input_ids[:i], input_ids[i+1:]), dim=0)
                        else:
                            input_ids = input_ids[:i]

        ## input_ids = input_ids.cpu().detach().numpy().tolist()
        return [tokenizer.decode(i) for i in result]

    def top_k_sampling(model, tokenizer, input_ids, protein, protein_length,prop, k=40, max_length=constant.max_pos):
        with torch.no_grad():
            result = []
            for _ in range(max_length):
                outputs = model(input_ids, protein, protein_length, prop)
                next_token_logits = outputs[2][:, -1, :]

                # 获取top-k个token
                top_k_logits, top_k_indices = torch.topk(next_token_logits, k, dim=-1)

                # 从top-k中采样
                probs = F.softmax(top_k_logits, dim=-1)
                next_token_index = torch.multinomial(probs, num_samples=1)
                next_token = top_k_indices.gather(-1, next_token_index)

                input_ids = torch.cat([input_ids, next_token], dim=-1)

                end_sign = True
                remove_id = []

                for i in range(next_token.size(0)):
                    if not next_token[i].item() == constant.EOS_TOKEN_ID:
                        end_sign = False
                    else:
                        result.append(input_ids[i].cpu().detach().numpy().tolist())
                        remove_id.append(i)
                if end_sign:
                    break
                else:
                    remove_id = sorted(remove_id, reverse=True)
                    for i in remove_id:
                        if i < input_ids.size(0) - 1:
                            input_ids = torch.cat((input_ids[:i], input_ids[i + 1:]), dim=0)
                        else:
                            input_ids = input_ids[:i]

                ## input_ids = input_ids.cpu().detach().numpy().tolist()
            return [tokenizer.decode(i) for i in result]

    def greedy_decoder(self, dec_input, protein, protein_length):

        terminal = False
        start_dec_len = len(dec_input[0])
        # 一直预测下一个单词，直到预测到"<sep>"结束，如果一直不到"<sep>"，则根据长度退出循环，并在最后加上”<sep>“字符
        while not terminal:
            if len(dec_input[0]) - start_dec_len > 100:
                next_symbol = constant.EOS_TOKEN_ID
                dec_input = torch.cat(
                    [dec_input.detach(), torch.tensor([[next_symbol]], dtype=dec_input.dtype, device=device)], -1)
                break
            dec_outputs, _ = self.decoder(dec_input, protein, protein_length)
            projected = self.projection(dec_outputs)
            prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
            next_word = prob.data[-1]
            next_symbol = next_word
            if next_symbol == constant.EOS_TOKEN_ID:
                terminal = True

            dec_input = torch.cat(
                [dec_input.detach(), torch.tensor([[next_symbol]], dtype=dec_input.dtype, device=device)], -1)

        return dec_input

    def answer(self, sentence, protein, tokenizer, prop=None, method='greedy', batch=1, model='t12_35M', pocket=None, arg=0.8):
        dec_input = tokenizer.encode(sentence).ids
        dec_input = torch.tensor(dec_input, dtype=torch.long, device=device).repeat(batch, 1)

        if protein is not None:
            if model == 't12_35M':
                param = ESM.load_model(model)
                protein_rep = ESM.protein2vector([(protein, '')], param=param)
                protein = protein_rep[2]
                protein_length = protein_rep[1]
            elif model == 'rACSF':
                from gpt.rACSF import cal_rACSF
                protein_rep = cal_rACSF(protein)[0]
                protein = protein_rep
                protein_length = len(protein_rep)
            else:
                protein = protein
                protein_length = len(protein)
            pocket = [i - 1 not in pocket for i in range(protein_length)]


            protein = torch.tensor(protein, device=device, dtype=torch.float)
            protein = protein.unsqueeze(0).repeat(batch, 1, 1)
            protein_length = torch.tensor(protein_length, dtype=torch.long, device=device)
            protein_length = protein_length.repeat(batch)
            pocket = torch.tensor(pocket, device=device, dtype=torch.bool).unsqueeze(0).repeat(batch, 1)
        else:
            protein_length = None
        if prop is not None:
            prop = torch.tensor(prop, dtype=torch.float, device=device).repeat(batch, 1)

        if method == 'greedy':
            output = self.greedy_decoder(dec_input,protein=protein,protein_length=protein_length).squeeze(0).cpu().numpy().tolist()
            output = tokenizer.decode(output)
            answer = output
            return answer
        elif method == 'temperature':
            output = self.temperature_sampling(input_ids=dec_input, tokenizer=tokenizer,protein=protein,protein_length=protein_length, prop=prop, temperature=arg, pocket=pocket)
            return output
        elif method == 'top_k':
            output = self.top_k_sampling(input_ids=dec_input, tokenizer=tokenizer,protein=protein,protein_length=protein_length, prop=prop)
            return output

        return ''


def train_step(model, data_loader, optimizer, criterion, clip=1, print_every=None, vs=0, conditional=['unconditional']):
    if data_loader is None:
        return -1

    if print_every == 0:
        print_every = 1

    print_loss_total = 0  # 每次打印都重置

    epoch_loss = 0

    for i, (dec_inputs, dec_outputs, protein, protein_len, pocket, props) in enumerate(tqdm(data_loader)):
        '''
        dec_inputs: [batch_size, tgt_len]
        dec_outputs: [batch_size, tgt_len]
        '''
        if not 'protein' in conditional:
            protein = None
        if not 'prop' in conditional:
            props = None
        optimizer.zero_grad()
        dec_inputs = torch.tensor(dec_inputs, dtype=torch.long, device=device)
        dec_outputs = torch.tensor(dec_outputs, dtype=torch.long, device=device)

        if protein is not None:
            protein = torch.tensor(protein, dtype=torch.float, device=device)
            protein.unsqueeze(dim=-1)
        if props is not None:
            props = torch.tensor(props, dtype=torch.float, device=device)

        # dec_inputs, dec_outputs = dec_inputs.to(device), dec_outputs.to(device)
        # outputs: [batch_size * tgt_len, tgt_vocab_size]

        outputs, dec_self_attns, dec_logits, _ = model(dec_inputs, protein, protein_len, pocket, props)

        loss = criterion(outputs, dec_outputs.view(-1))
        print_loss_total += loss.item()
        epoch_loss += loss.item()
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        if print_every and (i + 1) % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('\tCurrent Loss: %.4f' % print_loss_avg)

    return epoch_loss / len(data_loader)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def train(data_loader, epochs, vs, lr, model = None, p_type='', conditional=['unconditional'], save_name='GPT.pt'):
    if model is None:
        model = GPT(vocab_size=vs, prop_len=prop_len, p_type=p_type, conditional=conditional)
        try:
            model.load_state_dict(torch.load('checkpoints/fragGPT/GPT.pt'))
        except:
            pass
    else:
        model = model

    params = list(model.parameters())
    total_params = sum(p.numel() for p in params)
    print(f'总参数数量: {total_params}')

    print(model)

    if 'unconditional' in conditional:
        for name, param in model.named_parameters():
            if 'conditional' in name.lower():  # 使用 lower() 忽略大小写
                param.requires_grad = False
                print(f"冻结层: {name}")

        if p_type == 'lora_0':
            for name, param in model.named_parameters():
                if 'lora_0' in name.lower():  # 使用 lower() 忽略大小写
                    param.requires_grad = False
                    print(f"冻结层: {name}")

        elif p_type == 'lora_1':
            for name, param in model.named_parameters():
                if 'lora_1' in name.lower():  # 使用 lower() 忽略大小写
                    param.requires_grad = False
                    print(f"冻结层: {name}")


    from datetime import datetime

    criterion = nn.CrossEntropyLoss(ignore_index=0).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    X = []
    Y = []

    last_loss = math.inf
    model.to(device)

    for train_data in data_loader[0]:
        for epoch in range(epochs):
            X.append(epoch)
            start_time = time.time()
            train_loss = train_step(model, train_data, optimizer, criterion, CLIP, print_every=1000, conditional=conditional)
            Y.append(train_loss)
            end_time = time.time()

            if last_loss > train_loss:
                torch.save(model.state_dict(), 'checkpoints/fragGPT/' + save_name)
                last_loss = train_loss

            for datas in data_loader[1]:
                if datas is not None:
                    with torch.no_grad():
                        epoch_loss_valid = 0

                        for i, (dec_inputs, dec_outputs, protein, protein_len, pocket, props) in enumerate(tqdm(datas)):
                                dec_inputs = torch.tensor(dec_inputs, dtype=torch.long, device=device)
                                dec_outputs = torch.tensor(dec_outputs, dtype=torch.long, device=device)
                                if protein is not None:
                                    protein = torch.tensor(protein, dtype=torch.float, device=device)
                                    protein.unsqueeze(dim=-1)
                                if protein_len is not None:
                                    props = torch.tensor(props, dtype=torch.float, device=device)
                                outputs, dec_self_attns, _, _ = model(dec_inputs, protein, protein_len, pocket, props)

                                loss = criterion(outputs, dec_outputs.view(-1))
                                epoch_loss_valid += loss.item()

                    print(f"valid loss:{epoch_loss_valid / len(datas)}")

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f}')

    now = datetime.now()
    now = now.strftime('%Y-%m-%d_%H-%M-%S')

    np.save('checkpoints/fragGPT/' + now + '_X.npy', X)
    np.save('checkpoints/fragGPT/' + now + '_Y.npy', Y)

    with torch.no_grad():
        epoch_loss = 0
        for datas in data_loader[2]:
            for i, (dec_inputs, dec_outputs, protein, protein_len, pocket, props) in enumerate(tqdm(datas)):
                dec_inputs = torch.tensor(dec_inputs, dtype=torch.long, device=device)
                dec_outputs = torch.tensor(dec_outputs, dtype=torch.long, device=device)
                if protein is not None:
                    protein = torch.tensor(protein, dtype=torch.float, device=device)
                    protein.unsqueeze(dim=-1)
                if protein_len is not None:
                    props = torch.tensor(props, dtype=torch.float, device=device)
                outputs, dec_self_attns, _, _ = model(dec_inputs, protein, protein_len, pocket, props)

                loss = criterion(outputs, dec_outputs.view(-1))
                epoch_loss += loss.item()

        print(f"test loss:{epoch_loss / len(datas)}")

    return model

def test(smile, path, p_type='', conditional=['unconditional']):
    model_path = path + 'GPT.pt'
    tokenizer_path = path + 'frag_tokenizer.json'

    tokenizer_ = tokenizer.tokenizer_from_file(file_path=tokenizer_path)
    smile = dataprocess._mol_decom_mp([smile], n_core=1)[0]
    smiles = tokenizer_.encode(smile[0]).ids
    title = tokenizer_.encode(smile[0]).tokens[:-1]

    start = tokenizer_.encode('{').ids[0]
    end = tokenizer_.encode('}').ids[0]
    blocks = []

    count = 0
    if_start = False

    for i in smiles:
        if not if_start:
            if not i == start:
                count += 1
            else:
                if count > 0:
                    blocks.append(count)
                count = 1
                if_start = True
        else:
            if i == end:
                count += 1
                blocks.append(count)
                count = 0
                if_start = False
            else:
                count += 1

    model = GPT(vocab_size=tokenizer_.get_vocab_size(), prop_len=prop_len, p_type=p_type, conditional=conditional)
    try:
        model.load_state_dict(torch.load(model_path))
    except Exception as e:
        print(e)
        return None

    model.to(device)
    model.eval()

    dataset = PLDataSet([(None, smiles, None)])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False,
                                  collate_fn=dataset.padding_batch)


    with torch.no_grad():
        for datas in dataloader:
            dec_inputs, dec_outputs, protein, protein_len, props = datas
            dec_inputs = torch.tensor(dec_inputs, dtype=torch.long, device=device)
            _, dec_self_attns, _, dec_out = model(dec_inputs, protein, protein_len, props)


    result = torch.stack(dec_self_attns).sum(dim=2).squeeze(dim=1).cpu().numpy()

    results = [i.cpu().numpy() for i in dec_self_attns]
    return results, result, blocks, title

def fine_tune(model_path, data_loader, epochs, vs, lr, p_type='', conditional=['prop'], save_name='GPT.pt'):
    model = GPT(vocab_size=vs, prop_len=prop_len, p_type=p_type, conditional=conditional)
    model.load_state_dict(torch.load(model_path), strict=False)

    train(data_loader, epochs, vs, lr, model, p_type=p_type, conditional=conditional, save_name=save_name)

def train_fragGPT(epoch):
    data_loader, tokenizer__ = get_frag_default_dataloader()
    train(data_loader, epoch, tokenizer__.get_vocab_size())

def fine_tune_fragGPT(epoch, model_path, plk_path, tokenizer_path):
    data_loaders, tokenizer__ = get_PL_dataloader(plk_path=plk_path, tokenizer_path=tokenizer_path)
    lr = 5e-5
    model = GPT(vocab_size=tokenizer__.get_vocab_size())
    model.load_state_dict(torch.load(model_path), strict=False)
    train(((data_loaders,),None,data_loaders), epoch, tokenizer__.get_vocab_size(), lr, model=model)

def fine_tune_fragGPT_molonly(epoch, model_path, tokenizer_path, train_path='gpt/finetune.txt'):
    tokenizer__ = tokenizer.tokenizer_from_file(file_path=tokenizer_path)
    smiles = []
    with open(train_path, 'r') as file_:
        line = file_.readline().strip('\n')
        while line:
            smiles.append(line)
            line = file_.readline().strip('\n')
    dataprocess.mol_decomp_mp_(smiles, n_core=60)
    data_loaders = get_frag_dataloader_without_split(frag_token_fun_1, tokenizer__,
                                             train_file='gpt/frag_decom_test_other.txt',
                                             test_file='', batch_size=30,
                                             multiset=1)

    lr = 5e-5
    model = GPT(vocab_size=tokenizer__.get_vocab_size())
    model.load_state_dict(torch.load(model_path), strict=False)

    params = list(model.parameters())
    total_params = sum(p.numel() for p in params)
    print(f'总参数数量: {total_params}')

    train(data_loaders, epoch, tokenizer__.get_vocab_size(), lr, model=model)

def train_PL_fragGPT(epoch, plk_path):
    data_loaders, tokenizer__ = get_PL_dataloader(plk_path=plk_path)
    lr = 4e-4
    train(((data_loaders,),None,data_loaders), epoch, tokenizer__.get_vocab_size(), lr)


if __name__ == '__main__':
    x, xx, blocks, title = test('CC(C)CC1=CC=C(C=C1)C(C)C(=O)O', path='../checkpoints/fragGPT/D4_geom_lora_1_unconditional/', p_type='lora_1', conditional=['unconditional'])
    from heatmap import main
    main(data=xx[-12], row_blocks=blocks, col_blocks=blocks, row_headers=title, col_headers=title)

    print(x)


