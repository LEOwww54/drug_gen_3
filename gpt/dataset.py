import pickle as pkl
from bisect import bisect_right
from math import floor

import torch
import torch.utils.data as Data
from torch.utils.data import DataLoader

import constant
import gpt.tokenizer as tokenizer

from constant import *
from tqdm import tqdm

class fragDataSet(Data.Dataset):
    def __init__(self, datas):
        self.datas = datas

    def __getitem__(self, item):
        index = self.datas[item][0]
        data = self.datas[item][1]
        props = self.datas[item][2]
        decoder_input = data[:-1]
        decoder_output = data[1:]

        decoder_input_len = len(decoder_input)
        decoder_output_len = len(decoder_output)

        # return (decoder_input, decoder_output)

        return {"decoder_input": decoder_input, "decoder_input_len": decoder_input_len,
                "decoder_output": decoder_output, "decoder_output_len": decoder_output_len, "index":index, 'props': props}

    def __len__(self):
        return len(self.datas)

    def padding_batch(self, batch):
        decoder_input_lens = [d["decoder_input_len"] for d in batch]
        decoder_output_lens = [d["decoder_output_len"] for d in batch]

        decoder_input_maxlen = max(decoder_input_lens)
        decoder_output_maxlen = max(decoder_output_lens)

        pad_id = tokenizer.PAD_TOKEN_ID

        for d in batch:
            d["decoder_input"].extend([pad_id] * (decoder_input_maxlen - d["decoder_input_len"]))
            d["decoder_output"].extend([pad_id] * (decoder_output_maxlen - d["decoder_output_len"]))
        decoder_inputs = torch.tensor([d["decoder_input"] for d in batch], dtype=torch.long)
        decoder_outputs = torch.tensor([d["decoder_output"] for d in batch], dtype=torch.long)

        return decoder_inputs, decoder_outputs, None, None

    def load_from_frag_file(self, frag_file='frag_test.txt'):
        with open(frag_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip('\t')
                line = line[1:]

            self.datas = lines

def frag_token_fun(tokenizer_, r : list):
    result = []
    for token in r:
        result.append(tokenizer_.token_to_id(token))

    return result

def frag_token_fun_1(tokenizer_, r : list):
    sentence = ''
    for token in r:
        sentence += token + ' '

    sentence = sentence[:-1]

    idss = tokenizer_.encode(sentence).ids

    return idss

def _split_validation_test(dataset, batch_size, collate_fn):
    """Deterministic, disjoint holdout split; leave an odd extra sample for test."""
    if len(dataset) < 2:
        raise ValueError('At least two test samples are required when no validation set is provided')
    valid_size = len(dataset) // 2
    valid, test = Data.random_split(
        dataset, [valid_size, len(dataset) - valid_size],
        generator=torch.Generator().manual_seed(42))
    return tuple(DataLoader(part, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn) for part in (valid, test))


def get_frag_dataloader_without_split(token_fun, tokenizer_, batch_size=400, train_file = "gpt/frag_test.txt", valid_file='', test_file = '', multiset = 1, protein=True):
    total = 0
    datas = []
    line_count = 0

    datas = load_PL_datapair_from_pkl(train_file, tokenizer_=tokenizer_, token_fun=token_fun, protein=protein)
    total = len(datas)

    total_each = floor(total / multiset)
    dataloader_g = []

    count_g = 0

    for i_set in range(multiset):
        i_begin = i_set * total_each

        if i_set == multiset - 1:
            i_end = total
        else:
            i_end = (i_set + 1) * total_each

        data_ = datas[i_begin:i_end]

        dataset_train = PLDataSet(data_)

        dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True,
                                      collate_fn=dataset_train.padding_batch)

        dataloader_g.append(dataloader_train)


    if not valid_file == '':
        datas = load_PL_datapair_from_pkl(valid_file, tokenizer_=tokenizer_, token_fun=token_fun)

        dataset_valid = PLDataSet(datas)
        dataloader_valid = DataLoader(dataset_valid, batch_size=batch_size, shuffle=True,
                                      collate_fn=dataset_valid.padding_batch)
    else:
        dataloader_valid = None

    if not test_file == '':
        datas = load_PL_datapair_from_pkl(test_file, tokenizer_=tokenizer_, token_fun=token_fun)

        dataset_test = PLDataSet(datas)
        dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True,
                                      collate_fn=dataset_test.padding_batch)
        if dataloader_valid is None:
            dataloader_valid, dataloader_test = _split_validation_test(
                dataset_test, batch_size, dataset_test.padding_batch)
    else:
        dataloader_test = None

    return dataloader_g, [dataloader_valid,], [dataloader_test,]

def get_frag_dataloader(token_fun, tokenizer_, data_split = [], batch_size=400, file = "gpt/frag_file/frag.txt", multiset = 1, protein=True):
    total = 0
    datas = []
    line_count = 0
    print(f'loading smiles from {file} with batch size {batch_size} ------ multiset {multiset}')

    print('loading pkl file...')
    datas = load_PL_datapair_from_pkl(path = file, tokenizer_=tokenizer_, token_fun=token_fun, protein=protein)
    total = len(datas)
    print(f'{total} datas are loaded')

    import random
    random.shuffle(datas)

    total_each = floor(total / multiset)
    dataloader_g = []
    dataloader_v = []
    dataloader_t = []

    count_g = 0

    for i_set in range(multiset):
        i_begin = i_set * total_each

        if i_set == multiset - 1:
            i_end = total
        else:
            i_end = (i_set + 1) * total_each

        data_ = datas[i_begin:i_end]
        data_count = len(data_)

        smiles_train, smiles_valid, smiles_test = [], [], []

        i_train_begin = 0
        i_train_end = round(data_count * data_split[0])

        i_valid_begin = i_train_end
        i_valid_end = round(data_count * data_split[1])

        i_test_begin = i_valid_end
        i_test_end = data_count

        try:
            smiles_train = data_[i_train_begin:i_train_end]
            dataset_train = PLDataSet(smiles_train)
            dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True,
                                          collate_fn=dataset_train.padding_batch)
            dataloader_g.append(dataloader_train)
        except:
            dataloader_g.append(None)

        try:
            smiles_valid = data_[i_valid_begin:i_valid_end]
            dataset_valid = PLDataSet(smiles_valid)
            dataloader_valid = DataLoader(dataset_valid, batch_size=batch_size, shuffle=True,
                                          collate_fn=dataset_valid.padding_batch)
            dataloader_v.append(dataloader_valid)
        except:
            dataloader_v.append(None)

        try:
            smiles_test = data_[i_test_begin:i_test_end]
            dataset_test = PLDataSet(smiles_test)
            dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True, collate_fn=dataset_test.padding_batch)
            dataloader_t.append(dataloader_test)
        except:
            dataloader_t.append(None)

    print('dataset done')

    return dataloader_g, dataloader_v, dataloader_t

class PLDataSet(Data.Dataset):
    def __init__(self, datas):
        self.datas = datas

    def __getitem__(self, item):
        data = self.datas[item]

        if data[0] is None:
            protein = None
            protein_len = 0
        else:
            protein = data[0][0]
            protein_len = len(protein)

        smiles = data[1]

        if data[2] is None:
            props = None
        else:
            props = data[2]

        if data[3] is None:
            pocket = None
        else:
            pocket = data[3]
            pocket = [i - 1 for i in pocket]

        if not smiles[0] == START_TOKEN_ID:
            smiles = [START_TOKEN_ID] + smiles
        if not smiles[-1] == EOS_TOKEN_ID:
            smiles = smiles + [EOS_TOKEN_ID]

        decoder_input = smiles[:-1]
        decoder_output = smiles[1:]

        decoder_input_len = len(decoder_input)
        decoder_output_len = len(decoder_output)

        # return (decoder_input, decoder_output)


        return {"decoder_input": decoder_input, "decoder_input_len": decoder_input_len,
                "decoder_output": decoder_output, "decoder_output_len": decoder_output_len, 'protein_len': protein_len,
                'protein': protein, 'props': props, 'pocket': pocket}

    def __len__(self):
        return len(self.datas)

    def padding_batch(self, batch):
        decoder_input_lens = [d["decoder_input_len"] for d in batch]
        decoder_output_lens = [d["decoder_output_len"] for d in batch]
        protein_lens = [d["protein_len"] for d in batch]

        decoder_input_maxlen = max(decoder_input_lens)
        decoder_output_maxlen = max(decoder_output_lens)
        protein_maxlen = max(protein_lens)

        pad_id = PAD_TOKEN_ID

        is_protein = True
        is_props = True
        is_pocket = True

        for d in batch:
            d["decoder_input"].extend([pad_id] * (decoder_input_maxlen - d["decoder_input_len"]))
            d["decoder_output"].extend([pad_id] * (decoder_output_maxlen - d["decoder_output_len"]))
            if d["protein"] is not None:
                d["protein"].extend([[pad_id] * protein_emb_size] * (protein_maxlen - d["protein_len"]))
            else:
                is_protein = False

            if d["props"] is None:
                is_props = False

            if d["pocket"] is not None:
                x = [i not in d["pocket"] for i in range(protein_maxlen)]
                d["pocket"] = x
            else:
                is_pocket = False

        decoder_inputs = torch.tensor([d["decoder_input"] for d in batch], dtype=torch.long)
        decoder_outputs = torch.tensor([d["decoder_output"] for d in batch], dtype=torch.long)

        if not is_protein:
            protein = protein_lens = None
        else:
            protein = torch.tensor([d["protein"] for d in batch], dtype=torch.float)
            protein_lens = torch.tensor([d["protein_len"] for d in batch], dtype=torch.long)

        if is_props:
            props = torch.tensor([d["props"] for d in batch], dtype=torch.float)
        else:
            props = None

        if is_pocket:
            pocket = torch.tensor([d["pocket"] for d in batch], dtype=torch.bool)
        else:
            pocket = None

        return decoder_inputs, decoder_outputs, protein, protein_lens, pocket, props

def token_fun_2(tokenizer_, lines):
    results = tokenizer_.encode(lines)
    return results

def load_PL_datapair_from_pkl(path, tokenizer_, token_fun, protein=True):
    with open(path, 'rb') as f:
        x = pkl.load(f)
    print('pickle file loaded')

    protein_dict = x['protein_dict']
    mols = x['mol']

    datas = []

    for k, v in tqdm(mols.items()):
        frag = v['frag']
        frag = token_fun(tokenizer_, frag)
        if(len(frag.ids) >= constant.max_pos - 2):
            continue
        if constant.UNK_TOKEN_ID in frag.ids:
            print('warning: unknown token detected...')
        prop = v['props']

        protein_index = v['protein']
        pocket = v['pocket']

        if not protein:
            protein_rep = None
        else:
            if protein_index == -1:
                protein_rep = None
            else:
                try:
                    protein_rep = protein_dict[protein_index]['rep']
                except:
                    print('protein index' + str(protein_index) + 'not found')
                    protein_rep = None

        datas.append((protein_rep, frag.ids, prop, pocket))

    return datas

class PAMFDataSet(PLDataSet):
    """Pre-split PAMF records, retaining row IDs and original SMILES for audits."""

    def __init__(self, path, tokenizer_, *, require_properties=True):
        import math
        tokenizer.validate_special_tokens(tokenizer_)
        with open(path, 'rb') as handle:
            records = pkl.load(handle)['mol']
        datas = []
        self.row_ids, self.original_smiles = [], []
        self.skipped_rows = []
        self.unknown_tokens = 0
        for key, row in records.items():
            words = tokenizer.split_pamf_sentence(row['frag'])
            if words and words[0] == START_TOKEN:
                words = words[1:]
            if words and words[-1] == EOS_TOKEN:
                words = words[:-1]
            if not words or len(words) + 2 > constant.max_pos:
                self.skipped_rows.append(key)
                continue
            ids = [tokenizer_.token_to_id(word) for word in words]
            self.unknown_tokens += sum(i is None for i in ids)
            ids = [UNK_TOKEN_ID if i is None else i for i in ids]
            props = row.get('props')
            if require_properties and (props is None or len(props) != constant.prop_len
                                       or not all(math.isfinite(float(v)) for v in props)):
                raise ValueError(f'Invalid PAMF properties in {path}, row {key}')
            datas.append((None, [START_TOKEN_ID, *ids, EOS_TOKEN_ID],
                          list(props) if require_properties else None, None))
            self.row_ids.append(key)
            self.original_smiles.append(row['oring'])
        super().__init__(datas)


def get_pamf_dataloader_without_split(tokenizer_, train_file, test_file, *,
                                     batch_size=50, require_properties=True, valid_file=None):
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('batch_size must be a positive integer')
    train = PAMFDataSet(train_file, tokenizer_, require_properties=require_properties)
    test = PAMFDataSet(test_file, tokenizer_, require_properties=require_properties)
    if not len(train):
        raise ValueError('No usable PAMF training samples')
    if train.unknown_tokens:
        raise ValueError('PAMF training tokens absent from vocabulary; rebuild the matching tokenizer')
    print(f'PAMF train={len(train)}, test={len(test)}, '
          f'length-filtered={len(train.skipped_rows) + len(test.skipped_rows)}, '
          f'test unknown tokens={test.unknown_tokens}')
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True,
                              collate_fn=train.padding_batch)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False,
                             collate_fn=test.padding_batch)
    if valid_file:
        valid = PAMFDataSet(valid_file, tokenizer_, require_properties=require_properties)
        valid_loader = DataLoader(valid, batch_size=batch_size, shuffle=False,
                                  collate_fn=valid.padding_batch)
    else:
        valid_loader, test_loader = _split_validation_test(test, batch_size, test.padding_batch)
    print(f'PAMF validation={len(valid_loader.dataset)}, test={len(test_loader.dataset)}')
    return [train_loader], [valid_loader], [test_loader]


class FragPretrainDataSet(PLDataSet):
    """Count-expanded view of a compact single-fragment SISR PKL."""

    def __init__(self, path, tokenizer_, *, row_keys=None):
        tokenizer.validate_special_tokens(tokenizer_)
        with open(path, 'rb') as handle:
            payload = pkl.load(handle)
        if payload.get('format') != 'frag_pretrain_v1' or not isinstance(payload.get('mol'), dict):
            raise ValueError(f'Invalid fragment-pretraining PKL: {path}')
        records = payload['mol']
        keys = list(records) if row_keys is None else list(row_keys)
        datas, cumulative = [], []
        self.row_ids, self.original_smiles, self.counts = [], [], []
        self.skipped_rows, self.unknown_tokens = [], 0
        total = 0
        for key in keys:
            row = records[key]
            words = tokenizer.split_pamf_sentence(row['frag'])
            if words and words[0] == START_TOKEN:
                words = words[1:]
            if words and words[-1] == EOS_TOKEN:
                words = words[:-1]
            if not words or len(words) + 2 > constant.max_pos:
                self.skipped_rows.append(key)
                continue
            count = row.get('count')
            if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                raise ValueError(f'Invalid fragment count in {path}, row {key}')
            ids = [tokenizer_.token_to_id(word) for word in words]
            self.unknown_tokens += sum(i is None for i in ids) * count
            ids = [UNK_TOKEN_ID if i is None else i for i in ids]
            datas.append((None, [START_TOKEN_ID, *ids, EOS_TOKEN_ID], None, None))
            total += count
            cumulative.append(total)
            self.row_ids.append(key)
            self.original_smiles.append(row['oring'])
            self.counts.append(count)
        super().__init__(datas)
        self.cumulative_counts = cumulative

    def __len__(self):
        return self.cumulative_counts[-1] if self.cumulative_counts else 0

    def __getitem__(self, item):
        if item < 0:
            item += len(self)
        if item < 0 or item >= len(self):
            raise IndexError(item)
        unique_index = bisect_right(self.cumulative_counts, item)
        return super().__getitem__(unique_index)


def get_frag_pretrain_dataloader_without_split(tokenizer_, train_file, test_file, *,
                                               batch_size=50):
    """Load compact corpora and split test by unique fragment for validation."""
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('batch_size must be a positive integer')
    train = FragPretrainDataSet(train_file, tokenizer_)
    full_test = FragPretrainDataSet(test_file, tokenizer_)
    if not len(train):
        raise ValueError('No usable fragment-pretraining samples')
    if train.unknown_tokens:
        raise ValueError('Training tokens absent from vocabulary; rebuild the matching tokenizer')
    if len(full_test.row_ids) < 2:
        raise ValueError('At least two unique test fragments are required for validation/test splitting')
    order = torch.randperm(len(full_test.row_ids), generator=torch.Generator().manual_seed(42)).tolist()
    boundary = len(order) // 2
    valid_keys = [full_test.row_ids[i] for i in order[:boundary]]
    test_keys = [full_test.row_ids[i] for i in order[boundary:]]
    valid = FragPretrainDataSet(test_file, tokenizer_, row_keys=valid_keys)
    test = FragPretrainDataSet(test_file, tokenizer_, row_keys=test_keys)
    print(f'Fragment pretrain train={len(train)} ({len(train.row_ids)} unique), '
          f'validation={len(valid)} ({len(valid.row_ids)} unique), '
          f'test={len(test)} ({len(test.row_ids)} unique), '
          f'length-filtered={len(train.skipped_rows) + len(full_test.skipped_rows)}, '
          f'test unknown tokens={full_test.unknown_tokens}')
    return (
        [DataLoader(train, batch_size=batch_size, shuffle=True, collate_fn=train.padding_batch)],
        [DataLoader(valid, batch_size=batch_size, shuffle=False, collate_fn=valid.padding_batch)],
        [DataLoader(test, batch_size=batch_size, shuffle=False, collate_fn=test.padding_batch)],
    )


def load_protein_emb(pkl_file_path='gpt/protein2vector.pkl'):
    with open(pkl_file_path, 'rb') as f:
        results = pkl.load(f)

    return results

def load_PL_datapair_from_plk_PDBbind(tokenizer, path='gpt/protein2vector.pkl'):
    x = load_protein_emb(pkl_file_path=path)
    t = tokenizer
    data_pair = []
    unk_token = []
    for k, v in x.items():
        tmp1 = v['emb'][0]
        ligands = v['ligand']
        for ligand in ligands:
            frags = ligand[1]
            tmp2 = []
            tmp2.append(START_TOKEN_ID)
            for frag in frags:
                tok = t.token_to_id(frag)
                if tok is None:
                    unk_token.append(frag)
                    tmp2.append(UNK_TOKEN_ID)
                else:
                    tmp2.append(tok)
            tmp2.append(EOS_TOKEN_ID)
            data_pair.append((tmp1, tmp2))

    return data_pair, set(unk_token)

def get_PL_dataloader(plk_path, tokenizer_path):
    t = tokenizer.tokenizer_from_file(tokenizer_path)
    data_pair, unk_token = load_PL_datapair_from_plk_PDBbind(t, path=plk_path)
    PLdataset = PLDataSet(data_pair)

    PLdataloader = DataLoader(PLdataset, batch_size=2, shuffle=True,collate_fn=PLdataset.padding_batch)
    return PLdataloader, t

def get_PL_dataloader_pkl(pkl_path, tokenizer_path):
    t = tokenizer.tokenizer_from_file(tokenizer_path)
    datas = load_PL_datapair_from_pkl(path=pkl_path,tokenizer_=t)
    PLdataset = PLDataSet(datas)

    PLdataloader = DataLoader(PLdataset, batch_size=2, shuffle=True, collate_fn=PLdataset.padding_batch)
    return PLdataloader, t

def t1(path, tokenizer_, token_fun=token_fun_2):
    with open(path, 'rb') as f:
        x = pkl.load(f)
    print('pickle file loaded')

    protein_dict = x['protein_dict']
    mols = x['mol']

    datas = {}

    for k, v in mols.items():
        frag = v['frag']
        frag = token_fun(tokenizer_, frag)
        s = len(frag.ids)
        if s in datas:
            datas[s] += 1
        else:
            datas[s] = 1

    return datas


if __name__ == '__main__':
    t = t1(path='frag_file/frag_decom_chembl.pkl',
           tokenizer_=tokenizer.tokenizer_from_file('vocab/frag_tokenizer_chembl.json'))




