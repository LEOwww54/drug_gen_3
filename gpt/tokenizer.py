from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer, BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.pre_tokenizers import WhitespaceSplit
from pathlib import Path
import pickle
import constant
from constant import *

from ZINC_250K.dataprocess import mol_decomp_mp_ZINC_250K_pkl
from ZINC_refined.dataprocess import mol_decomp_mp_ZINC_refined

def get_token(source):
    print(f"getting token from smiles")
    n_core = 60

    mol_dict = {
        'ZINC_250K':mol_decomp_mp_ZINC_250K_pkl,
        'ZINC_refined': mol_decomp_mp_ZINC_refined,
    }

    mols = mol_dict.get(source)(n_core=n_core)

    frag_tokens = []

    for mol in mols:
        for frag in mol:
            frag_tokens.append(frag)

    frag_tokens = set(frag_tokens)
    frag_tokens = list(frag_tokens)

    print(f"getting token done")
    return frag_tokens

def get_new_tokenizer(source):
    if source in ('ZINC_250K_frag_pretrain', 'ZINC_refined_frag_pretrain',
                  'ZIINC_refined_frag_pretrain'):
        return get_new_frag_pretrain_tokenizer(source)
    if source in ('ZINC_250K_pamf', 'ZINC_refined_pamf'):
        return get_new_pamf_tokenizer(source.removesuffix('_pamf'))
    print(f"training new tokenizer")
    frag_tokens = get_token(source=source)

    tokenizer_frag = Tokenizer(WordLevel(unk_token=UNK_TOKEN))
    tokenizer_frag.pre_tokenizer = Whitespace()

    trainer_1 = WordLevelTrainer(
        min_frequency=1,
        special_tokens=SPECIAL_TOKENS
    )

    # 训练 tokenizer
    tokenizer_frag.train_from_iterator([], trainer_1)
    tokenizer_frag.add_tokens([token for token in frag_tokens if token not in REMOVED_SPECIAL_TOKENS])

    print(f"saving tokenizer")
    import os
    gpt_folder = "gpt"
    frag_file_path = os.path.join(gpt_folder, "vocab")
    if not os.path.exists(frag_file_path):
        os.makedirs(frag_file_path)
        print(f"创建文件夹: {frag_file_path}")
    else:
        pass

    tokenizer_frag.save("gpt/vocab/frag_tokenizer_" + source + ".json")

    return tokenizer_frag

def get_new_tokenizer_with_extra(n, source, extra_tokens=[]):
    print(f"training new tokenizer")
    frag_tokens = get_token(source=source)
    frag_tokens.extend(extra_tokens)

    tokenizer_frag = Tokenizer(WordLevel(unk_token=UNK_TOKEN))
    tokenizer_frag.pre_tokenizer = Whitespace()

    trainer_1 = WordLevelTrainer(
        min_frequency=1,
        special_tokens=SPECIAL_TOKENS
    )

    # 训练 tokenizer
    tokenizer_frag.train_from_iterator([], trainer_1)
    tokenizer_frag.add_tokens([token for token in frag_tokens if token not in REMOVED_SPECIAL_TOKENS])
    print(f"saving tokenizer")

    tokenizer_frag.save("gpt/vocab/frag_tokenizer_" + source + ".json")

    return tokenizer_frag

def validate_special_tokens(tokenizer):
    """Reject incompatible IDs before they can corrupt EOS handling or training."""
    if (any(tokenizer.token_to_id(token) != index
            for index, token in enumerate(SPECIAL_TOKENS))
            or any(tokenizer.token_to_id(token) is not None for token in REMOVED_SPECIAL_TOKENS)):
        raise ValueError(
            "Legacy or incompatible tokenizer: rebuild the vocabulary (PAMF: s=True) "
            "and retrain, or migrate the tokenizer and checkpoint together. "
            "Required special IDs: <pad>=0, </s>=1, <unk>=2, <start>=3.")
    return tokenizer


def tokenizer_from_file(file_path='gpt/vocab/frag_tokenizer.json'):
    return validate_special_tokens(Tokenizer.from_file(str(file_path)))


def pamf_training_paths(source):
    if source not in ('ZINC_250K', 'ZINC_refined'):
        raise ValueError('PAMF source must be ZINC_250K or ZINC_refined')
    root = Path(__file__).resolve().parents[1]
    return dict(train=root/'gpt'/'frag_file'/f'frag_decom_{source}_pamf_train.pkl',
                test=root/'gpt'/'frag_file'/f'frag_decom_{source}_pamf_test.pkl',
                tokenizer=root/'gpt'/'vocab'/f'frag_tokenizer_{source}_pamf.json',
                checkpoint=root/'checkpoints'/'fragGPT'/f'{source}_pamf')


def split_pamf_sentence(sentence):
    """Split on whitespace: '<m- 1>' represents the two tokens '<m-' and '1>'."""
    return sentence.split()


def get_new_pamf_tokenizer(source, *, train_file=None, output_path=None):
    """Build deterministic atomic-token vocabulary from the training split only.

    Uses existing PAMF PKL output; never reruns decomposition or reads test data.
    """
    paths = pamf_training_paths(source)
    train_file = Path(train_file) if train_file is not None else paths['train']
    output_path = Path(output_path) if output_path is not None else paths['tokenizer']
    with train_file.open('rb') as handle:
        records = pickle.load(handle)['mol']
    if not records:
        raise ValueError(f'PAMF training split is empty: {train_file}')
    tokens = set()
    for key, row in records.items():
        sentence = row['frag']
        if not isinstance(sentence, str) or not sentence.strip():
            raise ValueError(f'Invalid PAMF token sentence at row {key}')
        tokens.update(split_pamf_sentence(sentence))
    vocabulary = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
    for token in sorted(tokens - set(SPECIAL_TOKENS) - REMOVED_SPECIAL_TOKENS):
        vocabulary[token] = len(vocabulary)
    result = Tokenizer(WordLevel(vocab=vocabulary, unk_token=UNK_TOKEN))
    # Whitespace() splits punctuation; PAMF/FST tokens must remain atomic.
    result.pre_tokenizer = WhitespaceSplit()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(str(output_path))
    return result


def get_new_frag_pretrain_tokenizer(source, *, train_file=None, output_path=None,
                                    reuse_pamf_vocab=True, base_tokenizer_path=None,
                                    pamf_vocab_only=None):
    """Build fragment vocabulary, preserving PAMF token IDs by default.

    PAMF vocabulary reuse guarantees attachment markers exist; tokens observed
    only in the augmented fragment corpus are appended deterministically.
    """
    from gpt.frag_pretrain_dataprocess import (frag_pretrain_paths,
                                               normalize_frag_pretrain_source)

    source = normalize_frag_pretrain_source(source)
    paths = frag_pretrain_paths(source)
    train_file = Path(train_file) if train_file is not None else paths['train']
    output_path = Path(output_path) if output_path is not None else paths['tokenizer']
    with train_file.open('rb') as handle:
        payload = pickle.load(handle)
    if payload.get('format') != 'frag_pretrain_v1' or not payload.get('mol'):
        raise ValueError(f'Invalid or empty fragment-pretraining PKL: {train_file}')
    tokens = set()
    for key, row in payload['mol'].items():
        sentence = row.get('frag')
        if not isinstance(sentence, str) or not sentence.strip():
            raise ValueError(f'Invalid fragment-pretraining sentence at row {key}')
        words = split_pamf_sentence(sentence)
        if words and words[0] == START_TOKEN:
            words = words[1:]
        if words and words[-1] == EOS_TOKEN:
            words = words[:-1]
        if not words:
            raise ValueError(f'Empty fragment-pretraining sentence at row {key}')
        tokens.update(words)
    if not isinstance(reuse_pamf_vocab, bool):
        raise ValueError('reuse_pamf_vocab must be boolean')
    if pamf_vocab_only is None:
        pamf_vocab_only = payload.get('vocabulary_mode') == 'pamf_only'
    if not isinstance(pamf_vocab_only, bool):
        raise ValueError('pamf_vocab_only must be boolean or None')
    if pamf_vocab_only and not reuse_pamf_vocab:
        raise ValueError('pamf_vocab_only requires reuse_pamf_vocab=True')
    if reuse_pamf_vocab:
        base_path = (Path(base_tokenizer_path) if base_tokenizer_path is not None else
                     Path(__file__).resolve().parent / 'vocab' /
                     f'frag_tokenizer_{source}_pamf.json')
        if not base_path.is_file():
            raise FileNotFoundError(f'Missing PAMF base tokenizer: {base_path}')
        base = tokenizer_from_file(base_path)
        vocabulary = dict(sorted(base.get_vocab().items(), key=lambda item: item[1]))
    else:
        vocabulary = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
    missing = tokens - set(vocabulary)
    if pamf_vocab_only and missing:
        preview = ', '.join(repr(token) for token in sorted(missing)[:10])
        raise ValueError(f'Original PAMF vocabulary does not cover fragment tokens: {preview}')
    if not pamf_vocab_only:
        for token in sorted(missing - set(SPECIAL_TOKENS) - REMOVED_SPECIAL_TOKENS):
            vocabulary[token] = len(vocabulary)
    result = Tokenizer(WordLevel(vocab=vocabulary, unk_token=UNK_TOKEN))
    result.pre_tokenizer = WhitespaceSplit()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(str(output_path))
    return result

if __name__ == '__main__':
    get_new_tokenizer(source='ZINC_250K')
    # tokenizer = tokenizer_from_file()
    # results = get_token(100000)
    pass
