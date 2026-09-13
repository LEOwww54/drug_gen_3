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
    tokenizer_frag.add_tokens(frag_tokens)

    # for token in frag_tokens:
    #     if token[0] == '[':
    #         token_property_cal(mol_translate('<sep>' + token)[0])

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
    tokenizer_frag.add_tokens(frag_tokens)
    print(f"saving tokenizer")

    tokenizer_frag.save("gpt/vocab/frag_tokenizer_" + source + ".json")

    return tokenizer_frag

def tokenizer_from_file(file_path='gpt/vocab/frag_tokenizer.json'):
    """读取分词文件"""
    tokenizer = Tokenizer(WordLevel(unk_token=UNK_TOKEN))
    tokenizer = tokenizer.from_file(file_path)
    return tokenizer


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
    for token in sorted(tokens - set(SPECIAL_TOKENS)):
        vocabulary[token] = len(vocabulary)
    result = Tokenizer(WordLevel(vocab=vocabulary, unk_token=UNK_TOKEN))
    # Whitespace() splits punctuation; PAMF/FST tokens must remain atomic.
    result.pre_tokenizer = WhitespaceSplit()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(str(output_path))
    return result

if __name__ == '__main__':
    get_new_tokenizer(source='ZINC_250K')
    # tokenizer = tokenizer_from_file()
    # results = get_token(100000)
    pass
