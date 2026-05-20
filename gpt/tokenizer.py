from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer, BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import constant
from constant import *

from ZINC.dataprocess import mol_decomp_mp_ZINC_250K_pkl

def get_token(source):
    print(f"getting token from smiles")
    n_core = 60

    mol_dict = {
        'ZINC_250K':mol_decomp_mp_ZINC_250K_pkl,
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
    frag_tokens = get_token(n, source=source)
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

if __name__ == '__main__':
    get_new_tokenizer(n=20, source='ZINC_pkl')
    # tokenizer = tokenizer_from_file()
    # results = get_token(100000)
    pass