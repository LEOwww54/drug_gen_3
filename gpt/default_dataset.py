import gpt.tokenizer as tokenizer
from gpt.dataset import get_frag_dataloader_without_split, token_fun_2
from gpt.dataset import get_pamf_dataloader_without_split


def get_frag_default_dataloader_pamf(source, token_path=None, *, batch_size=50,
                                     require_properties=True):
    paths = tokenizer.pamf_training_paths(source)
    tokenizer_ = tokenizer.tokenizer_from_file(str(token_path or paths['tokenizer']))
    loaders = get_pamf_dataloader_without_split(
        tokenizer_, paths['train'], paths['test'], batch_size=batch_size,
        require_properties=require_properties)
    return loaders, tokenizer_


def get_frag_default_dataloader_ZINC_250K_pamf_pkl(token_path=None, *, batch_size=50,
                                                  require_properties=True):
    return get_frag_default_dataloader_pamf('ZINC_250K', token_path,
                                           batch_size=batch_size, require_properties=require_properties)


def get_frag_default_dataloader_ZINC_refined_pamf_pkl(token_path=None, *, batch_size=100,
                                                     require_properties=True):
    return get_frag_default_dataloader_pamf('ZINC_refined', token_path,
                                           batch_size=batch_size, require_properties=require_properties)

def get_frag_default_dataloader_ZINC_250K_pkl(token_path, s1):
    tokenizer__ = tokenizer.tokenizer_from_file(file_path=token_path)
    train_path = 'gpt/frag_file/frag_decom_ZINC_250K_train.pkl'
    test_path = 'gpt/frag_file/frag_decom_ZINC_250K_test.pkl'
    if s1:
        from decomposer import re_calculate_prop_by_smiles
        # re_calculate_prop_by_smiles(train_path)
        # re_calculate_prop_by_smiles(test_path)

    return get_frag_dataloader_without_split(token_fun=token_fun_2, tokenizer_=tokenizer__, train_file=train_path,
                                             test_file=test_path,
                                             batch_size=50, multiset=1), tokenizer__

def get_frag_default_dataloader_ZINC_refined_pkl(token_path, s1):
    tokenizer__ = tokenizer.tokenizer_from_file(file_path=token_path)
    train_path = 'gpt/frag_file/frag_decom_ZINC_refined_train.pkl'
    test_path = 'gpt/frag_file/frag_decom_ZINC_refined_test.pkl'
    if s1:
        from decomposer import re_calculate_prop_by_smiles
        # re_calculate_prop_by_smiles(train_path)
        # re_calculate_prop_by_smiles(test_path)

    return get_frag_dataloader_without_split(token_fun=token_fun_2, tokenizer_=tokenizer__, train_file=train_path,
                                             test_file=test_path,
                                             batch_size=100, multiset=1), tokenizer__
