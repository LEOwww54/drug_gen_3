
from ZINC_250K import *
from gpt.default_dataset import *
import gpt.test as gpt


from torch.utils.data import DataLoader


def train_fragGPT_pamf(source, epoch, s=False, *, batch_size=50, lr=8e-5,
                        p_type='Riemannian', conditional=('prop',), output_dir=None):
    """Train on existing PAMF train/test PKLs. s=True rebuilds TRAIN-only vocab.

    No automatic decomposition or loading of a legacy-vocabulary checkpoint.
    Default condition is the existing QED/logP/SA property vector; choose
    conditional=('unconditional',) for unconditional training.
    """
    import json
    import math
    from pathlib import Path
    from gpt.tokenizer import pamf_training_paths, get_new_pamf_tokenizer
    if type(epoch) is not int or epoch < 1:
        raise ValueError('epoch must be a positive integer')
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError('lr must be finite and positive')
    conditional = tuple(conditional)
    if conditional not in (('prop',), ('unconditional',)):
        raise ValueError('PAMF training supports prop or unconditional')
    if p_type not in ('Riemannian', 'lora_1'):
        raise ValueError('p_type must be Riemannian or lora_1')
    paths = pamf_training_paths(source)
    for split in ('train', 'test'):
        if not paths[split].is_file():
            raise FileNotFoundError(f'Missing PAMF {split} PKL: {paths[split]}. Run the dataset PAMF dataprocess function first.')
    if s or not paths['tokenizer'].is_file():
        get_new_pamf_tokenizer(source)
    loaders, tokenizer_ = get_frag_default_dataloader_pamf(
        source, batch_size=batch_size, require_properties='prop' in conditional)
    destination = Path(output_dir) if output_dir is not None else paths['checkpoint']/f'{conditional[0]}_{p_type}'
    destination.mkdir(parents=True, exist_ok=True)
    tokenizer_.save(str(destination/'frag_tokenizer.json'))
    (destination/'training_config.json').write_text(json.dumps(
        dict(source=source, decomposition='pamf', epochs=epoch, lr=lr,
             batch_size=batch_size, p_type=p_type, conditional=conditional,
             train_file=str(paths['train']), test_file=str(paths['test']),
             vocab_size=tokenizer_.get_vocab_size()), ensure_ascii=False, indent=2), encoding='utf-8')
    # Supplying a fresh model bypasses gpt.train's implicit legacy checkpoint load.
    model = gpt.GPT(vocab_size=tokenizer_.get_vocab_size(), p_type=p_type, conditional=conditional)
    return gpt.train(loaders, epoch, tokenizer_.get_vocab_size(), lr, model=model,
                     p_type=p_type, conditional=conditional, output_dir=destination)


def train_fragGPT_ZINC_250K_pamf(epoch, s=False, **kwargs):
    return train_fragGPT_pamf('ZINC_250K', epoch, s, **kwargs)


def train_fragGPT_ZINC_refined_pamf(epoch, s=False, **kwargs):
    kwargs.setdefault('batch_size', 100)
    return train_fragGPT_pamf('ZINC_refined', epoch, s, **kwargs)

def train_fragGPT_ZINC_250K_unconditional_lora_1(epoch, s, s1):
    if s:
        import gpt.tokenizer as tokenizer_gpt
        tokenizer_gpt.get_new_tokenizer(source='ZINC_250K')
    data_loaders, tokenizer__ = get_frag_default_dataloader_ZINC_250K_pkl('gpt/vocab/frag_tokenizer_ZINC_250K.json', s1)
    lr = 1e-4

    gpt.train(data_loaders, epoch, tokenizer__.get_vocab_size(), lr, p_type='lora_1', conditional=['unconditional'])

def train_fragGPT_ZINC_250K_prop_Riemmanian(epoch, s, s1):
    if s:
        import gpt.tokenizer as tokenizer_gpt
        tokenizer_gpt.get_new_tokenizer(source='ZINC_250K')
    data_loaders, tokenizer__ = get_frag_default_dataloader_ZINC_250K_pkl('gpt/vocab/frag_tokenizer_ZINC_250K.json', s1)
    lr = 8e-5


    gpt.train(data_loaders, epoch, tokenizer__.get_vocab_size(), lr, p_type='Riemannian', conditional=['prop'])

def train_fragGPT_ZINC_250K_prop_lora_1(epoch, s, s1):
    if s:
        import gpt.tokenizer as tokenizer_gpt
        tokenizer_gpt.get_new_tokenizer(source='ZINC_250K')
    data_loaders, tokenizer__ = get_frag_default_dataloader_ZINC_250K_pkl('gpt/vocab/frag_tokenizer_ZINC_250K.json', s1)
    lr = 1e-4

    gpt.train(data_loaders, epoch, tokenizer__.get_vocab_size(), lr, p_type='lora_1', conditional=['prop'])

def finetune_fragGPT_ZINC_250K_unconditional_lora_1(epoch, s, s1):
    if s:
        import gpt.tokenizer as tokenizer_gpt
        tokenizer_gpt.get_new_tokenizer(source='ZINC_250K')
    data_loaders, tokenizer__ = get_frag_default_dataloader_ZINC_250K_pkl('checkpoints/fragGPT/T1_ZINC_250K_unconditional/frag_tokenizer.json', s1)
    lr = 9e-5
    import  torch
    model = gpt.GPT(vocab_size=tokenizer__.get_vocab_size(),p_type='lora_1', conditional=['unconditional'])
    model.load_state_dict(torch.load('checkpoints/fragGPT/T1_ZINC_250K_unconditional/GPT.pt'), strict=False)

    gpt.train(data_loaders, epoch, tokenizer__.get_vocab_size(), lr, p_type='lora_1', conditional=['unconditional'], model=model)

def train_fragGPT_ZINC_refined_prop_Riemmanian(epoch, s, s1):
    if s:
        import gpt.tokenizer as tokenizer_gpt
        tokenizer_gpt.get_new_tokenizer(source='ZINC_refined')
    data_loaders, tokenizer__ = get_frag_default_dataloader_ZINC_refined_pkl('gpt/vocab/frag_tokenizer_ZINC_refined.json', s1)
    lr = 9e-5

    gpt.train(data_loaders, epoch, tokenizer__.get_vocab_size(), lr, p_type='Riemannian', conditional=['prop'])

if __name__ == '__main__':
    ## md_test()
    # train_fragGPT_chembl_unconditional_lora_1(2, True, False)
    train_fragGPT_ZINC_250K_pamf(5, False)

    # finetune_alot_ZINC_geom(epoch=8, s=True)



    print()
