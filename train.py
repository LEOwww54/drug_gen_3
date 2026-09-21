
from ZINC_250K import *
from gpt.default_dataset import *
import gpt.test as gpt


from torch.utils.data import DataLoader


def _training_tokenizer_path(default_path, checkpoint_dir):
    if checkpoint_dir is None:
        return default_path
    return gpt.checkpoint_artifact_paths(checkpoint_dir)['tokenizer']


def train_fragGPT_pamf(source, epoch, s=False, *, batch_size=50, lr=8e-5,
                        p_type='Riemannian', conditional=('prop',), output_dir=None,
                        model=None, checkpoint_dir=None):
    """Train on existing PAMF train/test PKLs. s=True rebuilds TRAIN-only vocab.

    Pass checkpoint_dir to continue from that directory's GPT.pt.
    Default condition is the existing QED/logP/SA property vector; choose
    conditional=('unconditional',) for unconditional training.
    """
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
    if checkpoint_dir is None and (s or not paths['tokenizer'].is_file()):
        get_new_pamf_tokenizer(source)
    token_path = _training_tokenizer_path(paths['tokenizer'], checkpoint_dir)
    loaders, tokenizer_ = get_frag_default_dataloader_pamf(
        source, token_path=token_path, batch_size=batch_size,
        require_properties='prop' in conditional)
    destination = Path(output_dir) if output_dir is not None else paths['checkpoint']
    config = dict(source=source, decomposition='pamf', batch_size=batch_size,
                  train_file=str(paths['train']), test_file=str(paths['test']))
    if model == 'new':
        model = None
    return gpt.train(loaders, epoch, tokenizer_.get_vocab_size(), lr, model=model,
                     p_type=p_type, conditional=conditional, output_dir=destination,
                     tokenizer=tokenizer_, training_config=config,
                     checkpoint_dir=checkpoint_dir)


def train_fragGPT_ZINC_250K_pamf(epoch, s=False, checkpoint_dir=None, **kwargs):
    return train_fragGPT_pamf(
        'ZINC_250K', epoch, s, checkpoint_dir=checkpoint_dir, **kwargs)


def train_fragGPT_ZINC_refined_pamf(epoch, s=False, checkpoint_dir=None, **kwargs):
    kwargs.setdefault('batch_size', 50)
    return train_fragGPT_pamf(
        'ZINC_refined', epoch, s, checkpoint_dir=checkpoint_dir, **kwargs)


def train_fragGPT_frag_pretrain(source, epoch, s=False, *, batch_size=50, lr=8e-5,
                                p_type='Riemannian', output_dir=None, checkpoint_dir=None,
                                reuse_pamf_vocab=True, base_tokenizer_path=None,
                                vocab_only=None):
    """Pretrain GPT, optionally continuing from checkpoint_dir/GPT.pt."""
    import math
    from pathlib import Path
    from gpt.frag_pretrain_dataprocess import frag_pretrain_paths, normalize_frag_pretrain_source
    from gpt.tokenizer import get_new_frag_pretrain_tokenizer

    if type(epoch) is not int or epoch < 1:
        raise ValueError('epoch must be a positive integer')
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('batch_size must be a positive integer')
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError('lr must be finite and positive')
    if p_type not in ('Riemannian', 'lora_1'):
        raise ValueError('p_type must be Riemannian or lora_1')
    if vocab_only is not None and not isinstance(vocab_only, bool):
        raise ValueError('vocab_only must be boolean or None')
    source = normalize_frag_pretrain_source(source)
    paths = frag_pretrain_paths(source)
    for split in ('train', 'test'):
        if not paths[split].is_file():
            raise FileNotFoundError(
                f'Missing fragment-pretraining {split} PKL: {paths[split]}. '
                f'Run prepare_frag_pretrain_data({source!r}) first.')
    if checkpoint_dir is None and (s or not paths['tokenizer'].is_file()):
        get_new_frag_pretrain_tokenizer(
            source, reuse_pamf_vocab=reuse_pamf_vocab,
            base_tokenizer_path=base_tokenizer_path,
            pamf_vocab_only=vocab_only)
    token_path = _training_tokenizer_path(paths['tokenizer'], checkpoint_dir)
    loaders, tokenizer_ = get_frag_pretrain_default_dataloader(
        source, token_path=token_path, batch_size=batch_size)
    conditional = ('unconditional',)
    destination = Path(output_dir) if output_dir is not None else paths['checkpoint']
    config = dict(source=source, dataset=f'{source}_frag_pretrain',
                  representation='single_fragment_sisr', frequency_expanded=True,
                  tokenizer_reuses_pamf_vocab=reuse_pamf_vocab,
                  vocab_only=vocab_only,
                  batch_size=batch_size, train_file=str(paths['train']),
                  test_file=str(paths['test']))
    return gpt.train(loaders, epoch, tokenizer_.get_vocab_size(), lr,
                     p_type=p_type, conditional=conditional, output_dir=destination,
                     tokenizer=tokenizer_, training_config=config,
                     checkpoint_dir=checkpoint_dir)


def train_fragGPT_ZINC_250K_frag_pretrain(epoch, s=False, checkpoint_dir=None, **kwargs):
    return train_fragGPT_frag_pretrain(
        'ZINC_250K', epoch, s, checkpoint_dir=checkpoint_dir, **kwargs)


def train_fragGPT_ZINC_refined_frag_pretrain(epoch, s=False, checkpoint_dir=None, **kwargs):
    kwargs.setdefault('batch_size', 100)
    return train_fragGPT_frag_pretrain(
        'ZINC_refined', epoch, s, checkpoint_dir=checkpoint_dir, **kwargs)


def train_fragGPT_ZIINC_refined_frag_pretrain(epoch, s=False, checkpoint_dir=None, **kwargs):
    """Backward-compatible alias for the spelling used in the request."""
    return train_fragGPT_ZINC_refined_frag_pretrain(
        epoch, s, checkpoint_dir=checkpoint_dir, **kwargs)

def train_fragGPT_ZINC_250K_unconditional_lora_1(epoch, s, s1, checkpoint_dir=None):
    if s and checkpoint_dir is None:
        import gpt.tokenizer as tokenizer_gpt
        tokenizer_gpt.get_new_tokenizer(source='ZINC_250K')
    token_path = _training_tokenizer_path(
        'gpt/vocab/frag_tokenizer_ZINC_250K.json', checkpoint_dir)
    data_loaders, tokenizer__ = get_frag_default_dataloader_ZINC_250K_pkl(token_path, s1)
    lr = 1e-4

    return gpt.train(data_loaders, epoch, tokenizer__.get_vocab_size(), lr,
                     p_type='lora_1', conditional=['unconditional'],
                     checkpoint_dir=checkpoint_dir, tokenizer=tokenizer__)

def train_fragGPT_ZINC_250K_prop_Riemmanian(epoch, s, s1, checkpoint_dir=None):
    if s and checkpoint_dir is None:
        import gpt.tokenizer as tokenizer_gpt
        tokenizer_gpt.get_new_tokenizer(source='ZINC_250K')
    token_path = _training_tokenizer_path(
        'gpt/vocab/frag_tokenizer_ZINC_250K.json', checkpoint_dir)
    data_loaders, tokenizer__ = get_frag_default_dataloader_ZINC_250K_pkl(token_path, s1)
    lr = 8e-5


    return gpt.train(data_loaders, epoch, tokenizer__.get_vocab_size(), lr,
                     p_type='Riemannian', conditional=['prop'],
                     checkpoint_dir=checkpoint_dir, tokenizer=tokenizer__)

def train_fragGPT_ZINC_250K_prop_lora_1(epoch, s, s1, checkpoint_dir=None):
    if s and checkpoint_dir is None:
        import gpt.tokenizer as tokenizer_gpt
        tokenizer_gpt.get_new_tokenizer(source='ZINC_250K')
    token_path = _training_tokenizer_path(
        'gpt/vocab/frag_tokenizer_ZINC_250K.json', checkpoint_dir)
    data_loaders, tokenizer__ = get_frag_default_dataloader_ZINC_250K_pkl(token_path, s1)
    lr = 1e-4

    return gpt.train(data_loaders, epoch, tokenizer__.get_vocab_size(), lr,
                     p_type='lora_1', conditional=['prop'],
                     checkpoint_dir=checkpoint_dir, tokenizer=tokenizer__)

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

def train_fragGPT_ZINC_refined_prop_Riemmanian(epoch, s, s1, checkpoint_dir=None):
    if s and checkpoint_dir is None:
        import gpt.tokenizer as tokenizer_gpt
        tokenizer_gpt.get_new_tokenizer(source='ZINC_refined')
    token_path = _training_tokenizer_path(
        'gpt/vocab/frag_tokenizer_ZINC_refined.json', checkpoint_dir)
    data_loaders, tokenizer__ = get_frag_default_dataloader_ZINC_refined_pkl(token_path, s1)
    lr = 9e-5

    return gpt.train(data_loaders, epoch, tokenizer__.get_vocab_size(), lr,
                     p_type='Riemannian', conditional=['prop'],
                     checkpoint_dir=checkpoint_dir, tokenizer=tokenizer__)

if __name__ == '__main__':
    ## md_test()
    # train_fragGPT_chembl_unconditional_lora_1(2, True, False)
    #train_fragGPT_ZINC_250K_prop_Riemmanian(5, True, False)
    train_fragGPT_ZINC_250K_pamf(80, True, batch_size=50, lr=9e-5, conditional=('unconditional',))
    # train_fragGPT_ZINC_refined_pamf(20, True, batch_size=50, lr=8e-5)


    # finetune_alot_ZINC_geom(epoch=8, s=True)



    print()
