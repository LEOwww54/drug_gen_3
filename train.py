
from ZINC_250K import *
from gpt.dataset import *
import gpt.test as gpt


from torch.utils.data import DataLoader

vocal_size = vocab_size

def train_fragGPT_ZINC_250K_unconditional_lora_1(epoch, s, s1):
    if s:
        import gpt.tokenizer as tokenizer_gpt
        tokenizer_gpt.get_new_tokenizer(source='ZINC_250K')
    data_loaders, tokenizer__ = get_frag_default_dataloader_ZINC_250K_pkl('gpt/vocab/frag_tokenizer_ZINC_250K.json', s1)
    lr = 5e-4

    gpt.train(data_loaders, epoch, tokenizer__.get_vocab_size(), lr, p_type='lora_1', conditional=['unconditional'])

def train_fragGPT_ZINC_250K_prop_Riemmanian(epoch, s, s1):
    if s:
        import gpt.tokenizer as tokenizer_gpt
        tokenizer_gpt.get_new_tokenizer(source='ZINC_250K')
    data_loaders, tokenizer__ = get_frag_default_dataloader_ZINC_250K_pkl('gpt/vocab/frag_tokenizer_ZINC_250K.json', s1)
    lr = 5e-4

    gpt.train(data_loaders, epoch, tokenizer__.get_vocab_size(), lr, p_type='Riemannian', conditional=['prop'])

if __name__ == '__main__':
    ## md_test()
    # train_fragGPT_chembl_unconditional_lora_1(2, True, False)
    train_fragGPT_ZINC_250K_prop_Riemmanian(10, False, False)

    # finetune_alot_ZINC_geom(epoch=8, s=True)



    print()
