from dataprocess import *
from ZINC import *
from gpt.dataset import *
from gpt.dataset_ori import CommonDataSet
import gpt.test as gpt
from torch.utils.data import DataLoader

vocal_size = vocab_size

def train_fragGPT_ZINC(epoch, s):
    if s:
        import gpt.tokenizer as tokenizer_gpt
        tokenizer_gpt.get_new_tokenizer(0, source='ZINC')
    data_loaders, tokenizer__ = get_frag_default_dataloader_ZINC_test()
    lr = 2e-4

    gpt.train(data_loaders, epoch, tokenizer__.get_vocab_size(), lr)

def fine_tune_fragGPT_PL(epoch, s):
    gpt.fine_tune_fragGPT(epoch, 'checkpoints/test/GPT.pt', plk_path='checkpoints/test/protein.plk', tokenizer_path='checkpoints/test/frag_tokenizer.json')

def fine_tune_fragGPT_PL_molonly(epoch, s):
    gpt.fine_tune_fragGPT_molonly(epoch=epoch, model_path='checkpoints/fragGPT/C10_ZINC/GPT.pt', tokenizer_path='checkpoints/fragGPT/C10_ZINC/frag_tokenizer.json', train_path='gpt/PL_smiles.txt')

def train_fragGPT_ZINC_chembl(epoch, s):
    if s:
        import gpt.tokenizer as tokenizer_gpt
        tokenizer_gpt.get_new_tokenizer(0, source='ZINC_chembl')
    data_loaders, tokenizer__ = get_frag_default_dataloader_ZINC_test(token_path='gpt/vocab/frag_tokenizer_ZINC_chembl.json')
    lr = 2e-4

    gpt.train(data_loaders, epoch, tokenizer__.get_vocab_size(), lr)

def train_fragGPT_chembl(epoch, s):
    if s:
        import gpt.tokenizer as tokenizer_gpt
        tokenizer_gpt.get_new_tokenizer(0, source='chembl')
    data_loaders, tokenizer__ = get_frag_default_dataloader_chembl_test(token_path='gpt/vocab/frag_tokenizer_chembl.json')
    lr = 2e-4

    gpt.train(data_loaders, epoch, tokenizer__.get_vocab_size(), lr)

def train_fragGPT_chembl_refined(epoch, s):
    if s:
        import gpt.tokenizer as tokenizer_gpt
        tokenizer_gpt.get_new_tokenizer(0, source='chembl_refined')
    data_loaders, tokenizer__ = get_frag_default_dataloader_chembl_refined(token_path='gpt/vocab/frag_tokenizer_chembl_refined.json')
    lr = 2e-4

    gpt.train(data_loaders, epoch, tokenizer__.get_vocab_size(), lr)

def train_fragGPT_chembl_qed(epoch, s):
    if s:
        import gpt.tokenizer as tokenizer_gpt
        tokenizer_gpt.get_new_tokenizer(0, source='chembl_qed')
    data_loaders, tokenizer__ = get_frag_default_dataloader_chembl_qed(token_path='gpt/vocab/frag_tokenizer_chembl_qed.json')
    lr = 2e-4

    gpt.train(data_loaders, epoch, tokenizer__.get_vocab_size(), lr)

def md_test():
    # smiles = data_from_ZINC_refined(n=10000)[1]
    smiles = ['Cc1cc(-n2ncc(=O)[nH]c2=O)ccc1C(=O)c1ccccc1']
    smiles1 = []
    from rdkit import Chem
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        Chem.Kekulize(mol, True)
        smiles1.append(Chem.MolToSmiles(mol))

    result, oring = mol_decom_mp(smiles=smiles1, n_core=4, return_smiles=True, frag_output_file='tmp')
    from molConn import gen2mol, mol_translate
    import rdkit.Chem as Chem
    recon_smiles = []
    for i in range(len(result)):
        mol = gen2mol(['\t'.join(result[i])])[0][1]

        recon_smiles.append((mol, result[i]))

    count = 0
    for i in range(len(recon_smiles)):
        if not recon_smiles[i][0] == oring[i]:
            count = count + 1
            print(f'{i}: oring:{oring[i]}--------recon:{recon_smiles[i][0]}')
        else:
            print(recon_smiles[i][0])

def train_fragGPT_ZINC_pkl(epoch, s, s1):
    if s:
        import gpt.tokenizer as tokenizer_gpt
        tokenizer_gpt.get_new_tokenizer(0, source='ZINC_pkl')
    data_loaders, tokenizer__ = get_frag_default_dataloader_ZINC_pkl('gpt/vocab/frag_tokenizer_ZINC_pkl.json', s1)
    lr = 2e-4

    gpt.train(data_loaders, epoch, tokenizer__.get_vocab_size(), lr)

def finetune_fragGPT_ZINC_pkl_lora(epoch, path):
    data_loaders, tokenizer__ = get_frag_default_dataloader_ZINC_pkl(path + '/frag_tokenizer.json', False)
    lr = 2e-4
    gpt.fine_tune(model_path=path + '/GPT.pt',epochs=epoch, data_loader=data_loaders, vs=tokenizer__.get_vocab_size(), lr=lr, p_type='lora_0')

def train_fragGPT_ZINC_pkl_lora_1(epoch, s, s1):
    if s:
        import gpt.tokenizer as tokenizer_gpt
        tokenizer_gpt.get_new_tokenizer(0, source='ZINC_pkl')
    data_loaders, tokenizer__ = get_frag_default_dataloader_ZINC_pkl('gpt/vocab/frag_tokenizer_ZINC_pkl.json', s1)
    lr = 2e-4

    gpt.train(data_loaders, epoch, tokenizer__.get_vocab_size(), lr, p_type='lora_1', conditional=['prop'])

def finetune_fragGPT_ZINC_pkl_lora_1(epoch, path, s1):
    data_loaders, tokenizer__ = get_frag_default_dataloader_ZINC_pkl(path + '/frag_tokenizer.json', s1)
    lr = 1e-4
    gpt.fine_tune(model_path=path + '/GPT.pt',epochs=epoch, data_loader=data_loaders, vs=tokenizer__.get_vocab_size(), lr=lr, p_type='lora_1')

def train_GPT_BPE_ZINC_refined(epoch, s = True):
    from gpt.BPE import BPETokenizerManager
    vs = 30000
    bpe_manager = BPETokenizerManager(vocab_size=vs)

    smiles, all_smiles = data_from_ZINC_refined(0)
    train_smiles = smiles['train']
    test_smiles = smiles['test']

    if s:
        bpe_manager.train_from_list(all_smiles, 'gpt/vocab/gpt_bpe.json')
    else:
        bpe_manager.load_from_file(model_path='gpt/vocab/gpt_bpe.json')

    vs = bpe_manager.tokenizer.get_vocab_size()


    train_data = []
    test_data = []

    for s in train_smiles:
        x = bpe_manager.tokenize_text(s, return_tokens=False)
        train_data.append((None, x, None))

    for s in test_smiles:
        x = bpe_manager.tokenize_text(s, return_tokens=False)
        test_data.append((None, x, None))

    dataloader_g = []
    datas = PLDataSet(train_data)
    dataloader_train = DataLoader(datas, batch_size=100, shuffle=True,
                                  collate_fn=datas.padding_batch)
    dataloader_g.append(dataloader_train)

    datas = PLDataSet(test_data)
    dataloader_t = DataLoader(datas, batch_size=100, shuffle=True,
                                  collate_fn=datas.padding_batch)

    gpt.train((dataloader_g, None, dataloader_t), epoch, vs=vs, p_type='lora_1', lr=4e-4, conditional=['unconditional'])

def train_fragGPT_BindindDB_pkl_lora_1(epoch, s, s1):
    if s:
        import gpt.tokenizer as tokenizer_gpt
        tokenizer_gpt.get_new_tokenizer(100, source='BindingDB_pkl')
    data_loaders, tokenizer__ = get_frag_default_dataloader_BindingDB_pkl('gpt/vocab/frag_tokenizer_BindingDB_pkl.json', s1)
    lr = 5e-4

    gpt.train(data_loaders, epoch, tokenizer__.get_vocab_size(), lr, p_type='lora_1', conditional=['protein','prop'])

def train_BRICS_ZINC(epoch, s, s1=False):
    if s:
        import gpt.tokenizer as tokenizer_gpt
        tokenizer_gpt.get_new_tokenizer(0, source='BRICS_ZINC')
    data_loaders, tokenizer__ = get_frag_default_dataloader_BRICS_ZINC('gpt/vocab/frag_tokenizer_BRICS_ZINC.json', s1)
    lr = 4e-4

    gpt.train(data_loaders, epoch, tokenizer__.get_vocab_size(), lr, p_type='lora_1', conditional=['prop'])

def train_fragGPT_geom_pkl_lora_1(epoch, s, s1):
    if s:
        import gpt.tokenizer as tokenizer_gpt
        tokenizer_gpt.get_new_tokenizer(0, source='geom')
    data_loaders, tokenizer__ = get_frag_default_dataloader_geom_pkl('gpt/vocab/frag_tokenizer_geom.json', s1)
    lr = 2e-4

    gpt.train(data_loaders, epoch, tokenizer__.get_vocab_size(), lr, p_type='lora_1', conditional=['unconditional'])

def pretrain_fragGPT_DRD2_lora_1(epoch, s, s1):
    if s:
        import gpt.tokenizer as tokenizer_gpt
        tokenizer_gpt.get_new_tokenizer(0, source='DRD2_ZINC')
    data_loaders, tokenizer__ = get_frag_default_dataloader_ZINC_pkl('gpt/vocab/frag_tokenizer_DRD2_ZINC.json', s1)
    lr = 2e-4

    gpt.train(data_loaders, epoch, tokenizer__.get_vocab_size(), lr, p_type='lora_1', conditional=['unconditional'])

def finetune_fragGPT_DRD2_lora_1(epoch, s, s1):
    if s:
        import gpt.tokenizer as tokenizer_gpt
        tokenizer_gpt.get_new_tokenizer(0, source='DRD2_ZINC')

    data_loaders, tokenizer__ = get_frag_default_dataloader_DRD2_pkl('gpt/vocab/frag_tokenizer_DRD2_ZINC.json', s1)
    lr = 2e-5

    gpt.fine_tune(model_path='checkpoints/fragGPT/D5_DRD2_ZINC_pretrain_prop3_lora_1/GPT.pt',data_loader=data_loaders,epochs=epoch,
                  vs=tokenizer__.get_vocab_size(),lr=lr,p_type='lora_1',conditional=['protein'])

def finetune_fragGPT_DRD2_rACSF_lora_1(epoch, s, s1):
    if s:
        import gpt.tokenizer as tokenizer_gpt
        tokenizer_gpt.get_new_tokenizer(0, source='DRD2_ZINC_rACSF')

    data_loaders, tokenizer__ = get_frag_default_dataloader_DRD2_rACSF_pkl('checkpoints/fragGPT/D5_DRD2_ZINC_pretrain_prop3_lora_1/frag_tokenizer.json', s1)
    lr = 2e-5

    gpt.fine_tune(model_path='checkpoints/fragGPT/D5_DRD2_ZINC_pretrain_prop3_lora_1/GPT.pt',data_loader=data_loaders,epochs=epoch,
                  vs=tokenizer__.get_vocab_size(),lr=lr,p_type='lora_1',conditional=['protein', 'protein_pocket'])

def finetune_fragGPT_DRD2_vina_lora_1(epoch, s, s1):
    if s:
        import gpt.tokenizer as tokenizer_gpt
        tokenizer_gpt.get_new_tokenizer(0, source='DRD2_ZINC_vina')

    data_loaders, tokenizer__ = get_frag_default_dataloader_DRD2_vina_pkl('checkpoints/fragGPT/D5_DRD2_ZINC_pretrain_prop3_lora_1/frag_tokenizer.json', s1)
    lr = 2e-5

    gpt.fine_tune(model_path='checkpoints/fragGPT/D5_DRD2_ZINC_pretrain_prop3_lora_1/GPT.pt',data_loader=data_loaders,epochs=epoch,
                  vs=tokenizer__.get_vocab_size(),lr=lr,p_type='lora_1',conditional=['prop'])

def _finetune_fragGPT_substructure_ZINC(epoch, name, pattern, s , model_path, vocab_path):
    name = name + "_ZINC"
    if s:
        mol_decom_mp_substructure_test_ZINC(n_core=60, pattern=pattern, name=name)

    data_loaders, tokenizer__ = get_frag_default_dataloader_subtest(
        vocab_path, name)
    lr = 2e-5

    gpt.fine_tune(model_path=model_path, data_loader=data_loaders,
                  epochs=epoch,
                  vs=tokenizer__.get_vocab_size(), lr=lr, p_type='lora_1', conditional=['unconditional'], save_name=name + '.pt')

def _finetune_fragGPT_substructure_geom(epoch, name, pattern, s, model_path, vocab_path):
        name = name + "_geom"
        if s:
            mol_decom_mp_substructure_test_geom(n_core=60, pattern=pattern, name=name)

        data_loaders, tokenizer__ = get_frag_default_dataloader_subtest(
            vocab_path, name)
        lr = 2e-5

        gpt.fine_tune(model_path=model_path, data_loader=data_loaders,
                      epochs=epoch,
                      vs=tokenizer__.get_vocab_size(), lr=lr, p_type='lora_1', conditional=['unconditional'],
                      save_name=name + '.pt')

def finetune_fragGPT_1_Benzothiophene(epoch, s, model_path, vocab_path):
    _finetune_fragGPT_substructure_ZINC(epoch=epoch,
                                  name='1-Benzothiophene',
                                  pattern='C1=CC=C2SC=CC2=C1',
                                  s=s,
                                   model_path=model_path,
                                   vocab_path=vocab_path)

def finetune_alot_ZINC_geom(epoch, s):
    if s:
        tokenizer1 = tokenizer.tokenizer_from_file(file_path='gpt/vocab/frag_tokenizer_ZINC_pkl.json')
        tokenizer2 = tokenizer.tokenizer_from_file(file_path='gpt/vocab/frag_tokenizer_geom.json')
        tokenizer1.add_tokens(list(tokenizer2.get_vocab().keys()))
        tokenizer1.save('gpt/vocab/frag_tokenizer_ZINC_geom.json')

    model_path = 'checkpoints/fragGPT/D5_DRD2_ZINC_pretrain_prop3_lora_1/GPT.pt'
    vocab_path = 'gpt/vocab/frag_tokenizer_ZINC_geom.json'

    names = {
        '1H-benzimidazole': 'C1=CC=C2NC=NC2=C1',
        'Thiazole' : 'C1=CSC=N1',
        '1,3,4-oxadiazole' : 'C1=NN=CO1',
        'Imidazolidine' : 'C1CNCN1',
        'trifluoromethyl' : 'FC(F)F',
        'Cyclopropane' : 'C1CC1'
    }
    for key, value in names.items():
        _finetune_fragGPT_substructure_geom(epoch=epoch,
                                       name=key,
                                       pattern=value,
                                       s=s,
                                       model_path=model_path,
                                        vocab_path=vocab_path)

def re_train_fragGPT_ZINC_with_geom_lora_1(epoch, s, s1):
    tokenizer1 = tokenizer.tokenizer_from_file(file_path='gpt/vocab/frag_tokenizer_ZINC_pkl.json')
    tokenizer2 = tokenizer.tokenizer_from_file(file_path='gpt/vocab/frag_tokenizer_geom.json')
    tokenizer1.add_tokens(list(tokenizer2.get_vocab().keys()))
    tokenizer1.save('gpt/vocab/frag_tokenizer_ZINC_geom.json')

    data_loaders, tokenizer__ = get_frag_default_dataloader_ZINC_pkl('gpt/vocab/frag_tokenizer_ZINC_geom.json', s1)
    lr = 2e-4

    gpt.train(data_loaders, epoch, tokenizer__.get_vocab_size(), lr, p_type='lora_1', conditional=['unconditional'])

def train_fragGPT_chembl_unconditional_lora_1(epoch, s, s1):
    if s:
        import gpt.tokenizer as tokenizer_gpt
        tokenizer_gpt.get_new_tokenizer(0, source='chembl')
    data_loaders, tokenizer__ = get_frag_default_dataloader_chembl_pkl('gpt/vocab/frag_tokenizer_chembl.json', s1)
    lr = 2e-4

    gpt.train(data_loaders, epoch, tokenizer__.get_vocab_size(), lr, p_type='lora_1', conditional=['unconditional'])

def finetune_fragGPT_chembl_unconditional_lora_1(epoch, s, s1):
    data_loaders, tokenizer__ = get_frag_default_dataloader_chembl_finetune_pkl('gpt/vocab/frag_tokenizer_chembl.json', s1)
    lr = 5e-5

    gpt.fine_tune('checkpoints/fragGPT/',data_loaders, epoch, tokenizer__.get_vocab_size(), lr, p_type='lora_1', conditional=['unconditional'])

def train_fragGPT_BindingDB_smiles_unconditional_lora_1(epoch, s, s1):
    if s:
        import gpt.tokenizer as tokenizer_gpt
        tokenizer_gpt.get_new_tokenizer(0, source='BindingDB_smiles')
    data_loaders, tokenizer__ = get_frag_default_dataloader_BindingDB_smiles_pkl('gpt/vocab/frag_tokenizer_BindingDB_smiles.json', s1)
    lr = 2e-4

    gpt.train(data_loaders, epoch, tokenizer__.get_vocab_size(), lr, p_type='lora_1', conditional=['unconditional'])

def _finetune_fragGPT_chembl_ACE_smiles_unconditional_lora_1(epoch, target, s1):
    if s1:
        mol_decomp_mp_chembl_ACE_smiles(n_core=60, target=target)
    data_loaders, tokenizer__ = get_frag_default_dataloader_chembl_ACE_smiles_pkl('gpt/vocab/frag_tokenizer_chembl.json', False, target)
    lr = 2e-5

    gpt.fine_tune('checkpoints/fragGPT/', data_loaders, epoch, tokenizer__.get_vocab_size(), lr, p_type='lora_1', conditional=['unconditional'], save_name=target)

def finetune_fragGPT_chembl_ACE_smiles_unconditional_lora_1():
    targets=[
        'CHEMBL204_Ki',
        'CHEMBL214_Ki',
        'CHEMBL233_Ki',
        'CHEMBL234_Ki',
    ]

    for target in targets:
        _finetune_fragGPT_chembl_ACE_smiles_unconditional_lora_1(8, target, False)

def train_fragGPT_ZINC_250K_unconditional_lora_1(epoch, s, s1):
    if s:
        import gpt.tokenizer as tokenizer_gpt
        tokenizer_gpt.get_new_tokenizer(0, source='ZINC_250K')
    data_loaders, tokenizer__ = get_frag_default_dataloader_ZINC_250K_pkl('gpt/vocab/frag_tokenizer_ZINC_250K.json', s1)
    lr = 5e-4

    gpt.train(data_loaders, epoch, tokenizer__.get_vocab_size(), lr, p_type='lora_1', conditional=['unconditional'])

if __name__ == '__main__':
    ## md_test()
    # train_fragGPT_chembl_unconditional_lora_1(2, True, False)
    finetune_fragGPT_DRD2_rACSF_lora_1(80, True, False)

    # finetune_alot_ZINC_geom(epoch=8, s=True)



    print()
