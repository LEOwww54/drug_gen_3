from ZINC.dataloader import data_from_ZINC_250K
from decomposer import mol_decom_mp

def mol_decomp_mp_ZINC_250K_pkl(n_core, stat_mode = False):
    train, test, all = data_from_ZINC_250K()
    path = 'gpt/frag_file/frag_decom_ZINC_250K_train.pkl'
    path2 = 'gpt/frag_file/frag_decom_ZINC_250K_test.pkl'

    result = \
    mol_decom_mp(smiles=train, n_core=n_core, output_format='pkl', output_path=[path], stat_only=stat_mode)[1]
    # #result1 = mol_decom_mp(smiles=smiles, n_core=n_core, output_format='pkl', output_path=[path1], stat_only=stat_mode)
    result2 = \
    mol_decom_mp(smiles=test, n_core=n_core, output_format='pkl', output_path=[path2], stat_only=stat_mode)[1]

    result.extend(result2)

    return result