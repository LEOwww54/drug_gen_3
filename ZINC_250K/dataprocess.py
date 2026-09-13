from ZINC_250K.dataloader import data_from_ZINC_250K
from decomposer import mol_decom_mp

def mol_decomp_mp_ZINC_250K_pamf_pkl(n_core, *, pamf_config=None, pamf_reference=None):
    """PAMF train/test PKLs; return tokens in train + test input order.

    Call under an if __name__ == '__main__' guard for multiprocessing.
    pamf_config defaults to GFN2-xTB; duplicate samples retain their positions.
    """
    train, test, _ = data_from_ZINC_250K()
    results = []
    for split, smiles in (('train', train), ('test', test)):
        path = f'gpt/frag_file/frag_decom_ZINC_250K_pamf_{split}.pkl'
        result = mol_decom_mp(
            smiles=smiles, n_core=n_core, output_format='pkl', output_path=[path],
            method='pamf', pamf_config=pamf_config, pamf_reference=pamf_reference)
        results.extend(result[1])
    return results


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

def test_decompose_all():
    train, test, all = data_from_ZINC_250K()
    return mol_decom_mp(smiles=all, n_core=60, output_format='pkl', output_path=['test'])

