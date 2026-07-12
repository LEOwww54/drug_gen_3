from ZINC_refined.dataloader import data_from_ZINC_refined
from decomposer import mol_decom_mp

def mol_decomp_mp_ZINC_refined(n_core):
    result, result2 = _mol_decomp_mp_ZINC_refined(n_core)

    x = result[1]
    x2 = result2[1]

    x.extend(x2)
    return x

def ZINC_refined_statistic(n_core):
    result, result2 = _mol_decomp_mp_ZINC_refined(n_core, stat_mode = True)
    x = result[4]
    x2 = result2[4]
    for key, value in x2.items():
        if key in x:
            x[key] += value
        else:
            x[key] = value

    import json
    with open('data.json', 'w', encoding='utf-8') as f:
        json.dump(x, f)

    return x

def _mol_decomp_mp_ZINC_refined(n_core, stat_mode = False):
    smiles, smiles_all = data_from_ZINC_refined(n = 0)
    path = 'gpt/frag_file/frag_decom_ZINC_refined_train.pkl'
    path1 = 'gpt/frag_file/frag_decom_ZINC_refined_test.pkl'

    result = mol_decom_mp(smiles=smiles['train'], n_core=n_core, output_format='pkl', output_path=[path], stat_only=stat_mode)
    result2 = mol_decom_mp(smiles=smiles['test'], n_core=n_core, output_format='pkl', output_path=[path1], stat_only=stat_mode)

    return result, result2