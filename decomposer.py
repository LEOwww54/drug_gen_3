from decompose.base import _mol_decom_mp, _mol_decom_mp_to_pkl_file, _re_calculate_prop_by_smiles
from utils import _process_list_parallel
import pickle


def mol_decom_mp(smiles, n_core, output_format='pkl', output_path=[], version=1, properties=None, stat_only=False):
    sentences, frags, oring, props, frags_stat = _mol_decom_mp(smiles, n_core, properties=properties, statistic_only=stat_only)
    if stat_only:
        return sentences, frags, oring, props, frags_stat

    if output_format == 'pkl':
        _mol_decom_mp_to_pkl_file(sentences=sentences, oring=oring, props=props, pkl_path=output_path[0])

    return sentences, frags, oring, props, frags_stat

def re_calculate_prop_by_smiles(pkl_path):
    data = pickle.load(open(pkl_path, 'rb'))
    mol_index = [k for k, x in data['mol'].items()]
    smiles = [x['oring'] for k, x in data['mol'].items()]

    print('re-calculating mol properties...')
    new_props = _process_list_parallel(smiles, 4, _re_calculate_prop_by_smiles)

    print('re-writing mol properties...')
    for i in mol_index:
        data['mol'][i]['props'] = new_props[i]

    pickle.dump(data, open(pkl_path, 'wb'))
