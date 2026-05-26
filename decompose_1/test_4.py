from decompose.molDecom_2 import VirtualAtomConnectionProcessor
from test3 import decompose_test

def decompose(smiles):
    frags = decompose_test([smiles])
    p = VirtualAtomConnectionProcessor()
    results = []
    for frag in frags[0]:
        result = p.process_smiles(frag)
        results.append(result)
    return results

if __name__ == '__main__':
    smiles = decompose('CC1(C)S[C@@H]2[C@H](NC(=O)Cc3ccccc3)C(=O)N2C1C(=O)O')

    pass