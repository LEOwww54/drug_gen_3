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
    smiles = decompose('[H][C@@]12C[C@H](O)[C@@]3(C)C(=O)[C@H](OC(C)=O)C4=C(C)[C@H](C[C@@](O)([C@@H](OC(=O)c5cc([1*])ccc5)[C@]3([H])[C@@]1(CO2)OC(C)=O)C4(C)C)OC(=O)[C@H](O)[C@@H](NC(=O)c6ccccc6)c7ccccc7')

    pass