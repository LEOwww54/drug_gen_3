from decompose.molDecom_2 import VirtualAtomConnectionProcessor
from test3 import decompose_smiles

def decompose(smiles):
    frags = decompose_smiles(smiles)
    p = VirtualAtomConnectionProcessor()
    results = []
    for frag in frags:
        result = p.process_smiles(frag)
        results.append(result)
    return results

if __name__ == '__main__':
    smiles = decompose("CC[C@H]1OC(=O)[C@H](C)[C@@H](OC2=CC=C(C=C2)[C@H]3NC(=O)[C@H](NC(=O)[C@H](CC(=O)N)NC(=O)[C@H]4C5=C(C(=CC(=C5)O)O)C6=C(C(=C(C=C6[C@H](NC3=O)C(=O)N4)O)O)Cl)C(C)C)OC1=O"
                       )

    smiles = decompose("CC(=O)O[C@H]1[C@H]2[C@@]([C@H]3[C@@]([C@]4(C[C@@H]5[C@]6(C[C@@H](C(=C([C@@H](O6)C(=O)[C@]5(C4=C(C3=O)C)OC(=O)C)OC(=O)c7ccccc7)O)C)OC(=O)C)O2)OC(=O)c8ccccc8)(C1(C)C)OC(=O)C")
    pass