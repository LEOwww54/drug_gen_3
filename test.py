from ZINC_250K import dataloader
import pickle as pkl
from gpt.tokenizer import get_new_tokenizer
from decompose.molConn import gen2mol
from decomposer import mol_decom_mp

def t1():
    x = pkl.load(open('gpt/frag_file/frag_decom_ZINC_250K_test.pkl', 'rb'))
    mols = [i['frag'] for t, i in x['mol'].items()]
    smiles = [i['oring'] for t, i in x['mol'].items()]
    mols = [(i['frag'], i['oring']) for t, i in x['mol'].items()]
    r = gen2mol(mols[:1000])
    r = [i[1] for i in r]
    print(r)

def t2():
    get_new_tokenizer('ZINC_250K')

def t3():
    result = mol_decom_mp(['CC1(C)CCCC[C@H]1[NH2+]Cc1c[nH]cn1'], n_core=1, output_path='tmp')
    r = gen2mol(result[0])
    return result

def t4():
    s = ['<start> { ^atom^ [C] <fc0> <m- 1> <m- 2> <sym0> <r> ( ^atom^ [CH2] <fc0> <sym1> <r> <r1> ) ( ^atom^ [CH] <fc0> <m- 3> <sym5> <r> ^atom^ [CH2] <fc0> <sym4> <r> ^atom^ [CH2] <fc0> <sym3> <r> ^atom^ [CH2] <fc0> <sym2> <r> <r1> ) } { ^atom^ [C] <fc0> <m- 4> <sym0> <r> ( = ^atom^ [CH] <fc0> <sym1> <r> <r1> ) ( ^atom^ [N] <fc0> <sym4> <r> = ^atom^ [CH] <fc0> <sym3> <r> ^atom^ [NH] <fc0> <sym2> <r> <r1> ) } { ^atom^ [NH2] <fc1> <m- 3> <sym0> ^atom^ [CH2] <fc0> <m- 4> <sym1> } { ^atom^ [CH3] <fc0> <m- 1> <sym0> } { ^atom^ [CH3] <fc0> <m- 2> <sym0> } </s>']
    r = gen2mol(s)
    return r

if "__main__" == __name__:
    t2()
    t1()
