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
    import os
    import graphviz

    # 如果需要，请取消注释并修改为你电脑上实际的 Graphviz bin 目录路径
    os.environ["PATH"] += os.pathsep + r'C:\Program Files\Graphviz\bin'

    source_code = """// FST Molecular Generation Workflow
digraph FST_Workflow {
    fontname="Arial" 
    fontsize=12 
    nodesep=0.4 
    rankdir=TB 
    ranksep=0.5
    
    // 全局节点与边样式设定
    node [fontname="Arial" fontsize=11 shape=box style=filled]
    edge [arrowhead=vee color="#4A5568" fontname="Arial" fontsize=10]

    // ==========================================
    // A. Functional Substructure Tokenization (FST)
    // ==========================================
    subgraph cluster_A {
        color="#DCE6F1" 
        fillcolor="#EBF2FA" 
        fontname="Arial-Bold" 
        fontsize=14 
        label="A. Functional Substructure Tokenization (FST)" 
        style="filled,rounded"
        
        // 强制 A 模块内部水平排布
        { rank=same A1 A2 A3 A4 A5 }

        A1 [shape=box style="filled,rounded" color="#4682B4" fillcolor="#D3E2F2" label="Data Collection\n& Cleaning\n\n[ZINC, ChEMBL,\nGEOM, BindingDB]"]
        
        A2 [shape=box style="filled,rounded" color="#A9CCE3" fillcolor="#EAF2F8" label="Molecule\nDecomposition\n\n[Molecular Fragments]"]
        
        // 修正 A3：将 shape=plaintext 移到最前，彻底隔离 HTML 标签
        A3 [shape=plaintext label=<
            <table border="0" cellborder="1" cellspacing="4" cellpadding="6">
                <tr><td bgcolor="#EDF6F9" bordercolor="#A8DADC"><b>Maximum Ring Matching</b><br/>Identify ring structures</td></tr>
                <tr><td bgcolor="#FEFAE0" bordercolor="#E9C46A"><b>MACCS Fingerprints - Chemically<br/>Significant Substructures</b> (128 sub-structures)</td></tr>
                <tr><td bgcolor="#FFE5D9" bordercolor="#F4A261"><b>Individual Specific Substructure</b><br/>Identify linkers or chain segments</td></tr>
            </table>
        >]
        
        // 修正 A4：属性统一前置，内部嵌入嵌套表格
        A4 [shape=box style="filled,rounded" color="#81C784" fillcolor="#E8F5E9" label=<
            Connection Handling<br/><b>&amp; Tokenization</b><br/><br/>
            Form connection processing,<br/>
            atom recognition,<br/>
            ring system notifications<br/>
            Sentences e.g.<br/>
            <table border="0" cellborder="1" cellspacing="6" cellpadding="6" style="rounded">
                <tr><td bgcolor="#FFFFFF" bordercolor="#495057">{ atom [C] &lt;m- 1&gt; = atom [O] }</td></tr>
                <tr><td bgcolor="#FFFFFF" bordercolor="#495057">{ atom [C] ( = atom [C] &lt;m- 1&gt; &lt;m- 2&gt; ) atom [O] }</td></tr>
                <tr><td bgcolor="#FFFFFF" bordercolor="#495057">{ atom [C] ( atom [N] ) # atom [C] &lt;m- 2&gt; }</td></tr>
            </table>
        >]
        
        A5 [shape=box style="filled,rounded" color="#73C6B6" fillcolor="#E8F8F5" label=<
            <b>Sentence<br/>formation</b><br/><br/>
            Complete molecular<br/>sequences<br/><br/>
            { atom [C] &lt;m- 1&gt; = atom [O] }<br/>
            { atom [C] ( = atom [C] &lt;m- 1&gt; ) atom [O] }<br/>
            { atom [C] ( atom [N] ) # atom [C] &lt;m- 2&gt; }
        >]

        A1 -> A2
        A2 -> A3
        A3 -> A4
        A4 -> A5
    }

    // 中间过渡桥梁
    Bridge [shape=none fontname="Arial-Bold" fontsize=12 label="Serialized\nMolecular\nTokens"]

    // ==========================================
    // B. Training and generation of GPT generative model
    // ==========================================
    subgraph cluster_B {
        color="#E8F8F5" 
        fillcolor="#F4FBF7" 
        fontname="Arial-Bold" 
        fontsize=14 
        label="B. Training and generation of GPT generative model" 
        style="filled,rounded"
        
        B1 [shape=box style="filled,rounded" color="#27AE60" fillcolor="#A9DFBF" label="Embedding &\nPositional\nEncoding"]
        
        B_TF1 [shape=plaintext label=<
            <table border="0" cellborder="1" cellspacing="2" cellpadding="4" bgcolor="#FFFFFF">
                <tr><td bgcolor="#FCF3CF">Add &amp; Norm</td></tr>
                <tr><td bgcolor="#D6EAF8">Feed Forward</td></tr>
                <tr><td bgcolor="#FCF3CF">Add &amp; Norm</td></tr>
                <tr><td bgcolor="#EDBB99">Multi-head Self-attention</td></tr>
            </table>
        >]
        
        B_Dots [shape=none fontname="Arial-Bold" fontsize=16 label="..."]
        
        B_TF2 [shape=plaintext label=<
            <table border="0" cellborder="1" cellspacing="2" cellpadding="4" bgcolor="#FFFFFF">
                <tr><td bgcolor="#FCF3CF">Add &amp; Norm</td></tr>
                <tr><td bgcolor="#D6EAF8">Feed Forward</td></tr>
                <tr><td bgcolor="#FCF3CF">Add &amp; Norm</td></tr>
                <tr><td bgcolor="#EDBB99">Multi-head Self-attention</td></tr>
            </table>
        >]
        
        B_Out [shape=none label="Sentence"]
        
        B1 -> B_TF1
        B_TF1 -> B_Dots
        B_Dots -> B_TF2
        B_TF2 -> B_Out
    }

    // ==========================================
    // C. Molecule generation
    // ==========================================
    subgraph cluster_C {
        color="#FDEBD0" 
        fillcolor="#FEF9E7" 
        fontname="Arial-Bold" 
        fontsize=14 
        label="C. Molecule generation" 
        style="filled,rounded"
        
        C1 [shape=box style="filled,rounded" color="#E67E22" fillcolor="#F8C471" label="Molecule\nReconstruction\nAlgorithm\n\nDecode back into\nchemical structures"]
        C2 [shape=none label="Generated Results\n\n[2D Molecular Structures]"]
        
        C1 -> C2
    }

    // ==========================================
    // 跨集群全局核心连线
    // ==========================================
    A5 -> Bridge [ltail=cluster_A minlen=2]
    Bridge -> B1 [lhead=cluster_B]
    B_Out -> C1 [label=" Output " lhead=cluster_C ltail=cluster_B]
}"""
    gv_file = graphviz.Source(source_code)
    gv_file.render('nature_molecular_workflow', format='svg', view=True)
