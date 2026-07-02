import os
import graphviz

# 如果需要，请取消注释并修改为你电脑上实际的 Graphviz bin 目录路径
os.environ["PATH"] += os.pathsep + r'C:\Program Files\Graphviz\bin'

# 创建有向图，设置 Nature 风格的全局字体和布局
dot = graphviz.Digraph('FST_Workflow', comment='FST Molecular Generation Workflow', format='svg')
dot.attr(rankdir='TB', nodesep='0.4', ranksep='0.5', fontname='Arial', fontsize='12')
dot.attr('node', fontname='Arial', fontsize='11', shape='box', style='filled')
dot.attr('edge', fontname='Arial', fontsize='10', color='#4A5568', arrowhead='vee')

# ==========================================
# A. Functional Substructure Tokenization (FST)
# ==========================================
with dot.subgraph(name='cluster_A') as a:
    a.attr(label='A. Functional Substructure Tokenization (FST)',
           style='filled,rounded', color='#DCE6F1', fillcolor='#EBF2FA', fontname='Arial-Bold', fontsize='14')

    a.attr(rank='same')

    # A1: Data Collection
    a.node('A1', 'Data Collection\n& Cleaning\n\n[ZINC, ChEMBL,\nGEOM, BindingDB]',
           fillcolor='#D3E2F2', color='#4682B4', style='filled,rounded')

    # A2: Molecule Decomposition
    a.node('A2', 'Molecule\nDecomposition\n\n[Molecular Fragments]',
           fillcolor='#EAF2F8', color='#A9CCE3', style='filled,rounded')

    # A3: 三个子结构识别并列
    sub_features = """<table border="0" cellborder="1" cellspacing="4" cellpadding="6">
        <tr><td bgcolor="#EDF6F9" bordercolor="#A8DADC"><b>Maximum Ring Matching</b><br/>Identify ring structures</td></tr>
        <tr><td bgcolor="#FEFAE0" bordercolor="#E9C46A"><b>MACCS Fingerprints - Chemically<br/>Significant Substructures</b> (128 sub-structures)</td></tr>
        <tr><td bgcolor="#FFE5D9" bordercolor="#F4A261"><b>Individual Specific Substructure</b><br/>Identify linkers or chain segments</td></tr>
    </table>"""
    a.node('A3', label=sub_features, shape='plaintext')

    # A4: Connection Handling & Tokenization（彻底修复转义和格式）
    token_examples = """<table border="0" cellborder="1" cellspacing="6" cellpadding="6" style="rounded">
        <tr><td bgcolor="#FFFFFF" bordercolor="#495057">{ atom [C] &lt;m- 1&gt; = atom [O] }</td></tr>
        <tr><td bgcolor="#FFFFFF" bordercolor="#495057">{ atom [C] ( = atom [C] &lt;m- 1&gt; &lt;m- 2&gt; ) atom [O] }</td></tr>
        <tr><td bgcolor="#FFFFFF" bordercolor="#495057">{ atom [C] ( atom [N] ) # atom [C] &lt;m- 2&gt; }</td></tr>
    </table>"""

    # 注意：在 HTML 标签中，必须用 <br/> 换行，不能用 \n
    a.node('A4',
           label='<Connection Handling<br/><b>&amp; Tokenization</b><br/><br/>Form connection processing,<br/>atom recognition,<br/>ring system notifications<br/>Sentences e.g.<br/>' + token_examples + '>',
           fillcolor='#E8F5E9', color='#81C784', style='filled,rounded')

    # A5: Sentence Formation
    sentence_examples = "{ atom [C] &lt;m- 1&gt; = atom [O] }<br/>{ atom [C] ( = atom [C] &lt;m- 1&gt; ) atom [O] }<br/>{ atom [C] ( atom [N] ) # atom [C] &lt;m- 2&gt; }"
    a.node('A5',
           label=f'<<b>Sentence<br/>formation</b><br/><br/>Complete molecular<br/>sequences<br/><br/>{sentence_examples}>',
           fillcolor='#E8F8F5', color='#73C6B6', style='filled,rounded')

    # 模块 A 内部连线
    a.edge('A1', 'A2')
    a.edge('A2', 'A3')
    a.edge('A3', 'A4')
    a.edge('A4', 'A5')

# 中间桥梁：过渡到 B 模块的前置节点
dot.node('Bridge', 'Serialized\nMolecular\nTokens', shape='none', fontname='Arial-Bold', fontsize='12')

# ==========================================
# B. Training and generation of GPT generative model
# ==========================================
with dot.subgraph(name='cluster_B') as b:
    b.attr(label='B. Training and generation of GPT generative model',
           style='filled,rounded', color='#E8F8F5', fillcolor='#F4FBF7', fontname='Arial-Bold', fontsize='14')

    b.node('B1', 'Embedding &\nPositional\nEncoding', fillcolor='#A9DFBF', color='#27AE60', style='filled,rounded')

    # 内部 Transformer 块
    tf_block1 = """<table border="0" cellborder="1" cellspacing="2" cellpadding="4" bgcolor="#FFFFFF">
        <tr><td bgcolor="#FCF3CF">Add &amp; Norm</td></tr>
        <tr><td bgcolor="#D6EAF8">Feed Forward</td></tr>
        <tr><td bgcolor="#FCF3CF">Add &amp; Norm</td></tr>
        <tr><td bgcolor="#EDBB99">Multi-head Self-attention</td></tr>
    </table>"""
    b.node('B_TF1', label=tf_block1, shape='plaintext')
    b.node('B_Dots', '...', shape='none', fontname='Arial-Bold', fontsize='16')
    b.node('B_TF2', label=tf_block1, shape='plaintext')

    b.node('B_Out', 'Sentence', shape='none')

    # B 模块内部连线
    b.edge('B1', 'B_TF1')
    b.edge('B_TF1', 'B_Dots')
    b.edge('B_Dots', 'B_TF2')
    b.edge('B_TF2', 'B_Out')

# ==========================================
# C. Molecule generation
# ==========================================
with dot.subgraph(name='cluster_C') as c:
    c.attr(label='C. Molecule generation',
           style='filled,rounded', color='#FDEBD0', fillcolor='#FEF9E7', fontname='Arial-Bold', fontsize='14')

    c.node('C1', 'Molecule\nReconstruction\nAlgorithm\n\nDecode back into\nchemical structures',
           fillcolor='#F8C471', color='#E67E22', style='filled,rounded')
    c.node('C2', 'Generated Results\n\n[2D Molecular Structures]', shape='none')

    c.edge('C1', 'C2')

# ==========================================
# 跨模块全局主连线
# ==========================================
dot.edge('A5', 'Bridge', ltail='cluster_A', minlen='2')
dot.edge('Bridge', 'B1', lhead='cluster_B')
dot.edge('B_Out', 'C1', ltail='cluster_B', lhead='cluster_C', label=' Output ')

# 保存并渲染
dot.render('nature_molecular_workflow', view=True)