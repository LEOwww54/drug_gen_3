import torch
from rdkit.Chem import FragmentCatalog
import os
from rdkit import RDConfig, Chem

elements = ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"]
ele_num = {
"H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
"Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20,
"Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30,
"Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36, "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40,
"Nb": 41, "Mo": 42, "Tc": 43, "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50,
"Sb": 51, "Te": 52, "I": 53, "Xe": 54, "Cs": 55, "Ba": 56, "La": 57, "Ce": 58, "Pr": 59, "Nd": 60,
"Pm": 61, "Sm": 62, "Eu": 63, "Gd": 64, "Tb": 65, "Dy": 66, "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70,
"Lu": 71, "Hf": 72, "Ta": 73, "W": 74, "Re": 75, "Os": 76, "Ir": 77, "Pt": 78, "Au": 79, "Hg": 80,
"Tl": 81, "Pb": 82, "Bi": 83, "Po": 84, "At": 85, "Rn": 86, "Fr": 87, "Ra": 88, "Ac": 89, "Th": 90,
"Pa": 91, "U": 92, "Np": 93, "Pu": 94, "Am": 95, "Cm": 96, "Bk": 97, "Cf": 98, "Es": 99, "Fm": 100,
"Md": 101, "No": 102, "Lr": 103, "Rf": 104, "Db": 105, "Sg": 106, "Bh": 107, "Hs": 108, "Mt": 109,
"Ds": 110, "Rg": 111, "Cn": 112, "Nh": 113, "Fl": 114, "Mc": 115, "Lv": 116, "Ts": 117, "Og": 118
}

H_elements = ['He', 'Hf', 'Hg', 'Ho', 'Hs']

num_ele = {k : v for v, k in ele_num.items()}


atomic_radii = {
    "H": 0.31,   # 氢
    "He": 0.28,  # 氦
    "Li": 1.28,  # 锂
    "Be": 0.96,  # 铍
    "B": 0.84,   # 硼
    "C": 0.76,   # 碳
    "N": 0.71,   # 氮
    "O": 0.66,   # 氧
    "F": 0.57,   # 氟
    "Ne": 0.58,  # 氖
    "Na": 1.66,  # 钠
    "Mg": 1.41,  # 镁
    "Al": 1.21,  # 铝
    "Si": 1.11,  # 硅
    "P": 1.07,   # 磷
    "S": 1.05,   # 硫
    "Cl": 1.02,  # 氯
    "Ar": 1.06,  # 氩
    "K": 2.03,   # 钾
    "Ca": 1.76,  # 钙
    "Fe": 1.24,  # 铁
    "Cu": 1.32,  # 铜
    "Zn": 1.22,  # 锌
    "Br": 1.20,  # 溴
    "I": 1.39,   # 碘
}

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

vocab_size = 100
# dict_datas = json.load(open('dict_datas.json', 'r'))
# word2id, id2word = dict_datas['word2id'], dict_datas['id2word']
emb_size = 768
max_pos = 1800
max_pos_1 = 1800

d_model = emb_size  # Embedding Size
d_ff = 2048  # FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
n_layers = 12  # number of Encoder of Decoder Layer
n_heads = 12  # number of heads in Multi-Head Attention
CLIP = 1
dropout_rate = 0.1

rank = 0

SUFFIX, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN, CLS_TOKEN, START_TOKEN, SEP_TOKEN, SEP1_TOKEN = "", "<pad>", "<s>", "</s>", "<unk>", "<cls>", "<start>", "<sep>", "<sep1>"
SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN, CLS_TOKEN, START_TOKEN, SEP_TOKEN, SEP1_TOKEN]
PAD_TOKEN_ID, BOS_TOKEN_ID, EOS_TOKEN_ID, UNK_TOKEN_ID, CLS_TOKEN_ID, START_TOKEN_ID, SEP_TOKEN_ID, SEP1_TOKEN = range(8)

protein_emb_size = 44
prop_len = 3

maccs_sub_patt = ['[#104]',
    '[#32,#33,#34,#50,#51,#52,#82,#83,#84]',
    '[Ac,Th,Pa,U,Np,Pu,Am,Cm,Bk,Cf,Es,Fm,Md,No,Lr]',
    '[Sc,Ti,Y,Zr,Hf]',
    '[La,Ce,Pr,Nd,Pm,Sm,Eu,Gd,Tb,Dy,Ho,Er,Tm,Yb,Lu]',
    '[V,Cr,Mn,Nb,Mo,Tc,Ta,W,Re]',
    '[Fe,Co,Ni,Ru,Rh,Pd,Os,Ir,Pt]',
    '[Be,Mg,Ca,Sr,Ba,Ra]',
    '[Cu,Zn,Ag,Cd,Au,Hg]',
    '[#8]~[#7](~[#6])~[#6]',
    '[#16]-[#16]',
    '[#8]~[#6](~[#8])~[#8]',
    '[#6]#[#6]',
    '[#5,#13,#31,#49,#81]',
    '[#14]',
    '[#6]=[#6](~[!#6;!#1])~[!#6;!#1]',
    '[#7]~[#6](~[#8])~[#8]',
    '[#7]-[#8]',
    '[#7]~[#6](~[#7])~[#7]',
    '[#6]=;@[#6](@*)@*',
    '[I]',
    '[!#6;!#1]~[CH2]~[!#6;!#1]',
    '[#15]',
    '[#6]~[!#6;!#1](~[#6])(~[#6])~*',
    '[!#6;!#1]~[F,Cl,Br,I]',
    '[#6]~[#16]~[#7]',
    '[#7]~[#16]',
    '[CH2]=*',
    '[Li,Na,K,Rb,Cs,Fr]',
    '[#7]~[#6](~[#8])~[#7]',
    '[#7]~[#6](~[#6])~[#7]',
    '[#8]~[#16](~[#8])~[#8]',
    '[#16]-[#8]',
    '[#6]#[#7]',
    'F',
    '[!#6;!#1;!H0]~*~[!#6;!#1;!H0]',
    '[#6]=[#6]~[#7]',
    'Br',
    '[#16]~*~[#7]',
    '[#8]~[!#6;!#1](~[#8])(~[#8])',
    '[#6]=[#6](~[#6])~[#6]',
    '[#6]~[#16]~[#8]',
    '[#7]~[#7]',
    '[!#6;!#1;!H0]~*~*~*~[!#6;!#1;!H0]',
    '[!#6;!#1;!H0]~*~*~[!#6;!#1;!H0]',
    '[#8]~[#16]~[#8]',
    '[#8]~[#7](~[#8])~[#6]',
    '[!#6;!#1]~[#16]~[!#6;!#1]',
    '[#16]=[#8]',
    '*~[#16](~*)~*',
    '[#7]=[#8]',
    '[#6]~[#6](~[#6])(~[#6])~*',
    '[!#6;!#1]~[#16]',
    '[!#6;!#1;!H0]~[!#6;!#1;!H0]',
    '[!#6;!#1]~[!#6;!#1;!H0]',
    '[!#6;!#1]~[#7]~[!#6;!#1]',
    '[#7]~[#8]',
    '[#8]~*~*~[#8]',
    '[#16]=*',
    '[CH3]~*~[CH3]',
    '[#6]=[#6](~*)~*',
    '[#7]~*~[#7]',
    '[#6]=[#7]',
    '[#7]~*~*~[#7]',
    '[#7]~*~*~*~[#7]',
    '[#16]~*(~*)~*',
    '*~[CH2]~[!#6;!#1;!H0]',
    '[NH2]',
    '[#6]~[#7](~[#6])~[#6]',
    '[C;H2,H3][!#6;!#1][C;H2,H3]',
    '[#16]',
    '[#8]~*~*~*~[#8]',
    '[#8]~[#6](~[#7])~[#6]',
    '[!#6;!#1]~[CH3]',
    '[!#6;!#1]~[#7]',
    '[#7]~*~*~[#8]',
    '[#7]~*~*~*~[#8]',
    '[#6]=[#6]',
    '*~[CH2]~[#7]',
    '[!#6;!#1]~[#8]',
    'Cl',
    '[!#6;!#1;!H0]~*~[CH2]~*',
    '[!#6;!#1]~*(~[!#6;!#1])~[!#6;!#1]',
    '[F,Cl,Br,I]~*(~*)~*',
    '[CH3]~*~*~*~[CH2]~*',
    '*~[CH2]~[#8]',
    '[#7]~[#6]~[#8]',
    '[#7]~*~[CH2]~*',
    '*~*(~*)(~*)~*',
    '[#8]!:*:*',
    '[CH3]~[CH2]~*',
    '[CH3]~*~[CH2]~*',
    '[#7]~*~[#8]',
    '[#7]=*',
    '*~[#7](~*)~*',
    '[#8]~[#6]~[#8]',
    '[!#6;!#1]~[!#6;!#1]',
    '[!#6;!#1]~[!#6;!#1]',
    '[!#6;!#1;!H0]',
    '[#8]~*~[CH2]~*',
    '*@*!@[#7]',
    '[F,Cl,Br,I]',
    '[#8]=*',
    '[!#6;!#1]~[CH2]~*',
    '[O;!H0]',
    '[#8]',
    '[CH3]',
    '[#7]',
    '*~[!#6;!#1](~*)~*',
    '[#7;!H0]',
    '[#8]~[#6](~[#6])~[#6]',
    '[!#6;!#1]~[CH2]~*',
    '[#6]=[#8]',
    '*!@[CH2]!@*',
    '[#7]~*(~*)~*',
    '[#6]-[#8]',
    '[#6]-[#7]',

    '[#6]=;@[#6](@*)@*',
    '*@*!@*@*',
    '*@*!@[#16]',
    '*!@[#7]@*',
    '[F,Cl,Br,I]!@*@*',
    '[!#6;!#1;!H0]~*~*~[CH2]~*',
    '[!#6;!#1;!H0]~*~*~*~[CH2]~*',
    '*@*!@[#7]',
    '*@*!@[#8]',
    '*!@*@*!@*',
    '*!@[CH2]!@*',
]

def fg_list():  # 47 FGs list
    fName = os.path.join(RDConfig.RDDataDir, 'FunctionalGroups.txt')
    # print(fName)
    fparams = FragmentCatalog.FragCatParams(1, 6, fName)

    xx = ['OP(O)(O)=O', 'NS(C)(=O)=O', 'S(=O)(=O)OC', 'NC(C)=O', 'S(=O)(=O)O', 'S(N)(=O)=O', 'C(C)(C)C', 'C(F)(F)F',
          'C(=O)OC', 'S(C)(=O)=O', 'S(=O)(=O)Cl', 'C(N)=O', 'N=C=S', 'N(O)=O', 'C(O)=O', 'OCC', 'N=C=O', 'S(C)=O', 'N=NC',
          'COC', 'C#C', '[O;D2]', 'NC', 'C=C', 'C=O', 'N#N', 'SC', 'OC', 'C#N', 'N=C', 'N=O', 'N=N', 'NO', '[P;D2]=O',
          'B', 'P', 'O', 'Cl', 'F', 'N', 'Br', '[NH+]', 'S', 'I', '[Si]', '[Se]', '[Na+]', '[C;!$(C-[#6])]']


    xx.extend(maccs_sub_patt)

    x1 = []
    for i in xx:
        x1.append((i, Chem.MolFromSmarts(i).GetNumAtoms()))
        pass

    x1.sort(key=lambda x: x[1], reverse=False)
    f_g_list = []
    for i in x1:
        s = i[0]
        f_g_list.append(s)

    # print(f_g_list)
    return ['C(=O)O']

func_group_list = fg_list()