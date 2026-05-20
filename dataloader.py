import xml.sax
import os
import numpy as np
from rdkit import Chem
import pandas as pd

def data_from_PDBbind(root_dir='G:\\database\\PDBbind_v2020_refined'):
    paths = os.walk(root_dir)
    PL_data = {}

    for path, dir_lst, file_lst in paths:
        for dir_name in dir_lst:
            if not (dir_name == 'index' or dir_name == 'readme'):
                full = str(os.path.join(path, dir_name))
                PL_data[dir_name] = {}
                ligand_sdf = full + '\\' +  dir_name + '_ligand.sdf'
                PL_data[dir_name]['ligand'] = Chem.SDMolSupplier(ligand_sdf)
                pocket_pdb = full + '\\' + dir_name + '_pocket.pdb'
                PL_data[dir_name]['pocket'] = Chem.MolFromPDBFile(pocket_pdb)
                protein_pdb = full + '\\' + dir_name + '_protein.pdb'
                PL_data[dir_name]['protein'] = Chem.MolFromPDBFile(protein_pdb)

    # np.save("traindata/PL_data.npy", PL_data)
    return PL_data

class ConfigHandler(xml.sax.ContentHandler):
    def __init__(self):
        self.attr = {}
        self.name = ""
        self.content = ""

        self.data = {}
        self.count = 0
        self.index = -1

    def startDocument(self):
        print("begin")

    def startElement(self, name, attributes):
        if not name==self.name:
            print('name:', name)

        if name == 'drug' or name == 'drugbank':
            self.data[self.count] = {}
            self.count = self.count + 1
            self.index = self.index + 1

        tmp = {}
        for key, value in attributes.items():
            tmp[key] = value

        self.data[self.index][name] = tmp

        self.name = name
        self.attr = attributes
        # for key, value in attributes.items():
        #     print(key, '----', value)

    def characters(self, content):
        self.content = content
        self.data[self.index][self.name].update({'content': content})

    def endElement(self, name):
        x = input()
        pass

    def endDocument(self):
        np.save('database', self.data, allow_pickle=True)
        print('end')

def data_from_drugbank_full(file=r'G:\database\drugbank\full database.xml'):
    parser = xml.sax.make_parser()
    parser.setFeature(xml.sax.handler.feature_namespaces, 0)
    Handler = ConfigHandler()
    parser.setContentHandler(Handler)
    parser.parse(file)

def data_from_drugbank_sdf(file=r'G:\database\drugbank\3D structures.sdf', isSave = False):
    suppl_h = Chem.SDMolSupplier(file)
    mols = [mol for mol in suppl_h if mol]

    smis = [mol.GetProp('SMILES') for mol in mols]
    print(len(smis))

    if isSave:
        with open(r'drugbank_smiles', 'w') as f:
            for smi in smis:
                f.write(smi + '\n')

    return smis

def data_from_target_protein_seq(path=r'G:\database\drugbank\protein.fasta'):
    items = []  # 存储所有项的列表
    current_item = None  # 当前正在构建的项

    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()  # 去除首尾空白字符

            if line.startswith('>'):
                # 遇到新项的开始
                if current_item is not None:
                    # 如果已经有正在构建的项，先保存它
                    items.append(current_item)

                # 开始新项
                current_item = ''
            else:
                # 将行内容添加到当前项
                if current_item is not None and line:  # 忽略空行
                    current_item += line
    # 添加最后一个项（如果存在）
    if current_item is not None and current_item:
        items.append(current_item)

    try:
        with open(r"database\protein_seq_pure", 'w', encoding='utf-8') as file:
            for item in items:
                # 写入每一项并在末尾添加.
                file.write(f"{item}.\n")
    except IOError as e:
        print(f"写入文件时出错: {e}")

    return items

def data_from_csv(path=r'G:\database\FG_BERT\Mole-BERT-data\toxcast_data.csv'):
    df = pd.read_csv(path)
    data_dict = df.to_dict('index')

    return data_dict


def data_from_ACE_smiles_only(target):
    data = data_from_csv(path='data/molACE/benchmark_data/' + target + '.csv')
    smiles_train = []
    smiles_test = []
    smiles_valid = []
    smiles = []
    for key, item in data.items():
        split = item['split']
        smiles.append(item['smiles'])
        if split == 'train':
            smiles_train.append(item['smiles'])
        elif split == 'test':
            smiles_test.append(item['smiles'])
        elif split == 'valid':
            smiles_valid.append(item['smiles'])

    return smiles_train, smiles_valid, smiles_test, smiles



def data_from_raw_file(path, n=0, skip_head = False):
    smiles = []
    with open(path, 'r') as file_:
        if not skip_head:
            line = file_.readline()
            count = 1
        else:
            line = file_.readline()
            line = file_.readline()
            count = 1
        while line:
            smiles.append(line)
            line = file_.readline()
            if count >= n > 0:
                break
            count = count + 1

    smiles_list = [smile.strip() for smile in smiles]

    return smiles_list

def data_from_geom(n):
    smiles_train = data_from_raw_file(n=n,path='data/geom_train.txt')
    smiles_test = data_from_raw_file(n=n,path='data/geom_test.txt')
    # smiles_val = set(data_from_raw_file(n=n,path='data/geom_val.txt'))

    return smiles_train, smiles_test, # list(smiles_val)

def data_from_pubchem(n, path=r'G:\database\CID-SMILES\CID-SMILES', is_Save=False):
    smiles = data_from_raw_file(n=n, path=path)

    smiles_list = [smile.strip() for smile in smiles]
    smile_list = [line.split('\t')[1] for line in smiles_list]


    if is_Save:
        with open(r"..\database\smiles_seq_pure_large", 'w', encoding='utf-8') as file:
            for item in smile_list:
                # 写入每一项并在末尾添加.
                file.write(f"{item}.\n")
            file.close()

    return smile_list

def data_from_ZINC_refined(n, path=r'data\ZINC_refined.txt'):
    smiles = {}
    smiles_all = []
    with open(path, 'r') as file_:
        lines = file_.readlines()
        count= 0
        for line in lines[1:]:
            if n > 0 and count >= n:
                break
            count = count + 1
            s = line.split(',')
            smile = s[0]
            type = s[1]
            type = type.replace('\n', '')
            if type not in smiles:
                smiles[type] = [smile]
            else:
                smiles[type].append(smile)
            smiles_all.append(smile)

    return smiles, smiles_all

def data_from_chembl_all_smiles(n=0):

    return data_from_raw_file(n=n, path=r'data/chembl_all_smiles.txt')

def data_from_chembl_smiles_refined(n=0):

    return data_from_raw_file(n=n, path=r'data/chembl_refined_dataset.txt')

def data_from_chembl_smiles_qed(n=0):

    return data_from_raw_file(n=n, path=r'data/chembl_qed_smiles_only.txt')

def data_from_BindingDB_smiles(n=0):

    return data_from_raw_file(n=n, path=r'data/BindingDB_smiles.txt')

def read_cube_file(filename):
    """读取 Cube 文件并返回电子密度数据"""

    with open(filename, 'r') as f:
        # 跳过前两行注释
        f.readline()
        f.readline()

        # 读取原子数量和原点坐标
        line = f.readline().split()
        natoms = int(line[0])
        origin = np.array([float(x) for x in line[1:4]])

        # 读取网格信息
        line = f.readline().split()
        nx = int(line[0])
        dx = np.array([float(x) for x in line[1:4]])

        line = f.readline().split()
        ny = int(line[0])
        dy = np.array([float(x) for x in line[1:4]])

        line = f.readline().split()
        nz = int(line[0])
        dz = np.array([float(x) for x in line[1:4]])

        # 读取原子信息
        atoms = []
        for i in range(natoms):
            line = f.readline().split()
            atomic_num = int(line[0])
            coord = np.array([float(x) for x in line[2:5]])
            atoms.append((atomic_num, coord))

        # 读取电子密度数据
        data = []
        for line in f:
            data.extend([float(x) for x in line.split()])

        # 将数据转换为三维数组
        density = np.array(data).reshape((nx, ny, nz))

        return {
            'origin': origin,
            'shape': (nx, ny, nz),
            'spacing': (dx, dy, dz),
            'atoms': atoms,
            'density': density
        }

import os

def read_file_cross_platform(filename):
    try:
        print(f"开始读取文件: {filename}")
        print("按回车键显示下一行，按 'q' 键退出...\n")

        with open(filename, 'r', encoding='utf-8') as file:
            line_count = 0

            while True:
                line = file.readline()

                if not line:  # 文件结束
                    print("\n已到达文件末尾！")
                    break

                line_count += 1
                line = line.rstrip('\n')  # 移除行尾换行符

                print(f"第 {line_count} 行: {line}")

                # 等待用户输入
                user_input = input("")  # 回车继续

                if user_input.lower() == 'q':
                    print(f"已读取 {line_count} 行，程序退出")
                    break

        print(f"\n文件读取完成，共读取 {line_count} 行")

    except FileNotFoundError:
        print(f"错误: 文件 '{filename}' 未找到")
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        print(f"发生错误: {e}")

def _parquet2smiles(path):
    import pyarrow.parquet as pq
    table = pq.read_table(path)
    df = table.to_pandas()
    return df

def data_from_ZINC_250K():
    train_df = _parquet2smiles('data/ZINC_250K_train.parquet')
    test_df = _parquet2smiles('data/ZINC_250K_test.parquet')

    train = train_df['smiles'].tolist()
    test = test_df['smiles'].tolist()
    all = []
    for target in train:
        all.append(target)
    for target in test:
        all.append(target)

    return train, test, all

def j():
    targets = [
        'CHEMBL204_Ki',
        'CHEMBL214_Ki',
        'CHEMBL233_Ki',
        'CHEMBL234_Ki',
    ]

    for target in targets:
        train, _, _, _ = data_from_ACE_smiles_only(target)
        with open(target + '_train', 'w') as f:
            f.write('\n'.join(train))

if __name__ == "__main__":
    x = data_from_ZINC_250K()
    pass
