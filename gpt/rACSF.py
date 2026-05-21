"""
增强版残基描述符：纳入分子间相互作用潜能
包含基础几何特征和分子间相互作用潜能
"""

import numpy as np
from Bio.PDB import PDBParser, is_aa, NeighborSearch
from scipy.spatial.distance import cdist
from scipy.special import logsumexp
import warnings


class EnhancedResidueDescriptor:
    """
    增强版残基描述符

    特征组成：
    - 基础特征 (36维)：径向分布(8) + 角度分布(8) + 氨基酸组成(20)
    - 相互作用潜能 (8维)：氢键给体/受体/方向性 + π-堆积(2) + π-阳离子 + 疏水 + 综合指数
    - 总计：44维
    """

    def __init__(self,
                 cutoff=12.0,  # 截断半径 (Å)
                 eta=0.5,  # RBF宽度参数
                 n_rbf=8,  # 径向基函数数量
                 n_angular_bins=8,  # 角度直方图bins数量
                 use_sequence_weight=True,
                 sequence_lambda=5.0,
                 include_interaction_potentials=True,
                 ph=7.0):  # pH值，影响质子化状态
        """
        参数:
            cutoff: 局部环境的截断半径
            eta: RBF的宽度参数
            n_rbf: 径向分布使用的RBF数量
            n_angular_bins: 角度直方图的bins数量
            use_sequence_weight: 是否使用序列距离权重
            sequence_lambda: 序列权重衰减长度
            include_interaction_potentials: 是否包含相互作用潜能
            ph: pH值，影响氢键给体/受体质子化状态
        """
        self.cutoff = cutoff
        self.eta = eta
        self.n_rbf = n_rbf
        self.n_angular_bins = n_angular_bins
        self.use_sequence_weight = use_sequence_weight
        self.sequence_lambda = sequence_lambda
        self.include_interaction_potentials = include_interaction_potentials
        self.ph = ph

        # 20种标准氨基酸
        self.aa_types = [
            'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
            'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'
        ]
        self.aa_to_idx = {aa: i for i, aa in enumerate(self.aa_types)}
        self.n_aa = len(self.aa_types)

        # 预计算RBF中心（均匀分布在[0, cutoff]）
        self.rbf_centers = np.linspace(0, cutoff, n_rbf)

        # 初始化相互作用相关的属性
        self._init_interaction_properties()

        # 存储结构相关的计算结果（在compute时填充）
        self._structure = None
        self._residue_asa = {}
        self._neighbor_search = None

    def _init_interaction_properties(self):
        """初始化相互作用相关的属性"""

        # 氢键给体原子（带极性氢的原子）
        self.donor_atoms = {
            # 主链NH
            'N': {'residues': 'all', 'pka': 15.0, 'weight': 1.0},
            # 侧链OH
            'OG': {'residues': ['SER'], 'pka': 13.0, 'weight': 1.0},
            'OG1': {'residues': ['THR'], 'pka': 13.0, 'weight': 1.0},
            'OH': {'residues': ['TYR'], 'pka': 10.0, 'weight': 1.0},
            # 侧链NH/NH2
            'NE': {'residues': ['ARG'], 'pka': 12.0, 'weight': 1.0},
            'NH1': {'residues': ['ARG'], 'pka': 12.0, 'weight': 0.8},
            'NH2': {'residues': ['ARG'], 'pka': 12.0, 'weight': 0.8},
            'NZ': {'residues': ['LYS'], 'pka': 10.5, 'weight': 1.0},
            'NE2': {'residues': ['HIS', 'GLN'], 'pka': 6.0, 'weight': 0.7},
            'ND1': {'residues': ['HIS'], 'pka': 6.0, 'weight': 0.7},
            # 巯基
            'SG': {'residues': ['CYS'], 'pka': 8.5, 'weight': 0.6},
        }

        # 氢键受体原子（孤对电子）
        self.acceptor_atoms = {
            # 主链羰基
            'O': {'residues': 'all', 'electronegativity': 3.5, 'weight': 0.8},
            # 侧链羰基/羧基
            'OD1': {'residues': ['ASP', 'ASN'], 'electronegativity': 3.5, 'weight': 1.0},
            'OD2': {'residues': ['ASP', 'ASN'], 'electronegativity': 3.5, 'weight': 1.0},
            'OE1': {'residues': ['GLU', 'GLN'], 'electronegativity': 3.5, 'weight': 1.0},
            'OE2': {'residues': ['GLU', 'GLN'], 'electronegativity': 3.5, 'weight': 1.0},
            # 羟基氧
            'OG': {'residues': ['SER'], 'electronegativity': 3.5, 'weight': 0.8},
            'OG1': {'residues': ['THR'], 'electronegativity': 3.5, 'weight': 0.8},
            'OH': {'residues': ['TYR'], 'electronegativity': 3.5, 'weight': 0.8},
            # 芳香氮
            'NE2': {'residues': ['HIS'], 'electronegativity': 3.0, 'weight': 0.6},
            'ND1': {'residues': ['HIS'], 'electronegativity': 3.0, 'weight': 0.6},
        }

        # 芳香残基
        self.aromatic_residues = {'PHE', 'TYR', 'TRP', 'HIS'}

        # 阳离子残基
        self.cationic_residues = {'ARG', 'LYS', 'HIS'}

        # 疏水性标度（Kyte-Doolittle）
        self.hydrophobicity = {
            'ILE': 4.5, 'VAL': 4.2, 'LEU': 3.8, 'PHE': 2.8, 'CYS': 2.5,
            'MET': 1.9, 'ALA': 1.8, 'GLY': -0.4, 'THR': -0.7, 'SER': -0.8,
            'TRP': -0.9, 'TYR': -1.3, 'PRO': -1.6, 'HIS': -3.2, 'GLU': -3.5,
            'GLN': -3.5, 'ASP': -3.5, 'ASN': -3.5, 'LYS': -3.9, 'ARG': -4.5
        }

        # 芳香环电子密度（基于取代基效应）
        self.ring_electron_density = {
            'PHE': 0.5,  # 中性
            'TYR': 0.6,  # -OH供电子
            'TRP': 0.7,  # 富电子吲哚
            'HIS': 0.4,  # 缺电子咪唑（pH依赖）
        }

    def _compute_surface_accessibility(self, structure):
        """
        计算每个原子的溶剂可及表面积（ASA）
        使用Shrake-Rupley算法
        """
        from Bio.PDB import ShrakeRupley

        sr = ShrakeRupley()
        sr.compute(structure, level='A')

        self._residue_asa = {}
        for model in structure:
            for chain in model:
                for residue in chain:
                    if not is_aa(residue):
                        continue
                    res_key = (chain.get_id(), residue.id[1])
                    total_asa = 0.0
                    for atom in residue:
                        if hasattr(atom, 'sasa'):
                            total_asa += atom.sasa
                    self._residue_asa[res_key] = total_asa

    def _protonation_probability(self, pka):
        """
        计算原子在给定pH下的质子化概率
        使用Henderson-Hasselbalch方程
        对于给体：pKa越高，越容易保持质子化
        """
        if pka > 14:  # 不可去质子化
            return 1.0
        return 1.0 / (1.0 + 10 ** (self.ph - pka))

    def _compute_steric_accessibility(self, atom):
        """
        计算原子周围的空间可用性
        使用局部邻居密度作为代理
        """
        coord = atom.get_coord()

        # 搜索周围原子（5Å内）
        if self._neighbor_search is None:
            return 0.5

        nearby = self._neighbor_search.search(coord, 5.0)

        # 排除自身
        nearby = [a for a in nearby if a != atom]

        # 空间拥挤度：邻居数量
        crowding = len(nearby) / 30.0  # 30个原子为完全拥挤
        crowding = min(1.0, crowding)

        # 可用性 = 1 - 拥挤度
        return 1.0 - crowding

    def _get_residue_asa(self, chain_id, res_id):
        """获取残基的ASA"""
        return self._residue_asa.get((chain_id, res_id), 0.0)

    def compute_hbond_potential(self, residue, chain_id, res_id):
        """
        计算残基的氢键潜能（与任意分子的潜在氢键能力）

        返回:
            donor_potential: 给体潜能 (0-1)
            acceptor_potential: 受体潜能 (0-1)
            directional_potential: 方向性潜能 (0-1)
        """
        res_name = residue.get_resname()
        if res_name == 'MSE':
            res_name = 'MET'

        donor_score = 0.0
        acceptor_score = 0.0

        # 获取残基的ASA
        asa = self._get_residue_asa(chain_id, res_id)
        exposure = min(1.0, asa / 100.0)  # 100 Å²作为完全暴露阈值

        # 检查给体原子
        for atom_name, info in self.donor_atoms.items():
            if atom_name in residue:
                atom = residue[atom_name]

                # 基础给体强度（基于pKa和ASA）
                pka = info.get('pka', 10.0)
                protonated = self._protonation_probability(pka)
                donor_strength = protonated * info.get('weight', 1.0)

                # 几何可用性（是否有空间容纳氢键伙伴）
                geometry_factor = self._compute_steric_accessibility(atom)

                donor_score += donor_strength * exposure * geometry_factor

        # 检查受体原子
        for atom_name, info in self.acceptor_atoms.items():
            if atom_name in residue:
                atom = residue[atom_name]

                # 受体强度（基于电负性）
                electronegativity = info.get('electronegativity', 3.0)
                acceptor_strength = (electronegativity - 2.5) / 1.5
                acceptor_strength = max(0, min(1, acceptor_strength))
                acceptor_strength *= info.get('weight', 1.0)

                geometry_factor = self._compute_steric_accessibility(atom)

                acceptor_score += acceptor_strength * exposure * geometry_factor

        # 归一化（最多2个有效给体/受体位点）
        donor_potential = min(1.0, donor_score / 2.0)
        acceptor_potential = min(1.0, acceptor_score / 2.0)

        # 方向性潜能（基于二级结构类型简化估计）
        directional_potential = self._estimate_directional_accessibility(residue)

        return donor_potential, acceptor_potential, directional_potential

    def _estimate_directional_accessibility(self, residue):
        """
        估计氢键的方向性可用性
        基于局部环境简化估计
        """
        # 简化实现：基于周围原子数量估计
        if self._neighbor_search is None:
            return 0.5

        try:
            ca_atom = residue['CA']
            coord = ca_atom.get_coord()
            nearby = self._neighbor_search.search(coord, 6.0)

            # 邻居越多，方向越受限
            crowding = len(nearby) / 40.0
            crowding = min(1.0, crowding)

            return 1.0 - crowding * 0.5
        except:
            return 0.5

    def compute_pipi_potential(self, residue, chain_id, res_id):
        """
        计算π-π堆积潜能（与芳香分子、配体等的堆积能力）

        返回:
            stacking_potential: 堆积潜能 (0-1)
            edge_to_face_potential: T型堆积潜能 (0-1)
        """
        res_name = residue.get_resname()
        if res_name == 'MSE':
            res_name = 'MET'

        if res_name not in self.aromatic_residues:
            return 0.0, 0.0

        # 获取残基ASA
        asa = self._get_residue_asa(chain_id, res_id)

        # 芳香环暴露程度（完全暴露约80 Å²）
        ring_exposure = min(1.0, asa / 80.0)

        # 电子特性
        electron_factor = self.ring_electron_density.get(res_name, 0.5)

        # 环平面可及性（天然氨基酸中接近1）
        planarity_factor = 1.0

        # 平行堆积潜能（适合与平面分子堆积）
        stacking_potential = ring_exposure * electron_factor * planarity_factor

        # T型堆积潜能（适合与边缘分子作用）
        edge_to_face_potential = ring_exposure * (1 - electron_factor) * planarity_factor

        return stacking_potential, edge_to_face_potential

    def compute_cation_pi_potential(self, residue, chain_id, res_id):
        """
        计算π-阳离子相互作用潜能

        对于芳香残基：评估与阳离子结合的能力
        对于阳离子残基：评估与芳香环结合的能力
        """
        res_name = residue.get_resname()
        if res_name == 'MSE':
            res_name = 'MET'

        # 获取ASA
        asa = self._get_residue_asa(chain_id, res_id)

        if res_name in self.aromatic_residues:
            # 芳香残基作为π体系
            ring_exposure = min(1.0, asa / 80.0)
            electron_factor = self.ring_electron_density.get(res_name, 0.5)
            return ring_exposure * electron_factor

        elif res_name in self.cationic_residues:
            # 阳离子残基
            charge_exposure = min(1.0, asa / 100.0)

            # 电荷可用性
            if res_name == 'ARG':
                charge_availability = 1.0
            elif res_name == 'LYS':
                charge_availability = 0.7
            else:  # HIS
                charge_availability = self._protonation_probability(6.0)

            return charge_exposure * charge_availability

        return 0.0

    def compute_hydrophobic_potential(self, residue, chain_id, res_id):
        """
        计算疏水相互作用潜能（与疏水配体/膜的作用能力）
        """
        res_name = residue.get_resname()
        if res_name == 'MSE':
            res_name = 'MET'

        # 获取疏水性
        h = self.hydrophobicity.get(res_name, 0.0)

        # 标准化到[0,1]
        h_norm = (h + 4.5) / 9.0
        h_norm = max(0, min(1, h_norm))

        # 获取ASA
        asa = self._get_residue_asa(chain_id, res_id)

        # 疏水残基需要适当暴露才能与其他疏水分子作用
        # 最优暴露程度约50%
        exposure_factor = np.exp(-((asa / 100.0 - 0.5) ** 2) / (2 * 0.3 ** 2))

        return h_norm * exposure_factor

    def _extract_residues(self, structure):
        """从结构中提取所有残基的Cα坐标和类型信息"""
        residues = []

        for model in structure:
            for chain in model:
                residue_list = list(chain.get_residues())
                for seq_idx, residue in enumerate(residue_list):
                    if not is_aa(residue) or 'CA' not in residue:
                        continue

                    res_name = residue.get_resname()
                    if res_name == 'MSE':
                        res_name = 'MET'

                    if res_name not in self.aa_to_idx:
                        continue

                    residues.append({
                        'coord': residue['CA'].get_coord().copy(),
                        'type_idx': self.aa_to_idx[res_name],
                        'type_name': res_name,
                        'residue': residue,
                        'chain_id': chain.get_id(),
                        'res_id': residue.id[1],
                        'seq_idx': seq_idx
                    })

        if len(residues) == 0:
            warnings.warn("未找到有效的氨基酸残基")

        return residues

    def _compute_sequence_weights(self, N, seq_indices):
        """计算序列距离权重矩阵"""
        if not self.use_sequence_weight:
            return np.ones((N, N))

        idx_i = seq_indices.reshape(-1, 1)
        idx_j = seq_indices.reshape(1, -1)
        seq_dist = np.abs(idx_i - idx_j)

        weights = np.exp(-seq_dist / self.sequence_lambda)
        return weights

    def _cutoff_function(self, distances):
        """平滑截断函数"""
        fc = np.zeros_like(distances)
        mask = distances < self.cutoff
        t = distances[mask] / self.cutoff
        fc[mask] = 0.5 * (np.cos(np.pi * t) + 1)
        return fc

    def _compute_radial_features(self, distances, neighbor_mask, seq_weights_row):
        """
        计算径向分布特征（使用RBF网格）
        """
        if neighbor_mask.sum() == 0:
            return np.zeros(self.n_rbf)

        neighbor_distances = distances[neighbor_mask]
        neighbor_weights = seq_weights_row[neighbor_mask]

        radial_feat = np.zeros(self.n_rbf)
        for m, center in enumerate(self.rbf_centers):
            # 高斯RBF
            rbf_values = np.exp(-self.eta * (neighbor_distances - center) ** 2)
            # 加权求和并归一化
            radial_feat[m] = np.sum(neighbor_weights * rbf_values)

        # 归一化
        radial_feat = radial_feat / (neighbor_distances.shape[0] + 1e-8)

        return radial_feat

    def _compute_angular_features(self, coords, center_idx, neighbor_indices,
                                  neighbor_mask, seq_weights_row):
        """
        计算角度分布特征（使用直方图）
        """
        neighbor_count = neighbor_mask.sum()
        if neighbor_count < 2:
            return np.zeros(self.n_angular_bins)

        # 获取中心坐标和邻居坐标
        center = coords[center_idx]
        neighbor_coords = coords[neighbor_indices]
        neighbor_weights = seq_weights_row[neighbor_indices]

        # 计算从中心指向每个邻居的向量
        vectors = neighbor_coords - center
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors_normed = vectors / (norms + 1e-8)

        # 计算所有邻居对之间的夹角
        n_neighbors = len(neighbor_coords)
        cos_angles = np.clip(np.dot(vectors_normed, vectors_normed.T), -1.0, 1.0)
        angles = np.arccos(cos_angles)

        # 获取上三角部分（排除自身配对）
        triu_indices = np.triu_indices(n_neighbors, k=1)
        angles_triu = angles[triu_indices]

        if len(angles_triu) == 0:
            return np.zeros(self.n_angular_bins)

        # 计算加权直方图
        hist, _ = np.histogram(angles_triu,
                               bins=self.n_angular_bins,
                               range=(0, np.pi),
                               weights=None)

        # 归一化为概率分布
        hist = hist / (hist.sum() + 1e-8)

        return hist

    def _compute_composition_features(self, neighbor_indices, neighbor_mask,
                                      types, seq_weights_row):
        """
        计算局部环境的氨基酸组成特征
        """
        if neighbor_mask.sum() == 0:
            return np.zeros(self.n_aa)

        neighbor_types = types[neighbor_indices]
        neighbor_weights = seq_weights_row[neighbor_indices]

        # 加权统计
        weighted_hist = np.zeros(self.n_aa)
        for t, w in zip(neighbor_types, neighbor_weights):
            weighted_hist[t] += w

        # 归一化
        weighted_hist = weighted_hist / (weighted_hist.sum() + 1e-8)

        return weighted_hist

    def _compute_base_features_for_residue(self, i, coords, types, seq_indices,
                                           distances, seq_weights):
        """
        计算单个残基的基础特征
        """
        # 邻居掩码（排除自身）
        neighbor_mask = (distances[i] < self.cutoff) & (distances[i] > 0)
        neighbor_indices = np.where(neighbor_mask)[0]

        seq_weights_row = seq_weights[i]

        # 径向特征
        radial_feat = self._compute_radial_features(
            distances[i], neighbor_mask, seq_weights_row
        )

        # 角度特征
        angular_feat = self._compute_angular_features(
            coords, i, neighbor_indices, neighbor_mask, seq_weights_row
        )

        # 组成特征
        comp_feat = self._compute_composition_features(
            neighbor_indices, neighbor_mask, types, seq_weights_row
        )

        return np.concatenate([radial_feat, angular_feat, comp_feat])

    def _compute_interaction_potentials_for_residue(self, residue_info):
        """
        计算单个残基的相互作用潜能
        """
        residue = residue_info['residue']
        chain_id = residue_info['chain_id']
        res_id = residue_info['res_id']
        res_name = residue_info['type_name']

        # 氢键潜能
        donor, acceptor, directional = self.compute_hbond_potential(
            residue, chain_id, res_id
        )

        # π-堆积潜能
        stacking, edge_to_face = self.compute_pipi_potential(
            residue, chain_id, res_id
        )

        # π-阳离子潜能
        cation_pi = self.compute_cation_pi_potential(
            residue, chain_id, res_id
        )

        # 疏水潜能
        hydrophobic = self.compute_hydrophobic_potential(
            residue, chain_id, res_id
        )

        # 综合相互作用指数
        interaction_index = (donor + acceptor + stacking + edge_to_face +
                             cation_pi + hydrophobic) / 6.0

        return np.array([
            donor, acceptor, directional,
            stacking, edge_to_face,
            cation_pi,
            hydrophobic,
            interaction_index
        ])

    def compute_descriptors(self, structure, normalize=True):
        """
        计算蛋白质中每个残基的增强描述符

        参数:
            structure: Bio.PDB.Structure对象
            normalize: 是否进行标准化归一化

        返回:
            descriptors: N x D 的描述符矩阵
            residues_info: 残基信息列表
            feature_names: 特征名称列表
        """
        # 存储结构引用
        self._structure = structure

        # 计算表面可及性
        self._compute_surface_accessibility(structure)

        # 建立邻居搜索树
        atoms = list(structure.get_atoms())
        self._neighbor_search = NeighborSearch(atoms)

        # 提取残基信息
        residues = self._extract_residues(structure)
        if len(residues) == 0:
            return np.array([]), [], []

        N = len(residues)

        # 提取坐标和类型
        coords = np.array([r['coord'] for r in residues])
        types = np.array([r['type_idx'] for r in residues])
        seq_indices = np.array([r['seq_idx'] for r in residues])

        # 计算距离矩阵
        distances = cdist(coords, coords)

        # 计算序列权重矩阵
        seq_weights = self._compute_sequence_weights(N, seq_indices)

        # 为每个残基计算描述符
        descriptors_list = []

        for i in range(N):
            # 基础特征
            base_feat = self._compute_base_features_for_residue(
                i, coords, types, seq_indices, distances, seq_weights
            )

            if self.include_interaction_potentials:
                # 相互作用潜能
                inter_feat = self._compute_interaction_potentials_for_residue(residues[i])
                desc = np.concatenate([base_feat, inter_feat])
            else:
                desc = base_feat

            descriptors_list.append(desc)

        descriptors = np.array(descriptors_list)

        # 归一化
        if normalize:
            descriptors = self._normalize_descriptors(descriptors)

        # 生成特征名称
        feature_names = self._get_feature_names()

        # 准备返回的残基信息
        residues_info = [{k: v for k, v in r.items()
                          if k not in ['coord', 'residue']} for r in residues]

        # 清理临时数据
        self._structure = None
        self._neighbor_search = None

        return descriptors, residues_info, feature_names

    def _normalize_descriptors(self, descriptors, method='standard'):
        """
        归一化描述符

        参数:
            method: 'standard' (标准化) 或 'minmax' (Min-Max)
        """
        if method == 'standard':
            # 对数变换处理偏态分布
            descriptors = np.log1p(np.maximum(descriptors, 0))

            # Z-score标准化
            mean = np.mean(descriptors, axis=0)
            std = np.std(descriptors, axis=0)
            std[std < 1e-6] = 1.0
            descriptors = (descriptors - mean) / std

        elif method == 'minmax':
            # Min-Max归一化到[0,1]
            min_val = np.min(descriptors, axis=0)
            max_val = np.max(descriptors, axis=0)
            range_val = max_val - min_val
            range_val[range_val < 1e-6] = 1.0
            descriptors = (descriptors - min_val) / range_val

        return descriptors

    def _get_feature_names(self):
        """获取特征名称列表"""
        names = []

        # 径向特征
        for i, center in enumerate(self.rbf_centers):
            names.append(f'radial_rbf_{center:.2f}')

        # 角度特征
        for i in range(self.n_angular_bins):
            names.append(f'angular_bin_{i}')

        # 组成特征
        for aa in self.aa_types:
            names.append(f'comp_{aa}')

        if self.include_interaction_potentials:
            names.extend([
                'hbond_donor_potential',
                'hbond_acceptor_potential',
                'hbond_directional_potential',
                'pipi_stacking_potential',
                'pipi_edge_to_face_potential',
                'cation_pi_potential',
                'hydrophobic_potential',
                'interaction_index'
            ])

        return names

    def get_feature_info(self):
        """获取特征信息（维度、名称等）"""
        base_dim = self.n_rbf + self.n_angular_bins + self.n_aa
        inter_dim = 8 if self.include_interaction_potentials else 0

        return {
            'total_dimension': base_dim + inter_dim,
            'base_dimension': base_dim,
            'interaction_dimension': inter_dim,
            'n_rbf': self.n_rbf,
            'n_angular_bins': self.n_angular_bins,
            'n_aa': self.n_aa,
            'include_interaction_potentials': self.include_interaction_potentials
        }


# ============= 便捷函数 =============

def cal_rACSF(pdb_file,
                                 cutoff=12.0,
                                 include_interaction_potentials=True,
                                 normalize=True,
                                 verbose=True):
    """
    便捷函数：从PDB文件计算增强版残基描述符

    参数:
        pdb_file: PDB文件路径
        cutoff: 截断半径
        include_interaction_potentials: 是否包含相互作用潜能
        normalize: 是否归一化
        verbose: 是否打印信息

    返回:
        descriptors: 描述符矩阵
        residues_info: 残基信息
        feature_names: 特征名称
    """
    # 加载结构
    parser = PDBParser(PERMISSIVE=1, QUIET=not verbose)
    structure = parser.get_structure("protein", pdb_file)

    # 创建描述符计算器
    calculator = EnhancedResidueDescriptor(
        cutoff=cutoff,
        include_interaction_potentials=include_interaction_potentials
    )

    # 计算描述符
    descriptors, residues_info, feature_names = calculator.compute_descriptors(
        structure, normalize=normalize
    )

    if verbose and len(descriptors) > 0:
        info = calculator.get_feature_info()
        print(f"\n{'=' * 60}")
        print(f"增强版残基描述符计算完成")
        print(f"{'=' * 60}")
        print(f"残基数量: {len(residues_info)}")
        print(f"总特征维度: {descriptors.shape[1]}")
        print(f"  - 基础特征: {info['base_dimension']} 维")
        if include_interaction_potentials:
            print(f"  - 相互作用潜能: {info['interaction_dimension']} 维")
        print(f"描述符矩阵形状: {descriptors.shape}")

        # 统计信息
        print(f"\n描述符统计:")
        print(f"  均值范围: [{descriptors.mean():.4f}, {descriptors.mean():.4f}]")
        print(f"  标准差范围: [{descriptors.std():.4f}, {descriptors.std():.4f}]")
        print(f"  最小值: {descriptors.min():.4f}")
        print(f"  最大值: {descriptors.max():.4f}")
        sparsity = np.mean(descriptors == 0)
        print(f"  稀疏度: {sparsity:.4f}")

    return descriptors.tolist(), residues_info

# ============= 使用示例 =============

if __name__ == "__main__":
    x = cal_rACSF('protein/6LUQ.pdb')
    print(x)