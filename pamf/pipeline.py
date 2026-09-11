"""Public SMILES -> paired attachment fragment pipeline."""

from dataclasses import asdict, dataclass
import math
from rdkit import Chem, rdBase
from .chemistry import parse_smiles, analyze_bonds, make_fragments
from .geometry import conformers, torsion_variability
from .electronic import run_xtb, add_electronic_features
from .optimizer import select_cuts
from .scoring import score_candidates


@dataclass(frozen=True)
class PAMFConfig:
    mode: str = 'xtb'
    min_heavy_atoms: int = 3
    max_heavy_atoms: int = 25
    strict_max_size: bool = False
    conformer_count: int = 10
    random_seed: int = 42
    energy_window: float = 10.0
    cross_radius: int = 2
    xtb_timeout: float = 300
    xtb_threads: int = 1
    unpaired_electrons: int | None = None
    exact_candidate_limit: int = 14
    beam_width: int = 128
    rotatable_weight: float = 1.0
    boundary_weight: float = 0.5
    flexibility_weight: float = 0.5
    wbo_weight: float = 0.5
    cross_wbo_weight: float = 0.5
    fragment_penalty: float = 0.8
    oversize_penalty: float = 0.1
    imbalance_penalty: float = 0.05
    extra_protected_smarts: tuple[str, ...] = ()

    def __post_init__(self):
        if self.mode not in ('xtb', 'rules'):
            raise ValueError('mode must be xtb or rules')
        for name in ('min_heavy_atoms', 'max_heavy_atoms', 'conformer_count', 'xtb_threads', 'beam_width'):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        for name in ('random_seed', 'cross_radius', 'exact_candidate_limit'):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f'{name} must be a nonnegative integer')
        if self.max_heavy_atoms < self.min_heavy_atoms:
            raise ValueError('max_heavy_atoms must be >= min_heavy_atoms')
        if self.unpaired_electrons is not None and (type(self.unpaired_electrons) is not int or self.unpaired_electrons < 0):
            raise ValueError('unpaired_electrons must be a nonnegative integer')
        for name in ('rotatable_weight', 'boundary_weight', 'flexibility_weight', 'wbo_weight',
                     'cross_wbo_weight', 'fragment_penalty', 'oversize_penalty',
                     'imbalance_penalty', 'energy_window', 'xtb_timeout'):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if self.xtb_timeout <= 0:
            raise ValueError('xtb_timeout must be positive')


def fragment_smiles(smiles: str, config: PAMFConfig | None = None, *, reference=None) -> list[str]:
    """输入完整 SMILES，返回带成对连接编号 [k*] 的分解片段列表。

    默认使用 xTB 物理模式，与 decompose_smiles 保持一致；纯规则模式请显式
    传入 PAMFConfig(mode='rules')。不会自动降级或吞掉无效输入/计算错误。
    每个连接编号在返回列表中恰好出现两次，无切割时返回原分子组分，
    不添加虚构连接点。保留重复片段，以确保重组时不丢失分子结构。

    Example:
        >>> fragment_smiles('CCCCCC', PAMFConfig(mode='rules'))
        ['[1*]CCC', '[1*]CCC']

    如需评分、警告、连接表或计算元数据，请调用 decompose_smiles。
    """
    return decompose_smiles(smiles, config, reference=reference)['fragments']


def decompose_smiles(smiles, config=None, *, reference=None):
    """Return JSON-serializable fragments, connections, features and provenance.

    Reference: {chemical_class: {wbo: {mean, std}, cross_wbo: {mean, std}}}.
    Atom indices always refer to RDKit's parsed input, BEFORE canonical reordering.
    No tautomer, salt, pH, formal charge, isotope or atom-map standardization.
    """
    config = config or PAMFConfig()
    mol = parse_smiles(smiles)
    candidates, protected = analyze_bonds(mol, config.extra_protected_smarts)
    warnings = []
    geometry = electronics = None
    if config.mode == 'xtb' and candidates:
        if len(Chem.GetMolFrags(mol)) != 1:
            raise ValueError('xTB mode requires a connected molecule; select the desired ionic component explicitly or use rules mode')
        mol_h, conf_ids, geometry = conformers(mol, config.conformer_count,
                                              config.random_seed, config.energy_window)
        # AddHs appends atoms; this invariant protects the XYZ -> parent mapping.
        if [a.GetAtomicNum() for a in mol_h.GetAtoms()][:mol.GetNumAtoms()] != [a.GetAtomicNum() for a in mol.GetAtoms()]:
            raise RuntimeError('Hydrogen addition changed parent atom ordering')
        torsions = torsion_variability(mol_h, candidates, conf_ids)
        for row in candidates:
            row['torsion_variability'] = torsions[row['bond_idx']]
        data = run_xtb(mol_h, conf_ids[0], timeout=config.xtb_timeout,
                       threads=config.xtb_threads, unpaired=config.unpaired_electrons)
        add_electronic_features(mol, candidates, data, config.cross_radius)
        electronics = data['metadata']
        electronics['parent_to_xyz_index'] = list(range(1, mol.GetNumAtoms()+1))
        if len(conf_ids) < 2:
            warnings.append('Fewer than two retained conformers; torsional variability is unavailable.')
    elif config.mode == 'rules':
        warnings.append('Rules-only baseline: no quantum chemistry or conformer features were computed.')
    else:
        warnings.append('No eligible cut bonds; geometry and xTB were not needed.')
    warnings.extend(score_candidates(candidates, config, reference))
    cuts, optimization = select_cuts(mol, candidates, config)
    fragments, connections, rebuilt = make_fragments(mol, cuts)
    if not optimization['size_limit_satisfied']:
        warnings.append('Some fragments exceed max_heavy_atoms; the default maximum is a soft penalty. Use strict_max_size for a hard constraint.')
    if not optimization['optimality_proven']:
        warnings.append('Beam search is approximate; global optimality is not certified.')
    for row in candidates:
        row['selected'] = row['bond_idx'] in cuts
    return dict(schema_version=1, method='PAMF-v1', mode=config.mode,
                input_smiles=smiles, parent_smiles=Chem.MolToSmiles(mol, isomericSmiles=True),
                fragments=fragments, connections=connections, candidates=candidates,
                protected_bonds=protected, optimization=optimization,
                geometry=geometry, electronics=electronics,
                normalization=('training_reference' if reference is not None else 'within_molecule')
                if electronics is not None else None,
                reference=reference, config=asdict(config), rdkit_version=rdBase.rdkitVersion,
                atom_index_convention='0-based RDKit-parsed input SMILES; XYZ indices are 1-based',
                reconstructed_smiles=rebuilt, reconstruction_valid=True,
                warnings=sorted(set(warnings)))
