from rdkit import Chem
from rdkit.Chem import PandasTools
import pandas as pd
from typing import List, Tuple


def analyze_generated_molecules(train_smiles: List[str],
                                generated_smiles: List[str],
                                verbose: bool = True) -> Tuple[float, float, float]:
    """
    分析生成分子的质量和多样性

    Parameters:
    -----------
    train_smiles : List[str]
        训练集的SMILES列表
    generated_smiles : List[str]
        生成集的SMILES列表
    verbose : bool
        是否打印详细信息，默认为True

    Returns:
    --------
    Tuple[float, float, float]
        (有效性比例, 生成集内部重复比例, 与训练集重复比例)
    """

    # 1. 清洗并验证SMILES，同时去重
    def canonicalize_smiles(smiles_list):
        """验证并规范化SMILES，同时去重"""
        valid_smiles_set = set()
        invalid_count = 0

        for smi in smiles_list:
            if not smi or not isinstance(smi, str):
                invalid_count += 1
                continue

            try:
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    # 转换为规范SMILES
                    canonical_smi = Chem.MolToSmiles(mol, isomericSmiles=True)
                    valid_smiles_set.add(canonical_smi)
                else:
                    invalid_count += 1
            except:
                invalid_count += 1
                continue

        return list(valid_smiles_set), invalid_count

    # 2. 处理训练集
    train_valid, train_invalid = canonicalize_smiles(train_smiles)
    train_set = set(train_valid)

    # 3. 处理生成集
    gen_valid, gen_invalid = canonicalize_smiles(generated_smiles)
    gen_set = set(gen_valid)

    # 4. 计算各项指标
    total_gen = len(generated_smiles)
    valid_count = len(gen_valid)
    total_valid = len(gen_set)  # 去重后的有效分子数

    # 有效性比例（基于原始输入）
    validity_ratio = valid_count / total_gen if total_gen > 0 else 0.0

    # 生成集内部重复比例（重复分子数 / 总有效分子数）
    duplicate_count_in_gen = valid_count - total_valid
    internal_duplicate_ratio = duplicate_count_in_gen / valid_count if valid_count > 0 else 0.0

    # 与训练集的重复比例（重复分子数 / 去重后的生成集大小）
    overlap_with_train = len(gen_set.intersection(train_set))
    overlap_ratio = overlap_with_train / total_valid if total_valid > 0 else 0.0

    # 5. 打印详细信息
    if verbose:
        print("=" * 60)
        print("分子生成分析报告")
        print("=" * 60)
        print(f"训练集分子总数: {len(train_smiles)}")
        print(f"训练集有效分子数: {len(train_set)}")
        print(f"训练集中无效分子数: {len(train_smiles) - len(train_set)}")
        print("-" * 60)
        print(f"生成集分子总数: {total_gen}")
        print(f"生成集有效分子数: {valid_count}")
        print(f"生成集无效分子数: {gen_invalid}")
        print(f"生成集去重后有效分子数: {total_valid}")
        print("-" * 60)
        print(f"📊 有效性比例: {validity_ratio:.2%} ({valid_count}/{total_gen})")
        print(f"📊 生成集内部重复率: {internal_duplicate_ratio:.2%} ({duplicate_count_in_gen}/{valid_count})")
        print(f"📊 与训练集重复率: {overlap_ratio:.2%} ({overlap_with_train}/{total_valid})")
        print("=" * 60)

        # 额外信息
        if overlap_with_train > 0:
            print(f"⚠️  发现 {overlap_with_train} 个分子与训练集重复")
        if duplicate_count_in_gen > 0:
            print(f"⚠️  生成集中有 {duplicate_count_in_gen} 个重复分子")
        print("=" * 60)

    return validity_ratio, internal_duplicate_ratio, overlap_ratio


# ============ 使用示例 ============
if __name__ == "__main__":
    # 示例数据
    train_smiles = [
        "CC(=O)OC1=CC=CC=C1C(=O)O",  # 阿司匹林
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # 咖啡因
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # 布洛芬
        "CC1=C(C(=O)NC2=CC=CC=C2)C=CC=C1",  # 扑热息痛
        "invalid_smiles_string",  # 无效分子
    ]

    generated_smiles = [
        "CC(=O)OC1=CC=CC=C1C(=O)O",  # 与训练集重复
        "CC(=O)OC1=CC=CC=C1C(=O)O",  # 内部重复
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # 与训练集重复
        "C1=CC=CC=C1",  # 新分子（苯）
        "C1=CC=CC=C1C(=O)O",  # 新分子（苯甲酸）
        "C1=CC=CC=C1C(=O)OC",  # 新分子（苯甲酸甲酯）
        "invalid_molecule",  # 无效分子
        "C1=CC=CC=C1",  # 内部重复
    ]

    # 调用分析函数
    validity, internal_dup, overlap = analyze_generated_molecules(
        train_smiles,
        generated_smiles,
        verbose=True
    )

    # 如果只需要数值
    print("\n仅数值输出:")
    print(f"有效性: {validity:.2%}")
    print(f"内部重复率: {internal_dup:.2%}")
    print(f"与训练集重复率: {overlap:.2%}")