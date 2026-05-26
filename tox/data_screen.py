import pandas as pd
import numpy as np
from itertools import combinations
from tqdm import tqdm


def load_tox21_data(csv_path: str):
    """
    加载 Tox21 数据集

    Args:
        csv_path: Tox21 数据集的 CSV 文件路径

    Returns:
        df: 包含所有数据的 DataFrame
        smiles_col: SMILES 列名（通常为 'smiles'）
        label_cols: 12个标签列名列表
    """
    df = pd.read_csv(csv_path)

    # Tox21 的 12 个标签列
    label_cols = [
        'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase',
        'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma',
        'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
    ]

    # 确认 SMILES 列名（通常是 'smiles' 或 'SMILES'）
    if 'smiles' in df.columns:
        smiles_col = 'smiles'
    elif 'SMILES' in df.columns:
        smiles_col = 'SMILES'
    else:
        # 如果都不是，尝试找到包含 'smiles' 的列
        smiles_col = [col for col in df.columns if 'smiles' in col.lower()][0]

    print(f"数据集形状: {df.shape}")
    print(f"SMILES 列: {smiles_col}")
    print(f"标签列: {label_cols}")

    return df, smiles_col, label_cols


def analyze_label_coverage(df, label_cols):
    """
    分析每个标签的覆盖情况

    Args:
        df: DataFrame
        label_cols: 标签列名列表

    Returns:
        coverage_df: 每个标签的覆盖率统计
    """
    coverage = {}
    for col in label_cols:
        # Tox21 中缺失值通常用空字符串或 NaN 表示
        # 有数据的行 = 非空且不为空字符串
        has_data = (~df[col].isna()) & (df[col].astype(str).str.strip() != '')
        n_has_data = has_data.sum()
        coverage[col] = {
            'count': n_has_data,
            'percentage': n_has_data / len(df) * 100
        }

    coverage_df = pd.DataFrame(coverage).T.sort_values('percentage', ascending=False)

    print("\n=== 各标签覆盖率 ===")
    print(coverage_df)

    return coverage_df


def find_best_label_combinations(df, label_cols, target_labels: int = 3,
                                 min_coverage: float = 90.0,
                                 max_coverage: float = 95.0,
                                 return_all_valid: bool = False):
    """
    寻找最佳标签组合，使得同时拥有这些标签数据的分子比例在目标范围内

    Args:
        df: DataFrame
        label_cols: 标签列名列表
        target_labels: 希望选择的标签数量 (2, 3, 4)
        min_coverage: 最小覆盖率 (%)
        max_coverage: 最大覆盖率 (%)
        return_all_valid: 是否返回所有符合条件的组合

    Returns:
        如果 return_all_valid=False: 返回最佳组合的 (selected_labels, filtered_df, coverage)
        如果 return_all_valid=True: 返回所有符合条件的组合列表
    """
    n_molecules = len(df)
    results = []

    # 预先计算每个分子哪些标签有数据
    # 创建布尔掩码矩阵 [n_molecules, n_labels]
    mask_matrix = pd.DataFrame(index=df.index)
    for col in label_cols:
        mask_matrix[col] = (~df[col].isna()) & (df[col].astype(str).str.strip() != '')

    print(f"\n=== 搜索 {target_labels} 个标签的组合 ===")

    # 遍历所有组合
    for combo in tqdm(list(combinations(label_cols, target_labels)),
                      desc=f"检查 {target_labels} 标签组合"):
        # 计算同时拥有所有这些标签数据的分子
        mask = mask_matrix[list(combo)].all(axis=1)
        n_valid = mask.sum()
        coverage = n_valid / n_molecules * 100

        if min_coverage <= coverage <= max_coverage:
            # 提取对应的分子和标签数据
            filtered_df = df[mask].copy()
            # 只保留选中的标签列和 SMILES
            selected_cols = [col for col in df.columns if col not in label_cols] + list(combo)
            filtered_df = filtered_df[selected_cols]

            # 将标签值转换为 0/1 整数（处理可能的字符串值）
            for col in combo:
                filtered_df[col] = pd.to_numeric(filtered_df[col], errors='coerce').fillna(0).astype(int)

            results.append({
                'labels': combo,
                'n_labels': target_labels,
                'coverage': coverage,
                'n_molecules': n_valid,
                'filtered_df': filtered_df,
                'mask': mask
            })

    if len(results) == 0:
        print(f"未找到覆盖率为 {min_coverage}-{max_coverage}% 的 {target_labels} 标签组合")
        print(f"尝试调整覆盖率范围...")

        # 如果没有找到，返回最接近的组合
        for combo in combinations(label_cols, target_labels):
            mask = mask_matrix[list(combo)].all(axis=1)
            coverage = mask.sum() / n_molecules * 100
            results.append({
                'labels': combo,
                'n_labels': target_labels,
                'coverage': coverage,
                'n_molecules': mask.sum(),
                'filtered_df': None,
                'mask': mask
            })

        # 按覆盖率排序，返回最接近 90% 的
        results.sort(key=lambda x: abs(x['coverage'] - 90))
        best = results[0]
        print(f"找到最接近的组合: {best['labels']}, 覆盖率: {best['coverage']:.2f}%")

        if return_all_valid:
            return results
        return best

    # 按覆盖率排序
    results.sort(key=lambda x: x['coverage'])

    print(f"\n找到 {len(results)} 个符合条件的组合:")
    for r in results:
        print(f"  {r['labels']} -> 覆盖 {r['n_molecules']} 个分子 ({r['coverage']:.2f}%)")

    if return_all_valid:
        return results
    return results[0]  # 返回覆盖率最小的那个（最接近下限）


def analyze_multi_label_distribution(filtered_df, label_cols):
    """
    分析多标签分布情况（一个分子可能有多个阳性标签）

    Args:
        filtered_df: 筛选后的 DataFrame
        label_cols: 选中的标签列
    """
    print("\n=== 多标签分布分析 ===")

    # 计算每个分子的阳性标签数量
    positive_counts = filtered_df[label_cols[0]].sum(axis=0)

    print(f"总分子数: {len(filtered_df)}")
    print(f"每个分子的阳性标签数量分布:")
    for i in range(len(label_cols) + 1):
        count = (positive_counts == i).sum()
        if count > 0:
            print(f"  {i} 个阳性: {count} 个分子 ({count / len(filtered_df) * 100:.2f}%)")

    return positive_counts


def save_filtered_data(filtered_df, output_path: str):
    """
    保存筛选后的数据
    """
    filtered_df.to_csv(output_path, index=False)
    print(f"\n数据已保存至: {output_path}")


def main():
    """主函数：完整的筛选流程"""

    # ========== 配置参数 ==========
    CSV_PATH = "../data/tox21.csv"  # 替换为你的 Tox21 数据文件路径
    OUTPUT_PATH = "tox21_filtered.csv"
    TARGET_N_LABELS = 3  # 希望选择的标签数量 (2, 3, 4)
    MIN_COVERAGE = 80.0  # 最小覆盖率 (%)
    MAX_COVERAGE = 95.0  # 最大覆盖率 (%)

    # ========== 1. 加载数据 ==========
    print("=" * 60)
    print("步骤 1: 加载 Tox21 数据集")
    print("=" * 60)
    df, smiles_col, label_cols = load_tox21_data(CSV_PATH)

    # ========== 2. 分析单标签覆盖率 ==========
    print("\n" + "=" * 60)
    print("步骤 2: 分析各标签覆盖率")
    print("=" * 60)
    coverage_df = analyze_label_coverage(df, label_cols)

    # ========== 3. 寻找最佳标签组合 ==========
    print("\n" + "=" * 60)
    print(f"步骤 3: 寻找 {TARGET_N_LABELS} 个标签的最佳组合")
    print(f"目标覆盖率: {MIN_COVERAGE}% - {MAX_COVERAGE}%")
    print("=" * 60)

    best_result = find_best_label_combinations(
        df, label_cols,
        target_labels=TARGET_N_LABELS,
        min_coverage=MIN_COVERAGE,
        max_coverage=MAX_COVERAGE,
        return_all_valid=False
    )

    if best_result['filtered_df'] is None:
        print("\n需要重新运行，获取最接近的组合...")
        return

    # ========== 4. 分析筛选结果 ==========
    print("\n" + "=" * 60)
    print("步骤 4: 分析筛选结果")
    print("=" * 60)

    selected_labels = best_result['labels']
    filtered_df = best_result['filtered_df']

    print(f"\n选中的标签: {selected_labels}")
    print(f"覆盖分子数: {best_result['n_molecules']} / {len(df)}")
    print(f"覆盖率: {best_result['coverage']:.2f}%")

    # 分析多标签分布
    positive_counts = analyze_multi_label_distribution(filtered_df, selected_labels)

    # ========== 5. 保存数据 ==========
    print("\n" + "=" * 60)
    print("步骤 5: 保存筛选后的数据")
    print("=" * 60)

    save_filtered_data(filtered_df, OUTPUT_PATH)

    # ========== 6. 输出摘要 ==========
    print("\n" + "=" * 60)
    print("筛选摘要")
    print("=" * 60)
    print(f"""
    原始数据:
      - 总分子数: {len(df)}
      - 标签数: {len(label_cols)}

    筛选后数据:
      - 选中的标签: {selected_labels}
      - 筛选出的分子数: {len(filtered_df)}
      - 覆盖率: {best_result['coverage']:.2f}%
      - 阳性样本分布: 
          每个分子平均阳性标签数: {positive_counts.mean():.2f}
          至少有1个阳性标签的分子: {(positive_counts > 0).sum()} ({(positive_counts > 0).sum() / len(filtered_df) * 100:.1f}%)

    输出文件: {OUTPUT_PATH}
    """)

    return filtered_df, selected_labels


if __name__ == "__main__":
    filtered_data, selected_labels = main()