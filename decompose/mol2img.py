"""
第二步：SMILES转2D图可视化
功能：将分子和子结构可视化，用不同颜色区分不同部分
（修复版：移除IPython依赖）
"""

import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from io import BytesIO
import base64
from typing import List, Dict, Set, Tuple, Optional
import numpy as np


class SubstructureVisualizer:
    """子结构可视化器"""

    def __init__(self, figsize=(12, 8)):
        self.figsize = figsize

        # 定义颜色方案 (RGB)
        self.colors = {
            'ring_system': (0.8, 0.2, 0.2, 0.6),    # 红色
            'bridge': (0.2, 0.6, 0.2, 0.6),         # 绿色
            'functional': (0.2, 0.2, 0.8, 0.6),     # 蓝色
            'side_chain': (0.8, 0.6, 0.2, 0.6),     # 橙色
            'linker': (0.6, 0.2, 0.8, 0.6),         # 紫色
            'attachment': (1.0, 0.0, 0.0, 0.8),     # 红色高亮
        }

    def draw_molecule_with_atom_colors(self,
                                       mol: Chem.Mol,
                                       atom_colors: Dict[int, Tuple],
                                       title: str = "",
                                       save_path: str = None) -> plt.Figure:
        """
        用自定义颜色绘制分子

        Args:
            mol: RDKit分子对象
            atom_colors: 原子索引 -> RGB颜色映射
            title: 图表标题
            save_path: 保存路径
        """
        # 生成2D坐标
        rdDepictor.Compute2DCoords(mol)

        # 准备高亮数据
        highlight_atoms = []
        highlight_colors = []

        for atom_idx, color in atom_colors.items():
            if atom_idx < mol.GetNumAtoms():
                highlight_atoms.append(atom_idx)
                highlight_colors.append(color)

        # 创建绘图器
        drawer = rdMolDraw2D.MolDraw2DCairo(800, 600)
        drawer.drawOptions().setBackgroundColour((1, 1, 1))

        # 绘制分子
        if highlight_atoms:
            drawer.DrawMolecule(mol, highlightAtoms=highlight_atoms,
                              highlightColors=highlight_colors)
        else:
            drawer.DrawMolecule(mol)

        drawer.FinishDrawing()

        # 获取图像数据
        png_data = drawer.GetDrawingText()

        # 使用matplotlib显示
        fig, ax = plt.subplots(figsize=self.figsize)

        # 将PNG数据转换为图像
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(png_data))
        ax.imshow(np.array(img))
        ax.axis('off')

        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        else:
            ax.set_title(f"分子可视化\n{Chem.MolToSmiles(mol)}", fontsize=14)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"图像已保存: {save_path}")

        return fig

    def draw_substructure(self,
                         substructure: Dict,
                         title: str = None,
                         save_path: str = None) -> plt.Figure:
        """
        绘制单个子结构

        Args:
            substructure: 子结构字典，包含'smiles'
            title: 标题
            save_path: 保存路径
        """
        smiles = substructure['smiles']
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            print(f"无法解析SMILES: {smiles}")
            return None

        # 生成2D坐标
        rdDepictor.Compute2DCoords(mol)

        # 找出虚拟原子（原子序数为0）
        virtual_atoms = []
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 0:
                virtual_atoms.append(atom.GetIdx())

        # 绘制
        fig, ax = plt.subplots(figsize=(8, 6))

        drawer = rdMolDraw2D.MolDraw2DCairo(600, 400)
        drawer.drawOptions().setBackgroundColour((1, 1, 1))

        if virtual_atoms:
            # 高亮虚拟原子（灰色）
            drawer.DrawMolecule(mol, highlightAtoms=virtual_atoms,
                              highlightColors=[(0.5, 0.5, 0.5, 0.8)] * len(virtual_atoms))
        else:
            drawer.DrawMolecule(mol)

        drawer.FinishDrawing()

        png_data = drawer.GetDrawingText()
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(png_data))
        ax.imshow(np.array(img))
        ax.axis('off')

        if title is None:
            title = f"{substructure['type']}/{substructure['name']}\n{substructure['smiles']}"

        if substructure.get('connections'):
            conn_str = ", ".join([f"{c['virtual_atom']}{c['bond_type']}"
                                 for c in substructure['connections']])
            title += f"\n连接: {conn_str}"

        ax.set_title(title, fontsize=12)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"图像已保存: {save_path}")

        return fig


def visualize_decomposition(result: Dict, save_path: str = None) -> plt.Figure:
    """
    可视化分解结果

    Args:
        result: decompose_to_scaffold_and_side_chains 的返回值
        save_path: 保存路径
    """
    mol = result['original_mol']
    scaffold_atoms = result['scaffold_atoms']
    side_chain_components = result['side_chain_components']
    attachment_points = result['attachment_points']

    # 构建原子颜色映射
    atom_colors = {}

    # 为骨架原子分配颜色
    for atom_idx in scaffold_atoms:
        # 检查是否是连接点
        is_attachment = any(ap['scaffold_atom'] == atom_idx for ap in attachment_points)
        if is_attachment:
            atom_colors[atom_idx] = (1.0, 0.5, 0.5, 0.8)  # 亮红色（连接点）
        else:
            atom_colors[atom_idx] = (1.0, 0.2, 0.2, 0.6)  # 暗红色（骨架）

    # 为侧链原子分配颜色
    side_chain_colors = [
        (0.2, 0.4, 0.8, 0.6),   # 蓝色
        (0.2, 0.6, 0.8, 0.6),   # 天蓝色
        (0.2, 0.8, 0.8, 0.6),   # 青色
        (0.4, 0.2, 0.8, 0.6),   # 紫蓝色
    ]

    for i, component in enumerate(side_chain_components):
        color = side_chain_colors[i % len(side_chain_colors)]
        for atom_idx in component:
            atom_colors[atom_idx] = color

    # 生成2D坐标
    rdDepictor.Compute2DCoords(mol)

    # 创建绘图器
    drawer = rdMolDraw2D.MolDraw2DCairo(1000, 800)
    drawer.drawOptions().setBackgroundColour((1, 1, 1))

    # 准备高亮数据
    highlight_atoms = []
    highlight_colors = []
    for atom_idx, color in atom_colors.items():
        highlight_atoms.append(atom_idx)
        highlight_colors.append(color)

    # 绘制分子
    if highlight_atoms:
        drawer.DrawMolecule(mol, highlightAtoms=highlight_atoms,
                          highlightColors=highlight_colors)
    else:
        drawer.DrawMolecule(mol)

    drawer.FinishDrawing()

    # 获取图像
    png_data = drawer.GetDrawingText()

    # 使用matplotlib显示
    fig, ax = plt.subplots(figsize=(12, 10))

    from PIL import Image
    import io
    img = Image.open(io.BytesIO(png_data))
    ax.imshow(np.array(img))
    ax.axis('off')

    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=(1.0, 0.2, 0.2, 0.7), label='骨架原子（环系统）'),
        Patch(facecolor=(1.0, 0.5, 0.5, 0.7), label='骨架连接点'),
        Patch(facecolor=(0.2, 0.4, 0.8, 0.6), label='侧链')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12)

    smiles = Chem.MolToSmiles(mol)
    title = f"分子分解可视化\n{smiles}\n骨架原子数: {len(scaffold_atoms)}, 侧链分量数: {len(side_chain_components)}"
    ax.set_title(title, fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图像已保存: {save_path}")

    return fig


def visualize_atom_groups(mol: Chem.Mol,
                         atom_groups: List[Dict],
                         title: str = "",
                         save_path: str = None) -> plt.Figure:
    """
    可视化原子分组

    Args:
        mol: RDKit分子对象
        atom_groups: 原子分组列表，每个包含'atoms'和'name'
        title: 标题
        save_path: 保存路径
    """
    # 分配颜色
    group_colors = [
        (0.8, 0.2, 0.2, 0.6),  # 红
        (0.2, 0.6, 0.2, 0.6),  # 绿
        (0.2, 0.2, 0.8, 0.6),  # 蓝
        (0.8, 0.6, 0.2, 0.6),  # 橙
        (0.6, 0.2, 0.8, 0.6),  # 紫
        (0.2, 0.8, 0.6, 0.6),  # 青
    ]

    atom_colors = {}
    for i, group in enumerate(atom_groups):
        color = group_colors[i % len(group_colors)]
        for atom_idx in group['atoms']:
            atom_colors[atom_idx] = color

    # 生成2D坐标
    rdDepictor.Compute2DCoords(mol)

    # 创建绘图器
    drawer = rdMolDraw2D.MolDraw2DCairo(1000, 800)
    drawer.drawOptions().setBackgroundColour((1, 1, 1))

    highlight_atoms = []
    highlight_colors = []
    for atom_idx, color in atom_colors.items():
        highlight_atoms.append(atom_idx)
        highlight_colors.append(color)

    drawer.DrawMolecule(mol, highlightAtoms=highlight_atoms, highlightColors=highlight_colors)
    drawer.FinishDrawing()

    png_data = drawer.GetDrawingText()

    fig, ax = plt.subplots(figsize=(12, 10))
    from PIL import Image
    import io
    img = Image.open(io.BytesIO(png_data))
    ax.imshow(np.array(img))
    ax.axis('off')

    if title:
        ax.set_title(title, fontsize=14)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图像已保存: {save_path}")

    return fig


# 简化版本：直接使用RDKit的绘图功能
def draw_molecule_simple(mol: Chem.Mol,
                        highlight_atoms: List[int] = None,
                        highlight_colors: List[str] = None,
                        save_path: str = None):
    """
    简化版分子绘图（使用RDKit内置功能）

    Args:
        mol: RDKit分子对象
        highlight_atoms: 要高亮的原子索引列表
        highlight_colors: 高亮颜色列表（可选）
        save_path: 保存路径
    """
    if highlight_atoms and highlight_colors:
        img = Draw.MolToImage(mol, highlightAtoms=highlight_atoms,
                             highlightColors=highlight_colors, size=(800, 600))
    else:
        img = Draw.MolToImage(mol, size=(800, 600))

    if save_path:
        img.save(save_path)
        print(f"图像已保存: {save_path}")

    return img


# 集成测试
def test_visualization():
    """测试可视化功能"""
    from test1 import decompose_to_scaffold_and_side_chains

    test_molecules = [
        ("Oc1ccccc1", "苯酚"),
        ("C1=CC=C(C=C1)CC2=CC=CC=C2", "二苯甲烷"),
    ]

    for smiles, name in test_molecules:
        print(f"\n处理: {name}")
        try:
            result = decompose_to_scaffold_and_side_chains(smiles)

            # 可视化分解结果
            fig = visualize_decomposition(result, save_path=f"{name}_decomposition.png")
            plt.close(fig)
            print(f"✓ {name} 可视化完成")
        except Exception as e:
            print(f"✗ {name} 可视化失败: {e}")


if __name__ == "__main__":
    print("RDKit 版本:", Chem.rdBase.rdkitVersion)
    print("Matplotlib 版本:", plt.matplotlib.__version__)
    test_visualization()