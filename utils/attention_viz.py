"""
Input:
  - 标准库: os
  - 第三方: numpy, matplotlib
Output:
  - plot_attention_boxplot: 按边类型绘制注意力权重分布箱线图
Position: 提供 GAT 注意力权重的可视化分析功能，用于解释模型决策机制

⚠️ 提醒：修改本文件后务必：
   1. 更新上方 Input/Output/Position 说明
   2. 更新所属文件夹的 README.md（如果影响模块结构）
   3. 更新根目录 README.md（如果影响核心架构）
"""

from __future__ import annotations

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.font_manager import FontProperties

# ============================================================
# 全局样式配置（与 figures/plot_figures.py 保持一致）
# ============================================================

# 字体：中文宋体，英文/数字用 Times New Roman
plt.rcParams['font.sans-serif'] = ['SimSun', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'

FONT_TNR = FontProperties(family='Times New Roman')
FONT_TNR_BOLD = FontProperties(family='Times New Roman', weight='bold')

COLORS = {
    'bg':    '#FAF9F5',   # 暖灰白背景
    'grid':  '#E8E6DC',   # 网格线
    'spine': '#666666',   # 坐标轴边框
    'text':  '#000000',   # 文字颜色
}

# 边类型配色
EDGE_COLORS = {
    "N→N": "#5B9A5B",  # 深绿色
    "N→F": "#C17832",  # 深橙色
    "F→N": "#4E81B1",  # 蓝色（与 BPNN 主色一致）
    "F→F": "#C15A38",  # 红色（与 GAT 主色一致）
}

SAVE_DPI = 600


def _apply_style(ax: Axes) -> None:
    """统一坐标轴样式（与 plot_figures.py 的 apply_style 一致）。"""
    ax.set_facecolor(COLORS['bg'])
    ax.figure.set_facecolor('#FFFFFF')

    # 网格
    ax.grid(True, axis='y', linestyle='--', linewidth=0.6,
            color=COLORS['grid'], alpha=0.8)
    ax.set_axisbelow(True)

    # 边框
    for spine in ax.spines.values():
        spine.set_color(COLORS['spine'])
        spine.set_linewidth(0.8)

    # 刻度
    ax.tick_params(colors=COLORS['text'], labelsize=10)


def plot_attention_boxplot(attn_data: dict, save_dir: str = "figures",
                           dimension: int | None = None,
                           fault_rate: float | None = None) -> str:
    """
    按边类型绘制注意力权重分布箱线图

    将注意力权重按边两端节点状态分为 4 类（N→N, N→F, F→N, F→F），
    绘制箱线图展示各类型的分布差异，论证 GAT 是否学到了区分故障/正常节点的能力。

    Args:
        attn_data: GAT.get_attention_weights() 的返回值，包含：
            - "by_type": {"N→N": array, "N→F": array, "F→N": array, "F→F": array}
            - "n_samples": 样本数
            - "n_heads": 注意力头数
        save_dir: 图片保存目录
        dimension: 超立方体维度（用于图标题）
        fault_rate: 故障率（用于图标题）

    Returns:
        保存的图片文件路径
    """
    by_type = attn_data["by_type"]
    n_samples = attn_data["n_samples"]
    n_heads = attn_data["n_heads"]

    # 准备数据和标签（只画有数据的类型）
    labels: list[str] = []
    data: list[np.ndarray] = []
    counts: list[int] = []
    for key in ["N→N", "N→F", "F→N", "F→F"]:
        arr = by_type[key]
        if len(arr) > 0:
            labels.append(key)
            data.append(arr)
            counts.append(len(arr))

    fig, ax = plt.subplots(figsize=(8, 5.5))
    _apply_style(ax)

    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True,
                    showfliers=False, widths=0.5,
                    medianprops=dict(color='black', linewidth=1.5))

    # 配色：按边类型区分
    for patch, label in zip(bp['boxes'], labels):
        patch.set_facecolor(EDGE_COLORS.get(label, "#adb5bd"))
        patch.set_alpha(0.85)

    # 在箱体上方标注样本数和均值
    q75_max = max(np.percentile(d, 75) for d in data)
    for i, (label, arr) in enumerate(zip(labels, data)):
        mean_val = np.mean(arr)
        ax.text(i + 1, q75_max + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02,
                f'$\\mathrm{{n{{=}}{counts[i]}}}$\n'
                f'$\\mu\\mathrm{{{{=}}{mean_val:.4f}}}$',
                ha='center', va='bottom', fontsize=9, color='#495057')

    ax.set_ylabel('注意力权重（多头平均）', fontsize=12, color='#000000',
                  fontweight='bold', labelpad=8)
    ax.set_xlabel('边类型（源节点 → 目标节点）', fontsize=12, color='#000000',
                  fontweight='bold', labelpad=8)

    # 标题
    title_parts = ["GAT 注意力权重分布"]
    sub_parts = []
    if dimension is not None:
        sub_parts.append(f"$\\mathrm{{d{{=}}{dimension}}}$")
    if fault_rate is not None:
        sub_parts.append(f"故障率$\\mathrm{{{{=}}{fault_rate}}}$")
    sub_parts.append(f"$\\mathrm{{{n_samples}}}$样本")
    sub_parts.append(f"$\\mathrm{{{n_heads}}}$头平均")
    title = title_parts[0] + "（" + "，".join(sub_parts) + "）"
    ax.set_title(title, fontsize=13, color='#000000', pad=12)

    plt.tight_layout()

    # tight_layout 后设置字体
    for lbl in ax.get_yticklabels():
        lbl.set_fontproperties(FONT_TNR)

    # 保存
    os.makedirs(save_dir, exist_ok=True)
    filename = "attention_boxplot"
    if dimension is not None:
        filename += f"_d{dimension}"
    if fault_rate is not None:
        filename += f"_f{fault_rate}"
    filename += ".png"
    filepath = os.path.join(save_dir, filename)
    fig.savefig(filepath, dpi=SAVE_DPI, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)

    return filepath
