"""
Input: 无（实验数据直接嵌入代码中）
Output: figures/ 下的对比分析图片（compare_fig*.png）
Position: GAT vs GATWF 多维度实验对比分析绘图脚本

用法:
    python compare.py --fig 1       # 维度扩展性准确率对比
    python compare.py --fig 2       # 故障率鲁棒性准确率对比
    python compare.py --fig 3       # 样本效率准确率对比
    python compare.py --fig 4       # 消融实验对比（双模型并排柱状图）
    python compare.py --fig 5       # 综合雷达图（五维评测）
    python compare.py --fig 6       # 效率对比（训练耗时 + 推理时延）
    python compare.py --fig all     # 绘制全部图

⚠️ 提醒：修改本文件后务必：
   1. 更新上方 Input/Output/Position 说明
   2. 如有新指标，同步更新实验数据区
"""

import argparse
import sys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.axes import Axes
from matplotlib.font_manager import FontProperties
import numpy as np

# ============================================================
# 全局样式配置
# ============================================================

plt.rcParams['font.sans-serif'] = ['SimSun', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'

FONT_TNR = FontProperties(family='Times New Roman')
FONT_TNR_BOLD = FontProperties(family='Times New Roman', weight='bold')

COLORS = {
    'GAT':   '#C15A38',   # GAT 主色（橙红）
    'GATWF': '#3A7D44',   # GATWF 主色（深绿）
    'bg':    '#FAF9F5',   # 暖灰白背景
    'grid':  '#E8E6DC',   # 网格线
    'spine': '#666666',   # 坐标轴边框
    'text':  '#000000',   # 文字颜色
}

MARKERS = {
    'GAT':   'o',   # 实心圆
    'GATWF': 's',   # 实心方块
}

SAVE_DPI = 600
FIG_DIR = Path(__file__).parent / 'figures'


# ============================================================
# 实验数据（全部嵌入）
# ============================================================

# A. 维度扩展性（故障率 25%，样本量 5000）
DIM_DATA = {
    'dims': [4, 5, 6, 7, 8],
    'node_counts': {4: 16, 5: 32, 6: 64, 7: 128, 8: 256},
    'GAT': {
        'accuracy':   [98.60, 97.60, 95.40, 93.60, 92.80],
        'f1':         [99.84, 99.84, 99.82, 99.88, 99.92],
        'precision':  [99.84, 99.84, 99.89, 99.89, 99.91],
        'recall':     [99.85, 99.83, 99.75, 99.86, 99.93],
        'n_params':   [278562, 279586, 280610, 281634, 282658],
        'train_time': [41.50, 94.43, 142.52, 439.40, 632.41],
        'infer_ms':   [1.533, 2.090, 1.683, 2.020, 2.542],
    },
    'GATWF': {
        'accuracy':   [99.00, 99.60, 99.80, 99.60, 98.80],
        'f1':         [99.88, 99.98, 99.99, 99.99, 99.99],
        'precision':  [99.92, 99.96, 100.00, 99.99, 99.99],
        'recall':     [99.85, 100.00, 99.99, 99.99, 99.99],
        'n_params':   [280098, 281122, 282146, 283170, 284194],
        'train_time': [27.37, 51.49, 150.24, 373.60, 1543.26],
        'infer_ms':   [1.359, 1.763, 2.451, 3.021, 3.819],
    },
}

# B. 故障率鲁棒性（d=6，样本量 5000）
FAULT_DATA = {
    'fault_rates': [10, 20, 25, 30, 40, 50],
    'GAT': {
        'accuracy':   [90.80, 94.60, 95.40, 97.00, 95.20, 87.80],
        'f1':         [99.19, 99.76, 99.82, 99.90, 99.87, 99.69],
        'train_time': [99.25, 230.88, 142.52, 152.77, 281.26, 265.63],
        'infer_ms':   [1.914, 2.154, 1.683, 2.302, 2.141, 2.358],
    },
    'GATWF': {
        'accuracy':   [91.00, 99.00, 99.80, 99.40, 96.40, 95.20],
        'f1':         [99.21, 99.95, 99.99, 99.98, 99.91, 99.89],
        'train_time': [143.26, 128.37, 150.24, 243.73, 366.79, 516.53],
        'infer_ms':   [7.054, 2.713, 2.451, 2.521, 2.597, 3.209],
    },
}

# C. 样本效率（d=6，故障率 25%）
SAMPLE_DATA = {
    'n_samples': [500, 1000, 2000, 5000, 10000],
    'GAT': {
        'accuracy':   [100.00, 95.00, 96.50, 95.40, 96.30],
        'f1':         [100.00, 99.80, 99.89, 99.82, 99.86],
        'train_time': [31.75, 45.74, 107.33, 142.52, 305.30],
        'infer_ms':   [1.433, 1.575, 1.981, 1.683, 2.259],
    },
    'GATWF': {
        'accuracy':   [100.00, 97.00, 97.50, 99.80, 99.40],
        'f1':         [100.00, 99.86, 99.91, 99.99, 99.98],
        'train_time': [40.55, 63.20, 110.68, 150.24, 335.30],
        'infer_ms':   [2.637, 2.536, 2.713, 2.451, 2.562],
    },
}

# D. 消融实验（d=6，故障率 25%，样本量 5000）
ABLATION_GAT = {
    'labels':   ['GAT\n（完整基线）', 'GAT-1layer\n（2层→1层）', 'GAT-1head\n（8头→1头）', 'GAT-unidir\n（双向→单向）'],
    'accuracy': [95.40, 92.40, 82.60, 48.60],
    'delta':    [None, -3.00, -12.80, -46.80],
    'n_params': [280610, 15906, 5510, 277538],
}

ABLATION_GATWF = {
    'labels':   ['GATWF\n（完整基线）', 'GATWF-1layer\n（2层→1层）', 'GATWF-1head\n（8头→1头）', 'GATWF-noWF\n（=GAT基线）'],
    'accuracy': [99.80, 99.80, 93.20, 95.40],
    'delta':    [None, 0.00, -6.60, -4.40],
    'n_params': [282146, 17442, 5702, 280610],
}


# ============================================================
# 通用辅助函数
# ============================================================

def apply_style(ax: Axes) -> None:
    """统一坐标轴样式。"""
    ax.set_facecolor(COLORS['bg'])
    ax.figure.set_facecolor('#FFFFFF')
    ax.grid(True, linestyle='--', linewidth=0.6, color=COLORS['grid'], alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(COLORS['spine'])
        spine.set_linewidth(0.8)
    ax.tick_params(colors=COLORS['text'], labelsize=10)


def set_tnr_ticks(ax: Axes) -> None:
    """将 Y 轴刻度字体设为 Times New Roman。"""
    for label in ax.get_yticklabels():
        label.set_fontproperties(FONT_TNR)


def set_legend_tnr(legend) -> None:
    """将图例文字设为 Times New Roman。"""
    for text in legend.get_texts():
        text.set_fontproperties(FONT_TNR)


def make_legend(ax: Axes, loc: str = 'lower left') -> None:
    """添加统一样式图例。"""
    legend = ax.legend(loc=loc, fontsize=11, frameon=True,
                       fancybox=False, edgecolor=COLORS['grid'],
                       facecolor='white', framealpha=0.9)
    legend.get_frame().set_linewidth(0.8)
    set_legend_tnr(legend)


# ============================================================
# 图1：维度扩展性 — GAT vs GATWF 准确率对比
# ============================================================

def plot_fig1() -> None:
    """compare_fig1: 不同维度下GAT与GATWF诊断准确率对比折线图。"""
    dims = np.array(DIM_DATA['dims'])
    node_counts = DIM_DATA['node_counts']
    acc_gat   = np.array(DIM_DATA['GAT']['accuracy'])
    acc_gatwf = np.array(DIM_DATA['GATWF']['accuracy'])

    fig, ax = plt.subplots(figsize=(8, 5.5))
    apply_style(ax)

    # 填充 GATWF 领先区域
    ax.fill_between(dims, acc_gat, acc_gatwf,
                    where=(acc_gatwf >= acc_gat).tolist(),
                    color=COLORS['GATWF'], alpha=0.08, interpolate=True)

    # GATWF 折线
    ax.plot(dims, acc_gatwf,
            color=COLORS['GATWF'], marker=MARKERS['GATWF'],
            markersize=8, linewidth=2.2, markeredgecolor='white',
            markeredgewidth=1.2, label='GATWF', zorder=5)

    # GAT 折线
    ax.plot(dims, acc_gat,
            color=COLORS['GAT'], marker=MARKERS['GAT'],
            markersize=8, linewidth=2.2, markeredgecolor='white',
            markeredgewidth=1.2, label='GAT', zorder=5)

    # 数据标注
    for i, d in enumerate(dims):
        a_gat   = acc_gat[i]
        a_gatwf = acc_gatwf[i]
        gap = a_gatwf - a_gat

        if gap < 1.5:
            # 两点极近：GATWF 上，GAT 下
            ax.annotate(f'{a_gatwf:.1f}%', xy=(d, a_gatwf),
                        xytext=(0, 10), textcoords='offset points',
                        ha='center', va='bottom', fontsize=10,
                        color=COLORS['GATWF'], fontproperties=FONT_TNR_BOLD)
            ax.annotate(f'{a_gat:.1f}%', xy=(d, a_gat),
                        xytext=(0, -10), textcoords='offset points',
                        ha='center', va='top', fontsize=10,
                        color=COLORS['GAT'], fontproperties=FONT_TNR_BOLD)
        else:
            ax.annotate(f'{a_gatwf:.1f}%', xy=(d, a_gatwf),
                        xytext=(0, 7), textcoords='offset points',
                        ha='center', va='bottom', fontsize=10,
                        color=COLORS['GATWF'], fontproperties=FONT_TNR_BOLD)
            ax.annotate(f'{a_gat:.1f}%', xy=(d, a_gat),
                        xytext=(0, -7), textcoords='offset points',
                        ha='center', va='top', fontsize=10,
                        color=COLORS['GAT'], fontproperties=FONT_TNR_BOLD)

    # X 轴双行标签
    ax.set_xticks(dims)
    xlabels = [
        f'$\\mathrm{{d{{=}}{d}}}$\n($\\mathrm{{{node_counts[d]}}}$节点)'
        for d in dims
    ]
    ax.set_xticklabels(xlabels, fontsize=10)

    ax.set_xlabel('超立方体维度', fontsize=12, fontweight='bold', labelpad=8)
    ax.set_ylabel('诊断准确率 (%)', fontsize=12, fontweight='bold', labelpad=8)
    ax.set_ylim(88, 104)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(2))

    make_legend(ax, loc='lower left')
    plt.tight_layout()
    set_tnr_ticks(ax)

    save_path = FIG_DIR / 'compare_fig1_dimension_accuracy.png'
    fig.savefig(save_path, dpi=SAVE_DPI, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f'已保存: {save_path}')
    plt.close(fig)


# ============================================================
# 图2：故障率鲁棒性 — GAT vs GATWF 准确率对比
# ============================================================

def plot_fig2() -> None:
    """compare_fig2: 不同故障率下GAT与GATWF诊断准确率对比折线图。"""
    frs = np.array(FAULT_DATA['fault_rates'], dtype=float)
    acc_gat   = np.array(FAULT_DATA['GAT']['accuracy'])
    acc_gatwf = np.array(FAULT_DATA['GATWF']['accuracy'])

    fig, ax = plt.subplots(figsize=(8, 5.5))
    apply_style(ax)

    # 填充 GATWF 领先区域
    ax.fill_between(frs, acc_gat, acc_gatwf,
                    where=(acc_gatwf >= acc_gat).tolist(),
                    color=COLORS['GATWF'], alpha=0.08, interpolate=True)

    # GATWF 折线
    ax.plot(frs, acc_gatwf,
            color=COLORS['GATWF'], marker=MARKERS['GATWF'],
            markersize=8, linewidth=2.2, markeredgecolor='white',
            markeredgewidth=1.2, label='GATWF', zorder=5)

    # GAT 折线
    ax.plot(frs, acc_gat,
            color=COLORS['GAT'], marker=MARKERS['GAT'],
            markersize=8, linewidth=2.2, markeredgecolor='white',
            markeredgewidth=1.2, label='GAT', zorder=5)

    # 数据标注
    for i, fr in enumerate(frs):
        a_gat   = acc_gat[i]
        a_gatwf = acc_gatwf[i]
        gatwf_above = (a_gatwf >= a_gat)

        y_top    = 7 if gatwf_above else -10
        va_top   = 'bottom' if gatwf_above else 'top'
        y_bot    = -10 if gatwf_above else 7
        va_bot   = 'top' if gatwf_above else 'bottom'

        # GATWF 标注
        ax.annotate(f'{a_gatwf:.1f}%', xy=(fr, a_gatwf),
                    xytext=(0, y_top), textcoords='offset points',
                    ha='center', va=va_top, fontsize=9,
                    color=COLORS['GATWF'], fontproperties=FONT_TNR_BOLD)
        # GAT 标注
        ax.annotate(f'{a_gat:.1f}%', xy=(fr, a_gat),
                    xytext=(0, y_bot), textcoords='offset points',
                    ha='center', va=va_bot, fontsize=9,
                    color=COLORS['GAT'], fontproperties=FONT_TNR_BOLD)

    ax.set_xlim(5, 56)
    ax.set_xticks([10, 20, 25, 30, 40, 50])
    ax.set_xticklabels(['10%', '20%', '25%', '30%', '40%', '50%'],
                       fontsize=10, fontproperties=FONT_TNR)

    ax.set_ylim(82, 106)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(4))

    ax.set_xlabel('故障率', fontsize=12, fontweight='bold', labelpad=8)
    ax.set_ylabel('诊断准确率 (%)', fontsize=12, fontweight='bold', labelpad=8)

    make_legend(ax, loc='lower left')
    plt.tight_layout()
    set_tnr_ticks(ax)

    save_path = FIG_DIR / 'compare_fig2_faultrate_accuracy.png'
    fig.savefig(save_path, dpi=SAVE_DPI, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f'已保存: {save_path}')
    plt.close(fig)


# ============================================================
# 图3：样本效率 — GAT vs GATWF 准确率对比
# ============================================================

def plot_fig3() -> None:
    """compare_fig3: 不同样本量下GAT与GATWF诊断准确率对比折线图（对数X轴）。"""
    ns = np.array(SAMPLE_DATA['n_samples'])
    acc_gat   = np.array(SAMPLE_DATA['GAT']['accuracy'])
    acc_gatwf = np.array(SAMPLE_DATA['GATWF']['accuracy'])

    fig, ax = plt.subplots(figsize=(8, 5.5))
    apply_style(ax)
    ax.set_xscale('log')

    ax.grid(True, which='major', linestyle='--', linewidth=0.6,
            color=COLORS['grid'], alpha=0.8)
    ax.grid(False, which='minor')

    # 填充 GATWF 领先区域（逐段渐变）
    gaps = acc_gatwf - acc_gat
    alpha_max, alpha_min = 0.20, 0.04
    gap_max = max(abs(gaps.max()), 1e-6)
    for i in range(len(ns) - 1):
        gap_seg = (abs(gaps[i]) + abs(gaps[i + 1])) / 2
        alpha_seg = alpha_min + (alpha_max - alpha_min) * (gap_seg / gap_max)
        if (acc_gatwf[i] + acc_gatwf[i + 1]) >= (acc_gat[i] + acc_gat[i + 1]):
            ax.fill_between(
                [ns[i], ns[i + 1]],
                [acc_gat[i], acc_gat[i + 1]],
                [acc_gatwf[i], acc_gatwf[i + 1]],
                color=COLORS['GATWF'], alpha=float(alpha_seg), linewidth=0
            )

    # GATWF 折线
    ax.plot(ns, acc_gatwf,
            color=COLORS['GATWF'], marker=MARKERS['GATWF'],
            markersize=8, linewidth=2.2, markeredgecolor='white',
            markeredgewidth=1.2, label='GATWF', zorder=5)

    # GAT 折线
    ax.plot(ns, acc_gat,
            color=COLORS['GAT'], marker=MARKERS['GAT'],
            markersize=8, linewidth=2.2, markeredgecolor='white',
            markeredgewidth=1.2, label='GAT', zorder=5)

    # 数据标注
    for i, n in enumerate(ns):
        a_gat   = acc_gat[i]
        a_gatwf = acc_gatwf[i]
        gatwf_above = (a_gatwf >= a_gat)

        y_top = 8 if gatwf_above else -10
        va_top = 'bottom' if gatwf_above else 'top'
        y_bot = -8 if gatwf_above else 10
        va_bot = 'top' if gatwf_above else 'bottom'

        ax.annotate(f'{a_gatwf:.1f}%', xy=(n, a_gatwf),
                    xytext=(0, y_top), textcoords='offset points',
                    ha='center', va=va_top, fontsize=9,
                    color=COLORS['GATWF'], fontproperties=FONT_TNR_BOLD)
        ax.annotate(f'{a_gat:.1f}%', xy=(n, a_gat),
                    xytext=(0, y_bot), textcoords='offset points',
                    ha='center', va=va_bot, fontsize=9,
                    color=COLORS['GAT'], fontproperties=FONT_TNR_BOLD)

    ax.set_xticks([500, 1000, 2000, 5000, 10000])
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    ax.set_xlim(350, 14000)

    ax.set_ylim(90, 106)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(2))

    ax.set_xlabel('训练样本量', fontsize=12, fontweight='bold', labelpad=8)
    ax.set_ylabel('诊断准确率 (%)', fontsize=12, fontweight='bold', labelpad=8)

    make_legend(ax, loc='lower right')
    plt.tight_layout()

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(FONT_TNR)

    save_path = FIG_DIR / 'compare_fig3_sample_accuracy.png'
    fig.savefig(save_path, dpi=SAVE_DPI, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f'已保存: {save_path}')
    plt.close(fig)


# ============================================================
# 图4：消融实验对比 — GAT 与 GATWF 各变体并排柱状图
# ============================================================

def _draw_ablation_bars(ax: Axes, ablation: dict, model_key: str,
                        color: str) -> None:
    """在给定坐标轴上绘制消融实验柱状图。"""
    labels   = ablation['labels']
    accs     = np.array(ablation['accuracy'])
    deltas   = ablation['delta']
    baseline = accs[0]

    x = np.arange(len(accs))
    bar_width = 0.52

    # 柱子颜色：基线用主色，其余用淡化版本
    bar_colors = []
    for d in deltas:
        if d is None:
            bar_colors.append(color)
        elif d == 0.00:
            # delta=0 的非基线变体（如GATWF-1layer）用半透明主色
            bar_colors.append(color)
        else:
            # 消融变体：颜色按衰减程度加深灰调
            fade = min(abs(d) / 50.0, 0.6)
            c = mcolors.to_rgb(color)
            gray = np.array([0.5, 0.5, 0.5])
            mixed = (1 - fade) * np.array(c) + fade * gray
            bar_colors.append(tuple(mixed))

    apply_style(ax)
    bars = ax.bar(x, accs, width=bar_width, color=bar_colors,
                  edgecolor='white', linewidth=1.2, zorder=4, alpha=0.92)

    # 基线参考横线（分段，避免与基线柱重叠）
    bw = bar_width / 2 + 0.04
    x_left = -0.55
    x_right = len(accs) - 0.30
    segments = [(x_left, x[0] - bw), (x[0] + bw, x_right)]
    for x0, x1 in segments:
        if x0 < x1:
            ax.plot([x0, x1], [baseline, baseline],
                    color=color, linestyle='--', linewidth=1.4,
                    zorder=6, alpha=0.55)

    ax.text(x_right, baseline + 0.8,
            '基线 ' + f'$\\mathrm{{{baseline:.1f}}}$%',
            fontsize=8.5, color=color, va='bottom', ha='right', zorder=7)

    # 柱顶标注 Accuracy，柱内标注 ΔAccuracy
    for bar, delta in zip(bars, deltas):
        h  = bar.get_height()
        cx = bar.get_x() + bar.get_width() / 2

        ax.annotate(f'{h:.1f}%', xy=(cx, h), xytext=(0, 5),
                    textcoords='offset points', ha='center', va='bottom',
                    fontsize=10, color=bar.get_facecolor(),
                    fontproperties=FONT_TNR_BOLD, zorder=7)

        if delta is None:
            ax.text(cx, h * 0.48, '基线',
                    ha='center', va='center', fontsize=10,
                    color='white', fontweight='bold', zorder=5)
        elif delta == 0.00:
            ax.text(cx, h * 0.48, 'Δ0.0%',
                    ha='center', va='center', fontsize=10,
                    color='white', fontproperties=FONT_TNR_BOLD, zorder=5)
        else:
            ax.text(cx, h * 0.55, f'Δ{delta:.1f}%',
                    ha='center', va='center', fontsize=10,
                    color='white', fontproperties=FONT_TNR_BOLD, zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_xlim(-0.55, len(accs) - 0.30)
    ax.set_ylim(0, 115)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(10))
    ax.set_ylabel('诊断准确率 (%)', fontsize=11, fontweight='bold', labelpad=6)
    ax.set_title(model_key, fontsize=12, fontweight='bold',
                 color=color, pad=6, fontproperties=FONT_TNR_BOLD)


def plot_fig4() -> None:
    """compare_fig4: GAT与GATWF消融实验并排对比柱状图。"""
    fig, (ax_gat, ax_gatwf) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.set_facecolor('#FFFFFF')

    _draw_ablation_bars(ax_gat,   ABLATION_GAT,   'GAT',   COLORS['GAT'])
    _draw_ablation_bars(ax_gatwf, ABLATION_GATWF, 'GATWF', COLORS['GATWF'])

    plt.tight_layout(pad=2.0)

    for ax in (ax_gat, ax_gatwf):
        set_tnr_ticks(ax)

    save_path = FIG_DIR / 'compare_fig4_ablation.png'
    fig.savefig(save_path, dpi=SAVE_DPI, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f'已保存: {save_path}')
    plt.close(fig)


# ============================================================
# 图5：综合雷达图 — GAT vs GATWF 五维评测
# ============================================================

def plot_fig5() -> None:
    """compare_fig5: GAT与GATWF综合五维雷达图对比（基准：d=6，故障率25%，样本5000）。

    五个维度：
        1. 诊断准确率（基准点）
        2. F1分数（基准点）
        3. 高故障鲁棒性（故障率50%准确率）
        4. 维度扩展性（d=8准确率）
        5. 小样本效率（n=1000准确率）
    """
    categories = ['诊断准确率\n(d=6,f=25%)', 'F1分数\n(d=6,f=25%)',
                  '高故障鲁棒性\n(f=50%)', '维度扩展性\n(d=8)',
                  '小样本效率\n(n=1000)']
    N = len(categories)

    # GAT 数据
    gat_vals = np.array([
        95.40,                                        # 诊断准确率
        99.82,                                        # F1
        FAULT_DATA['GAT']['accuracy'][-1],            # 高故障鲁棒性 f=50%
        DIM_DATA['GAT']['accuracy'][-1],              # d=8
        SAMPLE_DATA['GAT']['accuracy'][1],            # n=1000
    ])

    # GATWF 数据
    gatwf_vals = np.array([
        99.80,
        99.99,
        FAULT_DATA['GATWF']['accuracy'][-1],
        DIM_DATA['GATWF']['accuracy'][-1],
        SAMPLE_DATA['GATWF']['accuracy'][1],
    ])

    # 角度
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    gat_plot   = gat_vals.tolist()   + gat_vals[:1].tolist()
    gatwf_plot = gatwf_vals.tolist() + gatwf_vals[:1].tolist()

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    fig.set_facecolor('#FFFFFF')
    ax.set_facecolor(COLORS['bg'])

    # 刻度范围
    r_min, r_max = 80, 100
    ax.set_ylim(r_min, r_max)
    ax.set_yticks([82, 86, 90, 94, 98])
    ax.set_yticklabels(['82', '86', '90', '94', '98'],
                       fontsize=8, color='#555555',
                       fontproperties=FONT_TNR)

    # 网格线样式
    ax.yaxis.grid(True, linestyle='--', linewidth=0.5,
                  color=COLORS['grid'], alpha=0.9)
    ax.xaxis.grid(True, linestyle='-', linewidth=0.5,
                  color=COLORS['grid'], alpha=0.6)
    ax.spines['polar'].set_color(COLORS['spine'])
    ax.spines['polar'].set_linewidth(0.8)

    # 绘制 GATWF 区域
    ax.fill(angles, gatwf_plot, color=COLORS['GATWF'], alpha=0.15)
    ax.plot(angles, gatwf_plot,
            color=COLORS['GATWF'], linewidth=2.2,
            marker=MARKERS['GATWF'], markersize=8,
            markeredgecolor='white', markeredgewidth=1.2,
            label='GATWF', zorder=5)

    # 绘制 GAT 区域
    ax.fill(angles, gat_plot, color=COLORS['GAT'], alpha=0.12)
    ax.plot(angles, gat_plot,
            color=COLORS['GAT'], linewidth=2.2,
            marker=MARKERS['GAT'], markersize=8,
            markeredgecolor='white', markeredgewidth=1.2,
            label='GAT', zorder=5)

    # 数值标注（外侧）
    for i, angle in enumerate(angles[:-1]):
        r_gat   = gat_vals[i]
        r_gatwf = gatwf_vals[i]
        ha = 'center'
        # 根据角度动态调整 ha
        if 0.1 < angle < np.pi - 0.1:
            ha = 'left'
        elif np.pi + 0.1 < angle < 2 * np.pi - 0.1:
            ha = 'right'

        # GATWF 标注（外侧）
        ax.annotate(f'{r_gatwf:.1f}', xy=(angle, r_gatwf),
                    xytext=(angle, r_gatwf + 1.2),
                    fontsize=8.5, ha=ha, va='center',
                    color=COLORS['GATWF'],
                    fontproperties=FONT_TNR_BOLD)

        # GAT 标注（内侧）
        ax.annotate(f'{r_gat:.1f}', xy=(angle, r_gat),
                    xytext=(angle, r_gat - 1.8),
                    fontsize=8.5, ha=ha, va='center',
                    color=COLORS['GAT'],
                    fontproperties=FONT_TNR_BOLD)

    # 类别标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10, color='#000000')
    for label in ax.get_xticklabels():
        label.set_fontname('SimSun')

    legend = ax.legend(loc='upper right', bbox_to_anchor=(1.28, 1.15),
                       fontsize=11, frameon=True, fancybox=False,
                       edgecolor=COLORS['grid'], facecolor='white',
                       framealpha=0.9)
    legend.get_frame().set_linewidth(0.8)
    set_legend_tnr(legend)

    plt.tight_layout()

    save_path = FIG_DIR / 'compare_fig5_radar.png'
    fig.savefig(save_path, dpi=SAVE_DPI, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f'已保存: {save_path}')
    plt.close(fig)


# ============================================================
# 图6：效率对比 — 训练耗时与推理时延（双子图，维度维度）
# ============================================================

def plot_fig6() -> None:
    """compare_fig6: GAT与GATWF训练耗时与推理时延双维度效率对比。

    左子图：不同维度下的训练耗时（柱状图，对数纵轴）。
    右子图：不同维度下的推理时延（折线图）。
    """
    dims = np.array(DIM_DATA['dims'])
    node_counts = DIM_DATA['node_counts']
    train_gat   = np.array(DIM_DATA['GAT']['train_time'])
    train_gatwf = np.array(DIM_DATA['GATWF']['train_time'])
    infer_gat   = np.array(DIM_DATA['GAT']['infer_ms'])
    infer_gatwf = np.array(DIM_DATA['GATWF']['infer_ms'])

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.set_facecolor('#FFFFFF')

    # ── 左子图：训练耗时（柱状图，对数） ────────────────────────
    apply_style(ax_left)
    ax_left.set_yscale('log')
    ax_left.grid(True, which='minor', linestyle=':', linewidth=0.4,
                 color=COLORS['grid'], alpha=0.5)

    x = np.arange(len(dims))
    bar_w = 0.35

    bars_gat   = ax_left.bar(x - bar_w / 2, train_gat,   bar_w,
                              color=COLORS['GAT'],   edgecolor='white',
                              linewidth=1.0, alpha=0.88, label='GAT',   zorder=4)
    bars_gatwf = ax_left.bar(x + bar_w / 2, train_gatwf, bar_w,
                              color=COLORS['GATWF'], edgecolor='white',
                              linewidth=1.0, alpha=0.88, label='GATWF', zorder=4)

    # 柱顶数值
    for bar, val in zip(bars_gat, train_gat):
        ax_left.annotate(f'{val:.0f}s',
                         xy=(bar.get_x() + bar.get_width() / 2, val),
                         xytext=(0, 4), textcoords='offset points',
                         ha='center', va='bottom', fontsize=8.5,
                         color=COLORS['GAT'], fontproperties=FONT_TNR_BOLD)

    for bar, val in zip(bars_gatwf, train_gatwf):
        ax_left.annotate(f'{val:.0f}s',
                         xy=(bar.get_x() + bar.get_width() / 2, val),
                         xytext=(0, 4), textcoords='offset points',
                         ha='center', va='bottom', fontsize=8.5,
                         color=COLORS['GATWF'], fontproperties=FONT_TNR_BOLD)

    # X 轴标签
    ax_left.set_xticks(x)
    xlabels = [
        f'$\\mathrm{{d{{=}}{d}}}$\n($\\mathrm{{{node_counts[d]}}}$节点)'
        for d in dims
    ]
    ax_left.set_xticklabels(xlabels, fontsize=9.5)

    def log_fmt(v, _):
        if v >= 1000:
            return f'{v / 1000:.0f}k'
        return f'{v:.0f}'

    ax_left.yaxis.set_major_formatter(mticker.FuncFormatter(log_fmt))
    ax_left.set_ylim(15, 4000)
    ax_left.set_xlabel('超立方体维度', fontsize=11, fontweight='bold', labelpad=8)
    ax_left.set_ylabel('训练耗时 (s)', fontsize=11, fontweight='bold', labelpad=8)
    ax_left.set_title('训练耗时对比', fontsize=12, fontweight='bold', pad=8)

    legend_l = ax_left.legend(loc='upper left', fontsize=10, frameon=True,
                               fancybox=False, edgecolor=COLORS['grid'],
                               facecolor='white', framealpha=0.9)
    legend_l.get_frame().set_linewidth(0.8)
    set_legend_tnr(legend_l)
    for label in ax_left.get_yticklabels():
        label.set_fontproperties(FONT_TNR)

    # ── 右子图：推理时延（折线图） ───────────────────────────────
    apply_style(ax_right)

    ax_right.plot(dims, infer_gatwf,
                  color=COLORS['GATWF'], marker=MARKERS['GATWF'],
                  markersize=8, linewidth=2.2, markeredgecolor='white',
                  markeredgewidth=1.2, label='GATWF', zorder=5)
    ax_right.plot(dims, infer_gat,
                  color=COLORS['GAT'], marker=MARKERS['GAT'],
                  markersize=8, linewidth=2.2, markeredgecolor='white',
                  markeredgewidth=1.2, label='GAT', zorder=5)

    for i, d in enumerate(dims):
        ax_right.annotate(f'{infer_gatwf[i]:.2f}',
                          xy=(d, infer_gatwf[i]),
                          xytext=(0, 7), textcoords='offset points',
                          ha='center', va='bottom', fontsize=9,
                          color=COLORS['GATWF'], fontproperties=FONT_TNR_BOLD)
        ax_right.annotate(f'{infer_gat[i]:.2f}',
                          xy=(d, infer_gat[i]),
                          xytext=(0, -9), textcoords='offset points',
                          ha='center', va='top', fontsize=9,
                          color=COLORS['GAT'], fontproperties=FONT_TNR_BOLD)

    ax_right.set_xticks(dims)
    ax_right.set_xticklabels(xlabels, fontsize=9.5)
    ax_right.set_ylim(0.8, 5.0)
    ax_right.yaxis.set_major_locator(mticker.MultipleLocator(0.5))
    ax_right.set_xlabel('超立方体维度', fontsize=11, fontweight='bold', labelpad=8)
    ax_right.set_ylabel('推理时延 (ms)', fontsize=11, fontweight='bold', labelpad=8)
    ax_right.set_title('推理时延对比', fontsize=12, fontweight='bold', pad=8)

    legend_r = ax_right.legend(loc='upper left', fontsize=10, frameon=True,
                                fancybox=False, edgecolor=COLORS['grid'],
                                facecolor='white', framealpha=0.9)
    legend_r.get_frame().set_linewidth(0.8)
    set_legend_tnr(legend_r)
    set_tnr_ticks(ax_right)

    plt.tight_layout(pad=2.0)

    save_path = FIG_DIR / 'compare_fig6_efficiency.png'
    fig.savefig(save_path, dpi=SAVE_DPI, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f'已保存: {save_path}')
    plt.close(fig)


# ============================================================
# 主入口
# ============================================================

FIGURE_MAP = {
    '1': ('维度扩展性准确率对比',         plot_fig1),
    '2': ('故障率鲁棒性准确率对比',       plot_fig2),
    '3': ('样本效率准确率对比',           plot_fig3),
    '4': ('消融实验对比（双模型并排）',   plot_fig4),
    '5': ('综合雷达图（五维评测）',       plot_fig5),
    '6': ('效率对比（训练耗时+推理时延）', plot_fig6),
}


def main() -> None:
    parser = argparse.ArgumentParser(description='GAT vs GATWF 多维度对比分析绘图')
    parser.add_argument('--fig', type=str, default='all',
                        help='要绘制的图编号（1-6）或 all')
    args = parser.parse_args()

    if args.fig == 'all':
        for _, (desc, func) in FIGURE_MAP.items():
            print(f'正在绘制 {desc}...')
            func()
    elif args.fig in FIGURE_MAP:
        desc, func = FIGURE_MAP[args.fig]
        print(f'正在绘制 {desc}...')
        func()
    else:
        print(f'未知图编号: {args.fig}，可选: {", ".join(FIGURE_MAP.keys())}, all')
        sys.exit(1)


if __name__ == '__main__':
    main()
