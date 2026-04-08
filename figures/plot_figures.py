"""
Input: TrainingRecords/plot_data/ 下各实验的 JSON 数据文件
Output: figures/ 下的高清 PNG 图片
Position: 第四章实验结果可视化绘图脚本

用法:
    python figures/plot_figures.py --fig 1       # 绘制图4-1：维度-准确率对比
    python figures/plot_figures.py --fig 2       # 绘制图4-2：维度-参数量对比
    python figures/plot_figures.py --fig 3       # 绘制图4-3：故障率-准确率对比
    python figures/plot_figures.py --fig 4       # 绘制图4-4：样本量-准确率对比
    python figures/plot_figures.py --fig 5       # 绘制图4-5：消融实验-准确率柱状图
    python figures/plot_figures.py --fig all     # 绘制全部图
"""

import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.axes import Axes
import numpy as np

# ============================================================
# 全局样式配置
# ============================================================

# 字体：中文宋体（全局 sans-serif），英文/数字在标注中单独指定 Times New Roman
plt.rcParams['font.sans-serif'] = ['SimSun', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'

# Times New Roman 字体属性，用于数据标注中的纯数字文本
from matplotlib.font_manager import FontProperties
FONT_TNR = FontProperties(family='Times New Roman')
FONT_TNR_BOLD = FontProperties(family='Times New Roman', weight='bold')

COLORS = {
    'GAT':   "#C15A38",   # GAT 主色
    'BPNN':  "#4E81B1",   # BPNN 主色
    'fill':  '#C15A38',   # 填充（半透明）
    'bg':    '#FAF9F5',   # 暖灰白背景
    'grid':  '#E8E6DC',   # 网格线
    'spine': '#666666',   # 坐标轴边框
    'text':  '#000000',   # 文字颜色
}

# 标记样式
MARKERS = {
    'GAT':  'o',  # 实心圆
    'BPNN': 's',  # 实心方块
}

# 图片保存参数
SAVE_DPI = 600
FIG_DIR = Path(__file__).parent
DATA_DIR = Path(__file__).parent.parent / 'TrainingRecords' / 'plot_data'


def apply_style(ax: Axes) -> None:
    """统一坐标轴样式。"""
    ax.set_facecolor(COLORS['bg'])
    ax.figure.set_facecolor('#FFFFFF')

    # 网格
    ax.grid(True, linestyle='--', linewidth=0.6,
            color=COLORS['grid'], alpha=0.8)
    ax.set_axisbelow(True)

    # 边框
    for spine in ax.spines.values():
        spine.set_color(COLORS['spine'])
        spine.set_linewidth(0.8)

    # 刻度
    ax.tick_params(colors=COLORS['text'], labelsize=10)


def load_experiment_data(exp_folder: str) -> list[dict]:
    """从 plot_data 子文件夹加载所有 JSON 文件。

    Returns:
        包含解析后数据的字典列表，每个字典包含 JSON 原始内容
        以及从文件名解析出的 _model, _dim, _fault_rate, _n_samples。
    """
    folder = DATA_DIR / exp_folder
    results = []
    for f in sorted(folder.glob('*.json')):
        with open(f, 'r', encoding='utf-8') as fp:
            data = json.load(fp)

        # 从文件名解析元信息: {MODEL}_{d}d_f{rate}_n{samples}.json
        m = re.match(r'(.+?)_(\d+)d_f([\d.]+)_n(\d+)\.json', f.name)
        if m:
            data['_model'] = m.group(1)
            data['_dim'] = int(m.group(2))
            data['_fault_rate'] = float(m.group(3))
            data['_n_samples'] = int(m.group(4))
        results.append(data)
    return results


# ============================================================
# 图4-1：不同维度下 BPNN 与 GAT 的诊断准确率对比
# ============================================================

def plot_fig1() -> None:
    """图4-1：不同维度下BPNN与GAT的诊断准确率对比折线图。"""
    records = load_experiment_data('exp1_dimension_scalability')

    # 按模型分组
    dims_gat, acc_gat = [], []
    dims_bpnn, acc_bpnn = [], []

    for r in records:
        model = r['_model']
        dim = r['_dim']
        accuracy = r['results']['accuracy'] * 100  # 转为百分比

        if model == 'GAT':
            dims_gat.append(dim)
            acc_gat.append(accuracy)
        elif model == 'BPNN':
            dims_bpnn.append(dim)
            acc_bpnn.append(accuracy)

    # 排序
    order_gat = np.argsort(dims_gat)
    dims_gat = np.array(dims_gat)[order_gat]
    acc_gat = np.array(acc_gat)[order_gat]

    order_bpnn = np.argsort(dims_bpnn)
    dims_bpnn = np.array(dims_bpnn)[order_bpnn]
    acc_bpnn = np.array(acc_bpnn)[order_bpnn]

    # 节点数映射
    node_counts = {4: 16, 5: 32, 6: 64, 7: 128, 8: 256}

    # ---- 绑图 ----
    fig, ax = plt.subplots(figsize=(8, 5.5))
    apply_style(ax)

    # 填充差距区域（GAT 领先的部分）
    ax.fill_between(dims_gat, acc_bpnn, acc_gat,
                    where=(acc_gat >= acc_bpnn).tolist(),
                    color=COLORS['fill'], alpha=0.08,
                    interpolate=True)

    # GAT 折线
    ax.plot(dims_gat, acc_gat,
            color=COLORS['GAT'], marker=MARKERS['GAT'],
            markersize=8, linewidth=2.2, markeredgecolor='white',
            markeredgewidth=1.2, label='GAT', zorder=5)

    # BPNN 折线
    ax.plot(dims_bpnn, acc_bpnn,
            color=COLORS['BPNN'], marker=MARKERS['BPNN'],
            markersize=8, linewidth=2.2, markeredgecolor='white',
            markeredgewidth=1.2, label='BPNN', zorder=5)

    # 数据标注 —— 对每个维度点分别处理
    for i, d in enumerate(dims_gat):
        a_gat = acc_gat[i]
        a_bpnn = acc_bpnn[i]
        gap = abs(a_gat - a_bpnn)

        if gap < 3:
            # d=4：两点极近，BPNN 在上方标注，GAT 在下方标注，加大偏移避免重叠
            ax.annotate(f'{a_bpnn:.1f}%', xy=(d, a_bpnn),
                        xytext=(0, 10), textcoords='offset points',
                        ha='center', va='bottom', fontsize=11,
                        color=COLORS['BPNN'],
                        fontproperties=FONT_TNR_BOLD)
            ax.annotate(f'{a_gat:.1f}%', xy=(d, a_gat),
                        xytext=(0, -10), textcoords='offset points',
                        ha='center', va='top', fontsize=11,
                        color=COLORS['GAT'],
                        fontproperties=FONT_TNR_BOLD)
        else:
            # 其余维度：GAT 在上方，BPNN 在下方
            ax.annotate(f'{a_gat:.1f}%', xy=(d, a_gat),
                        xytext=(0, 7), textcoords='offset points',
                        ha='center', va='bottom', fontsize=11,
                        color=COLORS['GAT'],
                        fontproperties=FONT_TNR_BOLD)
            ax.annotate(f'{a_bpnn:.1f}%', xy=(d, a_bpnn),
                        xytext=(0, -7), textcoords='offset points',
                        ha='center', va='top', fontsize=11,
                        color=COLORS['BPNN'],
                        fontproperties=FONT_TNR_BOLD)

    # X 轴双行标签：英文/数字用 mathtext (STIX ≈ TNR)，中文用默认宋体
    ax.set_xticks(dims_gat)
    xlabels = [
        f'$\\mathrm{{d{{=}}{d}}}$\n($\\mathrm{{{node_counts[d]}}}$节点)'
        for d in dims_gat
    ]
    ax.set_xticklabels(xlabels, fontsize=10)

    # 轴标签
    ax.set_xlabel('超立方体维度', fontsize=12, color='#000000',
                  fontweight='bold', labelpad=8)
    ax.set_ylabel('诊断准确率 (%)', fontsize=12, color='#000000',
                  fontweight='bold', labelpad=8)

    # Y 轴范围
    ax.set_ylim(40, 110)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(10))

    # 图例
    legend = ax.legend(loc='lower left', fontsize=11, frameon=True,
                       fancybox=False, edgecolor=COLORS['grid'],
                       facecolor='white', framealpha=0.9)
    legend.get_frame().set_linewidth(0.8)

    plt.tight_layout()

    # tight_layout 后设置字体，避免被重新计算覆盖
    # Y 轴刻度数字 → TNR
    for label in ax.get_yticklabels():
        label.set_fontproperties(FONT_TNR)
    # 图例文字 → TNR
    for text in legend.get_texts():
        text.set_fontproperties(FONT_TNR)

    # 保存
    save_path = FIG_DIR / 'fig4_1_dimension_accuracy.png'
    fig.savefig(save_path, dpi=SAVE_DPI, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f'已保存: {save_path}')
    plt.close(fig)


# ============================================================
# 图4-2：不同维度下 BPNN 与 GAT 的模型参数量对比
# ============================================================

def plot_fig2() -> None:
    """图4-2：不同维度下BPNN与GAT的模型参数量对比（对数纵坐标）。"""
    records = load_experiment_data('exp1_dimension_scalability')

    # 按模型分组
    dims_gat, params_gat = [], []
    dims_bpnn, params_bpnn = [], []

    for r in records:
        model = r['_model']
        dim = r['_dim']
        n_params = r['results']['n_params']

        if model == 'GAT':
            dims_gat.append(dim)
            params_gat.append(n_params)
        elif model == 'BPNN':
            dims_bpnn.append(dim)
            params_bpnn.append(n_params)

    # 排序
    order = np.argsort(dims_gat)
    dims_gat = np.array(dims_gat)[order]
    params_gat = np.array(params_gat)[order]

    order = np.argsort(dims_bpnn)
    dims_bpnn = np.array(dims_bpnn)[order]
    params_bpnn = np.array(params_bpnn)[order]

    # 节点数映射
    node_counts = {4: 16, 5: 32, 6: 64, 7: 128, 8: 256}

    # 参数量格式化：<1M 用 K，≥1M 用 M
    def fmt_params(n: int) -> str:
        if n >= 1_000_000:
            return f'{n / 1_000_000:.2f}M'
        elif n >= 1_000:
            return f'{n / 1_000:.0f}K'
        return str(n)

    # ---- 绑图 ----
    fig, ax = plt.subplots(figsize=(8, 5.5))
    apply_style(ax)
    ax.set_yscale('log')

    # 对数刻度下添加次网格线
    ax.grid(True, which='minor', linestyle=':', linewidth=0.4,
            color=COLORS['grid'], alpha=0.5)

    # 填充 BPNN 参数量超过 GAT 的区域（参数爆炸警示）
    ax.fill_between(dims_bpnn, params_gat, params_bpnn,
                    where=(params_bpnn >= params_gat).tolist(),
                    color=COLORS['BPNN'], alpha=0.08,
                    interpolate=True)

    # BPNN 折线
    ax.plot(dims_bpnn, params_bpnn,
            color=COLORS['BPNN'], marker=MARKERS['BPNN'],
            markersize=8, linewidth=2.2, markeredgecolor='white',
            markeredgewidth=1.2, label='BPNN', zorder=5)

    # GAT 折线
    ax.plot(dims_gat, params_gat,
            color=COLORS['GAT'], marker=MARKERS['GAT'],
            markersize=8, linewidth=2.2, markeredgecolor='white',
            markeredgewidth=1.2, label='GAT', zorder=5)

    # BPNN 数据标注 —— 每个维度点都标注
    for i, d in enumerate(dims_bpnn):
        p_bpnn = params_bpnn[i]
        p_gat = params_gat[i]

        if p_bpnn < p_gat:
            # d=4, d=5：BPNN 在 GAT 下方，标注放在点下方
            offset_y, va = -8, 'top'
        elif p_bpnn / p_gat < 2:
            # d=6：BPNN 刚超过 GAT，两线很近，加大偏移
            offset_y, va = 12, 'bottom'
        else:
            # d=7, d=8：BPNN 远高于 GAT
            offset_y, va = 8, 'bottom'

        ax.annotate(fmt_params(p_bpnn), xy=(d, p_bpnn),
                    xytext=(0, offset_y), textcoords='offset points',
                    ha='center', va=va, fontsize=10,
                    color=COLORS['BPNN'],
                    fontproperties=FONT_TNR_BOLD)

    # GAT 数据标注 —— 所有点几乎相同（~280K），只标注一次
    # 放在 d=7 位置，下方，避开 d=6 处 BPNN 标签
    ax.annotate('\u2248280K', xy=(7, params_gat[3]),
                xytext=(0, -10), textcoords='offset points',
                ha='center', va='top', fontsize=10,
                color=COLORS['GAT'],
                fontproperties=FONT_TNR_BOLD)

    # X 轴双行标签：英文/数字用 mathtext (STIX ≈ TNR)，中文用默认宋体
    ax.set_xticks(dims_gat)
    xlabels = [
        f'$\\mathrm{{d{{=}}{d}}}$\n($\\mathrm{{{node_counts[d]}}}$节点)'
        for d in dims_gat
    ]
    ax.set_xticklabels(xlabels, fontsize=10)

    # Y 轴：可读格式化（10K, 100K, 1M, 10M）
    def log_fmt(x: float, pos: int) -> str:
        if x >= 1_000_000:
            return f'{x / 1_000_000:.0f}M'
        elif x >= 1_000:
            return f'{x / 1_000:.0f}K'
        return str(int(x))

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(log_fmt))
    ax.set_ylim(8_000, 15_000_000)

    # 轴标签
    ax.set_xlabel('超立方体维度', fontsize=12, color='#000000',
                  fontweight='bold', labelpad=8)
    ax.set_ylabel('模型参数量', fontsize=12, color='#000000',
                  fontweight='bold', labelpad=8)

    # 图例
    legend = ax.legend(loc='upper left', fontsize=11, frameon=True,
                       fancybox=False, edgecolor=COLORS['grid'],
                       facecolor='white', framealpha=0.9)
    legend.get_frame().set_linewidth(0.8)

    plt.tight_layout()

    # tight_layout 后设置字体
    for label in ax.get_yticklabels():
        label.set_fontproperties(FONT_TNR)
    for text in legend.get_texts():
        text.set_fontproperties(FONT_TNR)

    # 保存
    save_path = FIG_DIR / 'fig4_2_dimension_params.png'
    fig.savefig(save_path, dpi=SAVE_DPI, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f'已保存: {save_path}')
    plt.close(fig)


# ============================================================
# 图4-3：不同故障率下 BPNN 与 GAT 的诊断准确率对比
# ============================================================

def plot_fig3() -> None:
    """图4-3：不同故障率下BPNN与GAT的诊断准确率对比折线图。

    X轴采用线性比例轴，25%数据点自然落在20%与30%刻度之间。
    两线交叉点（约15~16%）以竖虚线标出。
    """
    records = load_experiment_data('exp2_fault_rate_robustness')

    faults_gat, acc_gat = [], []
    faults_bpnn, acc_bpnn = [], []

    for r in records:
        model = r['_model']
        # _fault_rate 来自文件名（如 0.10），round 避免浮点误差
        fr_pct = round(r['_fault_rate'] * 100)
        accuracy = r['results']['accuracy'] * 100  # 转为百分比

        if model == 'GAT':
            faults_gat.append(fr_pct)
            acc_gat.append(accuracy)
        elif model == 'BPNN':
            faults_bpnn.append(fr_pct)
            acc_bpnn.append(accuracy)

    # 排序（按故障率升序）
    order = np.argsort(faults_gat)
    faults_gat = np.array(faults_gat, dtype=float)[order]
    acc_gat = np.array(acc_gat)[order]

    order = np.argsort(faults_bpnn)
    faults_bpnn = np.array(faults_bpnn, dtype=float)[order]
    acc_bpnn = np.array(acc_bpnn)[order]

    # ---- 计算交叉点（10%~20% 之间线性插值）----
    # idx=0 时 GAT < BPNN，idx=1 时 GAT > BPNN
    diff = acc_gat - acc_bpnn          # diff[0]<0, diff[1]>0
    t = -diff[0] / (diff[1] - diff[0])  # 插值参数 t ∈ (0,1)
    x_cross = faults_gat[0] + t * (faults_gat[1] - faults_gat[0])
    y_cross = acc_gat[0] + t * (acc_gat[1] - acc_gat[0])

    # ---- 绘图 ----
    fig, ax = plt.subplots(figsize=(8, 5.5))
    apply_style(ax)

    # 填充 GAT 领先区域（interpolate=True 自动从交叉点开始填充）
    ax.fill_between(faults_gat, acc_bpnn, acc_gat,
                    where=(acc_gat >= acc_bpnn).tolist(),
                    color=COLORS['fill'], alpha=0.08,
                    interpolate=True)

    # GAT 折线
    ax.plot(faults_gat, acc_gat,
            color=COLORS['GAT'], marker=MARKERS['GAT'],
            markersize=8, linewidth=2.2, markeredgecolor='white',
            markeredgewidth=1.2, label='GAT', zorder=5)

    # BPNN 折线
    ax.plot(faults_bpnn, acc_bpnn,
            color=COLORS['BPNN'], marker=MARKERS['BPNN'],
            markersize=8, linewidth=2.2, markeredgecolor='white',
            markeredgewidth=1.2, label='BPNN', zorder=5)

    # ---- 交叉点竖虚线 + 三角标记 + 文字标注 ----
    ax.axvline(x=x_cross, color='#888888', linestyle='--',
               linewidth=1.2, alpha=0.75, zorder=3)
    # 黑色三角标记（向上三角，区别于圆形/方形）
    ax.plot(x_cross, y_cross, marker='^', markersize=9,
            color='#000000', markeredgecolor='white',
            markeredgewidth=1.0, zorder=7)
    ax.text(x_cross + 0.6, 33,
            '交叉点\n$\\approx$' + f'$\\mathrm{{{x_cross:.1f}}}$%',
            fontsize=9, color='#000000', va='bottom', ha='left',
            linespacing=1.5)

    # ---- 数据标注（每个点标注数值）----
    for i, fr in enumerate(faults_gat):
        a_gat = acc_gat[i]
        a_bpnn = acc_bpnn[i]
        gat_above = (a_gat >= a_bpnn)

        gat_x_off = 0
        gat_y_off = 8 if gat_above else -10
        gat_ha = 'center'
        gat_va = 'bottom' if gat_above else 'top'

        bpnn_y_off = -10 if gat_above else 8
        bpnn_va = 'top' if gat_above else 'bottom'

        ax.annotate(f'{a_gat:.1f}%', xy=(fr, a_gat),
                    xytext=(gat_x_off, gat_y_off),
                    textcoords='offset points',
                    ha=gat_ha, va=gat_va, fontsize=10,
                    color=COLORS['GAT'],
                    fontproperties=FONT_TNR_BOLD)

        ax.annotate(f'{a_bpnn:.1f}%', xy=(fr, a_bpnn),
                    xytext=(0, bpnn_y_off),
                    textcoords='offset points',
                    ha='center', va=bpnn_va, fontsize=10,
                    color=COLORS['BPNN'],
                    fontproperties=FONT_TNR_BOLD)

    # ---- X轴：线性比例轴，25% 自然落在 20% 与 30% 之间 ----
    ax.set_xlim(5, 56)
    ax.set_xticks([10, 20, 25, 30, 40, 50])
    ax.set_xticklabels(['10%', '20%', '25%', '30%', '40%', '50%'],
                        fontsize=10, fontproperties=FONT_TNR)

    # ---- Y轴 ----
    ax.set_ylim(30, 112)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(10))

    # ---- 轴标签 ----
    ax.set_xlabel('故障率', fontsize=12, color='#000000',
                  fontweight='bold', labelpad=8)
    ax.set_ylabel('诊断准确率 (%)', fontsize=12, color='#000000',
                  fontweight='bold', labelpad=8)

    # ---- 图例 ----
    legend = ax.legend(loc='upper right', fontsize=11, frameon=True,
                       fancybox=False, edgecolor=COLORS['grid'],
                       facecolor='white', framealpha=0.9)
    legend.get_frame().set_linewidth(0.8)

    plt.tight_layout()

    # tight_layout 后设置字体
    for label in ax.get_yticklabels():
        label.set_fontproperties(FONT_TNR)
    for text in legend.get_texts():
        text.set_fontproperties(FONT_TNR)

    save_path = FIG_DIR / 'fig4_3_faultrate_accuracy.png'
    fig.savefig(save_path, dpi=SAVE_DPI, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f'已保存: {save_path}')
    plt.close(fig)


# ============================================================
# 图4-4：不同样本量下 BPNN 与 GAT 的诊断准确率对比
# ============================================================

def plot_fig4() -> None:
    """图4-4：不同样本量下BPNN与GAT的诊断准确率对比折线图。

    X轴为对数坐标（样本量：500, 1000, 2000, 5000, 10000）。
    填充区域透明度随 X 轴从左到右递减，强调差距收窄趋势。
    """
    records = load_experiment_data('exp3_sample_efficiency')

    # 按模型分组
    ns_gat, acc_gat = [], []
    ns_bpnn, acc_bpnn = [], []

    for r in records:
        model = r['_model']
        n = r['_n_samples']
        accuracy = r['results']['accuracy'] * 100  # 转百分比

        if model == 'GAT':
            ns_gat.append(n)
            acc_gat.append(accuracy)
        elif model == 'BPNN':
            ns_bpnn.append(n)
            acc_bpnn.append(accuracy)

    # 按样本量升序排列
    order = np.argsort(ns_gat)
    ns_gat = np.array(ns_gat)[order]
    acc_gat = np.array(acc_gat)[order]

    order = np.argsort(ns_bpnn)
    ns_bpnn = np.array(ns_bpnn)[order]
    acc_bpnn = np.array(acc_bpnn)[order]

    # ---- 渐变填充：透明度从左（大差距）到右（小差距）线性递减 ----
    # 在相邻数据点间逐段填充，alpha 与差距成正比
    gaps = acc_gat - acc_bpnn                        # 每个点的差距值
    alpha_max, alpha_min = 0.22, 0.04
    gap_max = gaps.max()

    # ---- 绘图 ----
    fig, ax = plt.subplots(figsize=(8, 5.5))
    apply_style(ax)
    ax.set_xscale('log')

    # 对数 X 轴：关闭次网格，只保留主网格（否则次刻度线过密）
    ax.grid(True, which='major', linestyle='--', linewidth=0.6,
            color=COLORS['grid'], alpha=0.8)
    ax.grid(False, which='minor')

    # 逐段渐变填充（每两个相邻点之间填充一个矩形区域）
    for i in range(len(ns_gat) - 1):
        gap_seg = (gaps[i] + gaps[i + 1]) / 2          # 该段的平均差距
        alpha_seg = alpha_min + (alpha_max - alpha_min) * (gap_seg / gap_max)
        ax.fill_between(
            [ns_gat[i], ns_gat[i + 1]],
            [acc_bpnn[i], acc_bpnn[i + 1]],
            [acc_gat[i], acc_gat[i + 1]],
            color=COLORS['fill'], alpha=float(alpha_seg),
            linewidth=0
        )

    # GAT 折线
    ax.plot(ns_gat, acc_gat,
            color=COLORS['GAT'], marker=MARKERS['GAT'],
            markersize=8, linewidth=2.2, markeredgecolor='white',
            markeredgewidth=1.2, label='GAT', zorder=5)

    # BPNN 折线
    ax.plot(ns_bpnn, acc_bpnn,
            color=COLORS['BPNN'], marker=MARKERS['BPNN'],
            markersize=8, linewidth=2.2, markeredgecolor='white',
            markeredgewidth=1.2, label='BPNN', zorder=5)

    # ---- 数据标注 ----
    # 标注规则：GAT 始终在上方，BPNN 始终在下方；差距极小时加大偏移
    for i, n in enumerate(ns_gat):
        a_gat = acc_gat[i]
        a_bpnn = acc_bpnn[i]

        # GAT 标注（上方）
        ax.annotate(f'{a_gat:.1f}%', xy=(n, a_gat),
                    xytext=(0, 8), textcoords='offset points',
                    ha='center', va='bottom', fontsize=10,
                    color=COLORS['GAT'],
                    fontproperties=FONT_TNR_BOLD)

        # BPNN 标注（下方）
        # n=1000 时，右侧水平线段紧贴标注，加大向下偏移避让
        if n == 1000:
            bpnn_x_off, bpnn_y_off, bpnn_ha = 0, -18, 'center'
        else:
            bpnn_x_off, bpnn_y_off, bpnn_ha = 0, -8, 'center'
        ax.annotate(f'{a_bpnn:.1f}%', xy=(n, a_bpnn),
                    xytext=(bpnn_x_off, bpnn_y_off),
                    textcoords='offset points',
                    ha=bpnn_ha, va='top', fontsize=10,
                    color=COLORS['BPNN'],
                    fontproperties=FONT_TNR_BOLD)

    # ---- X 轴：对数刻度，显示实际样本量数值 ----
    x_ticks = [500, 1000, 2000, 5000, 10000]
    ax.set_xticks(x_ticks)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f'{int(x):,}'
    ))
    ax.set_xlim(350, 14000)

    # ---- Y 轴 ----
    ax.set_ylim(35, 113)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(10))

    # ---- 轴标签 ----
    ax.set_xlabel('训练样本量', fontsize=12, color='#000000',
                  fontweight='bold', labelpad=8)
    ax.set_ylabel('诊断准确率 (%)', fontsize=12, color='#000000',
                  fontweight='bold', labelpad=8)

    # ---- 图例 ----
    legend = ax.legend(loc='lower right', fontsize=11, frameon=True,
                       fancybox=False, edgecolor=COLORS['grid'],
                       facecolor='white', framealpha=0.9)
    legend.get_frame().set_linewidth(0.8)

    plt.tight_layout()

    # tight_layout 后设置字体
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(FONT_TNR)
    for text in legend.get_texts():
        text.set_fontproperties(FONT_TNR)

    save_path = FIG_DIR / 'fig4_4_sample_accuracy.png'
    fig.savefig(save_path, dpi=SAVE_DPI, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f'已保存: {save_path}')
    plt.close(fig)


# ============================================================
# 图4-5：消融实验结果柱状图
# ============================================================

def plot_fig5() -> None:
    """图4-5：消融实验结果柱状图（按Accuracy降序排列）。

    四个变体：GAT完整（基线）、GAT-1layer、GAT-1head、GAT-unidir。
    柱子按 Accuracy 降序排列；颜色随性能下降程度从蓝渐变为红；
    柱内标注 ΔAccuracy（白色），柱顶标注 Accuracy 数值；
    基线虚线贯穿全图便于视觉比较。
    """
    records = load_experiment_data('exp4_ablation')

    # ── 提取数据 ─────────────────────────────────────────────────
    variants: list[dict] = []
    for r in records:
        model = r['_model']
        acc   = r['results']['accuracy'] * 100
        variants.append({'model': model, 'acc': acc})

    # 按 Accuracy 降序排列
    variants.sort(key=lambda v: v['acc'], reverse=True)

    # 基线（完整 GAT）的 Accuracy
    baseline_acc = next(v['acc'] for v in variants if v['model'] == 'GAT')

    # 计算 ΔAccuracy（消融变体相对基线的差值，为负数）
    for v in variants:
        v['delta'] = None if v['model'] == 'GAT' else v['acc'] - baseline_acc

    # ── 标签与颜色映射 ────────────────────────────────────────────
    # 标签：两行，第一行英文用 mathtext (STIX ≈ TNR)，第二行中文用默认宋体
    label_map = {
        'GAT':        '$\\mathrm{GAT}$\n（完整基线）',
        'GAT-1layer': '$\\mathrm{GAT\\text{-}1layer}$\n（$\\mathrm{2}$层→$\\mathrm{1}$层）',
        'GAT-1head':  '$\\mathrm{GAT\\text{-}1head}$\n（$\\mathrm{8}$头→$\\mathrm{1}$头）',
        'GAT-unidir': '$\\mathrm{GAT\\text{-}unidir}$\n（双向→单向）',
    }
    # 颜色：基线深靛蓝；随下降程度由浅蓝渐变至深红
    # 使用 COLORS 中已有的 GAT 蓝和 BPNN 红，中间两档取自 RdBu 发散色板
    color_map = {
        'GAT':        COLORS['GAT'],      # 基线
        'GAT-1layer': "#B59161",          # ΔAcc = -3.0%
        'GAT-1head':  '#798C5E',          # ΔAcc = -12.8%
        'GAT-unidir': COLORS['BPNN'],     # ΔAcc = -46.8%
    }

    labels = [label_map[v['model']] for v in variants]
    accs   = [v['acc']              for v in variants]
    colors = [color_map[v['model']] for v in variants]

    x         = np.arange(len(variants))
    bar_width = 0.52

    # ── 绘图 ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5.5))
    apply_style(ax)

    bars = ax.bar(x, accs, width=bar_width, color=colors,
                  edgecolor='white', linewidth=1.2, zorder=4, alpha=0.92)

    # ── 基线参考横线（分段绘制）──────────────────────────────────
    # 柱子半宽（含小间距），用于计算各段边界
    bw = bar_width / 2 + 0.03

    # 各变体在 x 轴上的位置（降序排列后：0=GAT, 1=GAT-1layer, ...）
    x_start    = -0.55           # 与 xlim 左端保持一致
    x_gat_l    = 0 - bw          # GAT基线柱左侧边界
    x_gat_r    = 0 + bw          # GAT基线柱右侧边界
    x_1layer_l = 1 - bw          # GAT-1layer柱左侧边界
    x_1layer_r = 1 + bw          # GAT-1layer柱右侧边界
    x_end      = len(variants) - 0.30  # 图右端

    # 段0：图左端 → GAT基线柱左侧（正常透明度）
    ax.plot([x_start, x_gat_l], [baseline_acc, baseline_acc],
            color='#C6613F', linestyle='--', linewidth=1.4,
            zorder=6, alpha=0.60)
    # 段1：GAT基线柱右侧 → GAT-1layer柱左侧（正常透明度）
    ax.plot([x_gat_r, x_1layer_l], [baseline_acc, baseline_acc],
            color='#C6613F', linestyle='--', linewidth=1.4,
            zorder=6, alpha=0.60)
    # 段2：GAT-1layer柱范围内（大幅降低透明度，避免与92.4%标注重叠）
    ax.plot([x_1layer_l, x_1layer_r], [baseline_acc, baseline_acc],
            color='#C6613F', linestyle='--', linewidth=1.4,
            zorder=6, alpha=0.15)
    # 段3：GAT-1layer柱右侧 → 图右端（正常透明度）
    ax.plot([x_1layer_r, x_end], [baseline_acc, baseline_acc],
            color='#C6613F', linestyle='--', linewidth=1.4,
            zorder=6, alpha=0.60)

    # 参考线文字标注（贴右端，zorder=7 确保在线上方）
    ax.text(x_end, baseline_acc + 1.2,
            '基线 ' + f'$\\mathrm{{{baseline_acc:.1f}}}$%',
            fontsize=9, color="#A23E1D", va='bottom', ha='right',
            zorder=7)

    # ── 逐柱标注 ──────────────────────────────────────────────────
    for bar, v in zip(bars, variants):
        h   = bar.get_height()
        cx  = bar.get_x() + bar.get_width() / 2   # 柱中心 x 坐标
        col = color_map[v['model']]

        # Accuracy 数值：柱顶上方，zorder=7 确保在虚线之上
        ax.annotate(f'{h:.1f}%',
                    xy=(cx, h), xytext=(0, 6),
                    textcoords='offset points',
                    ha='center', va='bottom',
                    fontsize=11, color=col,
                    fontproperties=FONT_TNR_BOLD,
                    zorder=7)

        if v['delta'] is None:
            # 基线柱：柱内中部写"基线"（中文，宋体白色）
            # zorder=5 确保渲染在柱子（zorder=4）之上
            ax.text(cx, h * 0.50, '基线',
                    ha='center', va='center',
                    fontsize=10.5, color='white', fontweight='bold',
                    zorder=5)
        else:
            # 消融变体：柱内 55% 高度处标注 ΔAccuracy（TNR 白色）
            delta_str = f'Δ{v["delta"]:.1f}%'   # 值为负数，自然带负号
            ax.text(cx, h * 0.55, delta_str,
                    ha='center', va='center',
                    fontsize=10.5, color='white',
                    fontproperties=FONT_TNR_BOLD,
                    zorder=5)

    # ── X 轴 ──────────────────────────────────────────────────────
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    # 横轴两侧留白，右侧额外留空给基线文字
    ax.set_xlim(-0.55, len(variants) - 0.30)

    # ── Y 轴 ──────────────────────────────────────────────────────
    ax.set_ylim(0, 115)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(10))
    ax.set_ylabel('诊断准确率 (%)', fontsize=12, color='#000000',
                  fontweight='bold', labelpad=8)

    plt.tight_layout()

    # tight_layout 后设置字体
    for label in ax.get_yticklabels():
        label.set_fontproperties(FONT_TNR)

    save_path = FIG_DIR / 'fig4_5_ablation_accuracy.png'
    fig.savefig(save_path, dpi=SAVE_DPI, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f'已保存: {save_path}')
    plt.close(fig)


# ============================================================
# 图4-6：GAT 注意力权重分布箱线图（按边类型）
# ============================================================

# 边类型配色（独立于 COLORS 字典）
_EDGE_COLORS = {
    "N→N": "#5B9A5B",  # 深绿
    "N→F": "#C17832",  # 深橙
    "F→N": "#4E81B1",  # 蓝色
    "F→F": "#C15A38",  # 红色
}


def plot_fig6(attn_data: dict,
              dimension: int | None = None,
              fault_rate: float | None = None) -> None:
    """图4-6：GAT注意力权重按边类型分布箱线图。

    Args:
        attn_data: GAT.get_attention_weights() 的返回值，包含：
            - "by_type": {"N→N": array, "N→F": array, "F→N": array, "F→F": array}
            - "n_samples": 样本数
            - "n_heads": 注意力头数
        dimension: 超立方体维度（用于标题）
        fault_rate: 故障率（用于标题）
    """
    by_type  = attn_data["by_type"]
    n_samples = attn_data["n_samples"]
    n_heads   = attn_data["n_heads"]

    # 只保留有数据的边类型，固定顺序
    order = ["N→N", "N→F", "F→N", "F→F"]
    labels = [k for k in order if len(by_type[k]) > 0]
    data   = [by_type[k] for k in labels]
    counts = [len(by_type[k]) for k in labels]
    means  = [float(np.mean(by_type[k])) for k in labels]
    colors = [_EDGE_COLORS[k] for k in labels]

    x_pos = np.arange(1, len(labels) + 1)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    apply_style(ax)

    # ── 箱线图 ────────────────────────────────────────────────
    bp = ax.boxplot(
        data, positions=x_pos, widths=0.46,
        patch_artist=True, showfliers=False,
        medianprops=dict(color='white', linewidth=2.0),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        boxprops=dict(linewidth=1.2),
    )

    for patch, col in zip(bp['boxes'], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.88)

    # 须线 / 帽线颜色跟随箱体
    for i, col in enumerate(colors):
        bp['whiskers'][2 * i].set_color(col)
        bp['whiskers'][2 * i + 1].set_color(col)
        bp['caps'][2 * i].set_color(col)
        bp['caps'][2 * i + 1].set_color(col)

    # ── 均值散点（菱形） ─────────────────────────────────────
    ax.scatter(x_pos, means,
               marker='D', s=36, zorder=6,
               color=colors, edgecolors='white', linewidths=0.8)

    # ── 标注：样本数 + 均值 ───────────────────────────────────
    y_span = ax.get_ylim()[1] - ax.get_ylim()[0]
    for i, (lbl, cnt, mu) in enumerate(zip(labels, counts, means)):
        q75 = float(np.percentile(data[i], 75))
        # 放在 Q75 上方固定偏移
        y_ann = q75 + y_span * 0.04
        ax.text(x_pos[i], y_ann,
                f'n={cnt:,}\n$\\mu$={mu:.4f}',
                ha='center', va='bottom', fontsize=9,
                color='#333333', linespacing=1.4,
                fontproperties=FONT_TNR)

    # ── X 轴标签 ──────────────────────────────────────────────
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_xlim(0.4, len(labels) + 0.6)

    # ── Y 轴 ──────────────────────────────────────────────────
    ax.set_ylim(0.04, 0.42)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.05))
    ax.set_ylabel('注意力权重（多头平均）', fontsize=12, color='#000000',
                  fontweight='bold', labelpad=8)
    ax.set_xlabel('边类型（源节点 → 目标节点）', fontsize=12, color='#000000',
                  fontweight='bold', labelpad=8)

    # ── 标题 ──────────────────────────────────────────────────
    sub_parts = []
    if dimension is not None:
        sub_parts.append(f'$\\mathrm{{d{{=}}{dimension}}}$')
    if fault_rate is not None:
        sub_parts.append(f'故障率$\\mathrm{{{{=}}{fault_rate}}}$')
    sub_parts.append(f'$\\mathrm{{{n_samples}}}$样本')
    sub_parts.append(f'$\\mathrm{{{n_heads}}}$头平均')
    title = 'GAT 注意力权重分布（' + '，'.join(sub_parts) + '）'
    ax.set_title(title, fontsize=13, color='#000000', pad=12)

    plt.tight_layout()

    # tight_layout 后设置字体
    for lbl in ax.get_yticklabels():
        lbl.set_fontproperties(FONT_TNR)
    for lbl in ax.get_xticklabels():
        lbl.set_fontproperties(FONT_TNR)

    # ── 保存 ──────────────────────────────────────────────────
    save_path = FIG_DIR / 'fig4_6_attention_boxplot.png'
    fig.savefig(save_path, dpi=SAVE_DPI, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f'已保存: {save_path}')
    plt.close(fig)


# ============================================================
# 主入口
# ============================================================

FIGURE_MAP = {
    '1': ('图4-1：维度-准确率对比', plot_fig1),
    '2': ('图4-2：维度-参数量对比', plot_fig2),
    '3': ('图4-3：故障率-准确率对比', plot_fig3),
    '4': ('图4-4：样本量-准确率对比', plot_fig4),
    '5': ('图4-5：消融实验-准确率柱状图', plot_fig5),
}


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description='绘制第四章实验图表')
    parser.add_argument('--fig', type=str, default='1',
                        help='要绘制的图编号（1-5）或 all')
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
        print(f'未知图编号: {args.fig}')
        print(f'可选: {", ".join(FIGURE_MAP.keys())}, all')
        sys.exit(1)


if __name__ == '__main__':
    main()
