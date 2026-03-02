"""
Input:
  - 标准库: os, json
  - 第三方: numpy, matplotlib, networkx
Output:
  - visualize_syndrome: 可视化单个 syndrome 文件
  - visualize_hypercube_topology: 生成超立方体纯拓扑结构图（论文第二章用）
Position: 提供 syndrome 的可视化功能，生成网络拓扑图并标注故障节点和测试结果

⚠️ 提醒：修改本文件后务必：
   1. 更新上方 Input/Output/Position 说明
   2. 更新所属文件夹的 README.md（如果影响模块结构）
   3. 更新根目录 README.md（如果影响核心架构）
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D

# ============================================================
# 全局字体配置（与 figures/plot_figures.py 保持一致）
# ============================================================
plt.rcParams['font.sans-serif'] = ['SimSun', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'

FONT_TNR = FontProperties(family='Times New Roman')
FONT_TNR_BOLD = FontProperties(family='Times New Roman', weight='bold')

SAVE_DPI = 600


def _hypercube_layout(dimension: int) -> dict[int, tuple[float, float]]:
    """
    生成超立方体节点的固定2D坐标（伪3D投影）

    Args:
        dimension: 超立方体维度

    Returns:
        节点ID到(x, y)坐标的映射字典

    Notes:
        - bit 0 控制水平方向（x轴）
        - bit 1 控制垂直方向（y轴）
        - bit 2+ 沿对角线方向偏移，逐层缩小，模拟高维投影
    """
    pos = {}
    for node in range(2 ** dimension):
        x, y = 0.0, 0.0
        for bit in range(dimension):
            val = ((node >> bit) & 1) * 2 - 1  # -1 or +1
            if bit == 0:
                x += val
            elif bit == 1:
                y += val
            else:
                # 高维度：沿对角线方向偏移，逐层缩小
                scale = 0.6 * (0.5 ** (bit - 2))
                x += val * scale
                y += val * scale
        pos[node] = (x, y)
    return pos


def visualize_syndrome(syndrome_path: str, dimension: int | None = None) -> str:
    """
    可视化单个 syndrome 文件

    Args:
        syndrome_path: .npz 文件路径
        dimension: 超立方体维度（如果为 None，则从 metadata.json 读取）

    Returns:
        生成的图片文件路径

    Notes:
        图片会保存到与 syndrome 文件相同的目录，不会在屏幕上显示
    """
    # 加载数据
    data = np.load(syndrome_path)
    syndrome = data["syndrome"]
    label = data["label"]

    n_nodes = len(label)

    # 自动获取 dimension
    if dimension is None:
        # 尝试从 metadata.json 读取
        dataset_dir = os.path.dirname(os.path.dirname(syndrome_path))
        metadata_path = os.path.join(dataset_dir, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                dimension = metadata.get("dimension")

        # 如果还是 None，从节点数推断
        if dimension is None:
            dimension = int(np.log2(n_nodes))

    # 构建超立方体图
    G = nx.hypercube_graph(dimension)
    G = nx.relabel_nodes(G, {n: int("".join(map(str, n)), 2) for n in G.nodes()})

    # 故障节点
    faulty_nodes = set(np.where(label == 1)[0])
    node_colors = ["#ff6b6b" if i in faulty_nodes else "#69db7c" for i in range(n_nodes)]

    # 构建 syndrome 字典：(u, v) -> test_result
    syndrome_dict = {}
    idx = 0
    for u in range(n_nodes):
        for bit in range(dimension):
            v = u ^ (1 << bit)
            syndrome_dict[(u, v)] = syndrome[idx]
            idx += 1

    # 对于无向边，取可靠的测试结果
    edge_colors = []
    edge_styles = []
    edges = []

    for u in range(n_nodes):
        for bit in range(dimension):
            v = u ^ (1 << bit)
            if u > v:
                continue
            edges.append((u, v))

            if u not in faulty_nodes:
                test_result = syndrome_dict[(u, v)]
            elif v not in faulty_nodes:
                test_result = syndrome_dict[(v, u)]
            else:
                test_result = -1

            if test_result == -1:
                edge_colors.append("#adb5bd")
                edge_styles.append("dotted")
            elif test_result == 1:
                edge_colors.append("#ff6b6b")
                edge_styles.append("dashed")
            else:
                edge_colors.append("#69db7c")
                edge_styles.append("solid")

    # 绘图
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FAF9F5')
    ax.axis('off')

    # 使用固定的超立方体布局
    pos = _hypercube_layout(dimension)

    for (u, v), color, style in zip(edges, edge_colors, edge_styles):
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], edge_color=color,
                              style=style, width=2.2, ax=ax)

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800,
                           edgecolors='white', linewidths=1.5, ax=ax)

    binary_labels = {i: format(i, f'0{dimension}b') for i in range(n_nodes)}
    nx.draw_networkx_labels(G, pos, labels=binary_labels, font_size=9,
                            font_weight="bold", font_family='Times New Roman',
                            ax=ax)

    # 图例（中文标签，与论文风格统一）
    faulty_sorted = sorted(format(i, f'0{dimension}b') for i in faulty_nodes)
    legend_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff6b6b',
               markersize=12, label='故障节点'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#69db7c',
               markersize=12, label='正常节点'),
        Line2D([0], [0], color='#ff6b6b', linestyle='dashed', linewidth=2,
               label='测试结果：$\\mathrm{1}$（失败）'),
        Line2D([0], [0], color='#69db7c', linestyle='solid', linewidth=2,
               label='测试结果：$\\mathrm{0}$（通过）'),
        Line2D([0], [0], color='#adb5bd', linestyle='dotted', linewidth=2,
               label='不可靠（两端均故障）'),
    ]
    legend = ax.legend(handles=legend_handles, loc='lower right', fontsize=10,
                       frameon=True, fancybox=False, edgecolor='#E8E6DC',
                       facecolor='white', framealpha=0.9)
    legend.get_frame().set_linewidth(0.8)

    plt.tight_layout()

    # 保存图片并关闭，不显示
    output_path = syndrome_path.replace(".npz", ".png")
    plt.savefig(output_path, dpi=SAVE_DPI, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()

    return output_path


def visualize_hypercube_topology(dimension: int, output_path: str) -> str:
    """
    生成超立方体纯拓扑结构图（无故障信息，用于论文第二章理论介绍）

    Args:
        dimension: 超立方体维度
        output_path: 输出图片文件路径

    Returns:
        生成的图片文件路径
    """
    n_nodes = 2 ** dimension

    # 构建超立方体图
    G = nx.hypercube_graph(dimension)
    G = nx.relabel_nodes(G, {n: int("".join(map(str, n)), 2) for n in G.nodes()})

    # 收集所有边（u < v 去重）
    edges = []
    for u in range(n_nodes):
        for bit in range(dimension):
            v = u ^ (1 << bit)
            if u < v:
                edges.append((u, v))

    # 绘图
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')
    ax.axis('off')

    # 复用与 syndrome 可视化相同的布局
    pos = _hypercube_layout(dimension)

    # 统一黑色实线
    nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color='#333333',
                           style='solid', width=2.0, ax=ax)

    # 统一浅蓝色节点 + 黑色边框
    nx.draw_networkx_nodes(G, pos, node_color='#d0ebff', node_size=800,
                           edgecolors='#333333', linewidths=1.5, ax=ax)

    # 二进制编码标签
    binary_labels = {i: format(i, f'0{dimension}b') for i in range(n_nodes)}
    nx.draw_networkx_labels(G, pos, labels=binary_labels, font_size=9,
                            font_weight="bold", font_family='Times New Roman',
                            ax=ax)

    plt.tight_layout()

    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=SAVE_DPI, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()

    return output_path
