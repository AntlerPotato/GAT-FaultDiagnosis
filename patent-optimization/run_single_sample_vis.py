"""
来源：学长项目 AI4FaultDiagnosis-main/run_single_sample_vis.py
用途：提供 extract_features() 和 visualize_comparison() 供 patent-optimization 脚本使用

extract_features(): 从 syndrome 中提取加权 PMC 统计特征（被控诉比例、控诉比例、加权被控诉）
visualize_comparison(): 可视化普通 GNN vs 优化 GNN 的诊断对比
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from topologies import Hypercube


def extract_features(topo: Hypercube, syndrome: np.ndarray) -> np.ndarray:
    """
    特征提取：从 syndrome 计算三个加权 PMC 统计特征

    三个特征（每个节点）：
    - Accused: 被指控比例 A(x) = accused_count(x) / deg(x)
    - Accusing: 指控比例 C(x) = accusing_count(x) / deg(x)
    - WAccused: 加权被指控 = sum(w(u) for each T(u,v)=1) / deg(x)
      其中 w(u) = max(0, 1 - A(u))，即测试者的可靠性权重

    Args:
        topo: 超立方体拓扑
        syndrome: 一维 syndrome 数组

    Returns:
        展平的特征向量，shape = (n_nodes * 3,)
    """
    n_nodes = topo.n_nodes
    features = np.zeros((n_nodes, 2), dtype=np.float32)
    syndrome_idx = 0
    accused_counts = np.zeros(n_nodes)
    accusing_counts = np.zeros(n_nodes)

    for u in range(n_nodes):
        neighbors = topo.get_neighbors(u)
        for v in neighbors:
            result = syndrome[syndrome_idx]
            syndrome_idx += 1
            if result == 1:
                accusing_counts[u] += 1
                accused_counts[v] += 1

    dim = topo.dim
    norm_accused = accused_counts / dim
    norm_accusing = accusing_counts / dim

    # === 加权特征 ===
    weighted_accused_counts = np.zeros(n_nodes)
    syndrome_idx = 0
    for u in range(n_nodes):
        neighbors = topo.get_neighbors(u)
        for v in neighbors:
            result = syndrome[syndrome_idx]
            syndrome_idx += 1
            if result == 1:
                weight_u = max(0, 1.0 - norm_accused[u])
                weighted_accused_counts[v] += weight_u

    norm_weighted_accused = weighted_accused_counts / dim
    features[:, 0] = norm_accused
    features[:, 1] = norm_accusing
    features = np.column_stack((features, norm_weighted_accused))
    return features.flatten()


def visualize_comparison(topo, syndrome, true_labels, pred_ord, pred_opt,
                         save_path="comparison.png"):
    """
    可视化对比结果：Ground Truth / Ordinary GNN / Optimized GNN

    Args:
        topo: 超立方体拓扑
        syndrome: syndrome 数组
        true_labels: 真实标签
        pred_ord: 普通 GNN 预测
        pred_opt: 优化 GNN 预测
        save_path: 保存路径
    """
    dim = topo.dim
    n_nodes = topo.n_nodes
    G = nx.hypercube_graph(dim)
    mapping = {n: int("".join(map(str, n)), 2) for n in G.nodes()}
    G = nx.relabel_nodes(G, mapping)

    pos = nx.spring_layout(G, seed=42)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    titles = ["Ground Truth", "Ordinary GNN Prediction", "Optimized GNN Prediction"]
    label_sets = [true_labels, pred_ord, pred_opt]

    for ax, title, labels in zip(axes, titles, label_sets):
        node_colors = ["#ff6b6b" if labels[i] == 1 else "#69db7c"
                       for i in range(n_nodes)]

        if title != "Ground Truth":
            errors = np.where(labels != true_labels)[0]
            node_edge_colors = ["red" if i in errors else "black"
                                for i in range(n_nodes)]
            linewidths = [3.0 if i in errors else 1.0 for i in range(n_nodes)]
        else:
            node_edge_colors = "black"
            linewidths = 1.0

        nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax)
        nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                               edgecolors=node_edge_colors,
                               linewidths=linewidths, node_size=300, ax=ax)

        labels_map = {i: str(i) for i in range(n_nodes)}
        nx.draw_networkx_labels(G, pos, labels=labels_map, font_size=8, ax=ax)

        ax.set_title(title)
        ax.axis("off")

        if title != "Ground Truth":
            acc = np.mean(labels == true_labels) * 100
            ax.text(0.5, -0.05, f"Accuracy: {acc:.2f}%",
                    transform=ax.transAxes, ha='center', fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Visualization saved to {save_path}")
