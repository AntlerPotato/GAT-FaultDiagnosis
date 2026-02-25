"""
Input:
  - 标准库: 无
  - 第三方: numpy, torch, torch_geometric (Data, DataLoader)
  - 本地: topologies.hypercube.Hypercube
Output:
  - build_edge_index: 从拓扑构建 PyG 边索引
  - build_reverse_index_map: 预计算反向测试索引映射
  - syndrome_to_node_features: 从 syndrome 提取节点特征矩阵
  - batch_syndrome_to_features: 批量提取节点特征（向量化）
  - syndrome_to_graph: 将单个 syndrome 转为 PyG Data 对象
  - create_dataloader: 批量转换数据集为 PyG DataLoader
Position: 将现有 syndrome 数据转换为 PyG 图数据格式，供 GAT 模型使用

⚠️ 提醒：修改本文件后务必：
   1. 更新上方 Input/Output/Position 说明
   2. 更新所属文件夹的 README.md（如果影响模块结构）
   3. 更新根目录 README.md（如果影响核心架构）
"""

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from topologies.hypercube import Hypercube


def build_edge_index(topo: Hypercube) -> torch.Tensor:
    """
    从超立方体拓扑构建 PyG 格式的边索引（双向边）

    Args:
        topo: 超立方体拓扑对象

    Returns:
        edge_index: shape = (2, num_edges)，每列是一条有向边 (src, dst)
    """
    src, dst = [], []
    for u in range(topo.n_nodes):
        for v in topo.get_neighbors(u):
            src.append(u)
            dst.append(v)
    return torch.tensor([src, dst], dtype=torch.long)


def build_reverse_index_map(topo: Hypercube) -> np.ndarray:
    """
    预计算反向测试索引映射（一次性计算，后续复用）

    对于节点 u 的第 k 个邻居 v，找到 v 测试 u 的结果在 syndrome 中的位置。
    返回索引数组 reverse_map[u, k] = v * dim + idx_in_v

    Args:
        topo: 超立方体拓扑

    Returns:
        reverse_map: shape = (n_nodes, dim)，存储 syndrome 中的索引
    """
    n_nodes = topo.n_nodes
    dim = topo.dim
    reverse_map = np.zeros((n_nodes, dim), dtype=np.int64)

    for u in range(n_nodes):
        neighbors = topo.get_neighbors(u)
        for k, v in enumerate(neighbors):
            v_neighbors = topo.get_neighbors(v)
            idx_in_v = v_neighbors.index(u)
            reverse_map[u, k] = v * dim + idx_in_v

    return reverse_map


def batch_syndrome_to_features(X: np.ndarray, topo: Hypercube,
                               reverse_map: np.ndarray) -> np.ndarray:
    """
    批量将 syndrome 数组转换为节点特征矩阵（向量化，高性能）

    Args:
        X: syndrome 数组，shape = (n_samples, syndrome_size)
        topo: 超立方体拓扑
        reverse_map: 预计算的反向索引映射

    Returns:
        特征数组，shape = (n_samples, n_nodes, 2 * dim)
    """
    n_samples = X.shape[0]
    n_nodes = topo.n_nodes
    dim = topo.dim

    # 每个节点测试邻居的结果：直接 reshape
    test_others = X.reshape(n_samples, n_nodes, dim)

    # 被邻居测试的结果：用预计算的索引向量化提取
    flat_indices = reverse_map.flatten()  # (n_nodes * dim,)
    tested_by = X[:, flat_indices].reshape(n_samples, n_nodes, dim)

    # 拼接
    features = np.concatenate([test_others, tested_by], axis=2).astype(np.float32)
    return features


def syndrome_to_node_features(syndrome: np.ndarray, topo: Hypercube,
                              reverse_map: np.ndarray = None) -> torch.Tensor:
    """
    从单个 syndrome 提取节点特征矩阵

    Args:
        syndrome: 一维 syndrome 数组，shape = (n_nodes * dim,)
        topo: 超立方体拓扑
        reverse_map: 预计算的反向索引映射（可选，不传则现场计算）

    Returns:
        节点特征矩阵，shape = (n_nodes, 2 * dim)
    """
    if reverse_map is None:
        reverse_map = build_reverse_index_map(topo)
    features = batch_syndrome_to_features(
        syndrome.reshape(1, -1), topo, reverse_map
    )[0]
    return torch.tensor(features, dtype=torch.float)


def create_dataloader(X: np.ndarray, Y: np.ndarray, topo: Hypercube,
                      batch_size: int = 32, shuffle: bool = True) -> DataLoader:
    """
    将整个数据集转换为 PyG DataLoader（高性能批量转换）

    Args:
        X: syndrome 数组，shape = (n_samples, syndrome_size)
        Y: 标签数组，shape = (n_samples, n_nodes)
        topo: 超立方体拓扑
        batch_size: 批次大小
        shuffle: 是否打乱

    Returns:
        PyG DataLoader
    """
    edge_index = build_edge_index(topo)
    reverse_map = build_reverse_index_map(topo)

    # 批量转换所有特征（向量化）
    all_features = batch_syndrome_to_features(X, topo, reverse_map)

    data_list = []
    for i in range(len(X)):
        x = torch.tensor(all_features[i], dtype=torch.float)
        y = torch.tensor(Y[i], dtype=torch.long)
        data_list.append(Data(x=x, edge_index=edge_index, y=y))

    return DataLoader(data_list, batch_size=batch_size, shuffle=shuffle)
