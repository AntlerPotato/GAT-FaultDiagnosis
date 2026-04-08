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
  - extract_weighted_features: 提取学长加权 PMC 统计特征（被控诉/控诉/加权被控诉）
  - batch_syndrome_to_features_wf: 批量提取双向特征 + 加权统计特征的拼接（供 GAT-WF 使用）
  - syndrome_to_node_features_wf: 单样本版本的 wf 特征提取
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
                               reverse_map: np.ndarray,
                               feature_mode: str = "bidirectional") -> np.ndarray:
    """
    批量将 syndrome 数组转换为节点特征矩阵（向量化，高性能）

    Args:
        X: syndrome 数组，shape = (n_samples, syndrome_size)
        topo: 超立方体拓扑
        reverse_map: 预计算的反向索引映射
        feature_mode: 特征模式，"bidirectional"（双向，2d 维）或 "unidirectional"（单向，d 维）

    Returns:
        特征数组，shape = (n_samples, n_nodes, 2*dim) 或 (n_samples, n_nodes, dim)
    """
    n_samples = X.shape[0]
    n_nodes = topo.n_nodes
    dim = topo.dim

    # 每个节点测试邻居的结果：直接 reshape
    test_others = X.reshape(n_samples, n_nodes, dim)

    if feature_mode == "unidirectional":
        return test_others.astype(np.float32)

    # 被邻居测试的结果：用预计算的索引向量化提取
    flat_indices = reverse_map.flatten()  # (n_nodes * dim,)
    tested_by = X[:, flat_indices].reshape(n_samples, n_nodes, dim)

    # 拼接
    features = np.concatenate([test_others, tested_by], axis=2).astype(np.float32)
    return features


def syndrome_to_node_features(syndrome: np.ndarray, topo: Hypercube,
                              reverse_map: np.ndarray = None,
                              feature_mode: str = "bidirectional") -> torch.Tensor:
    """
    从单个 syndrome 提取节点特征矩阵

    Args:
        syndrome: 一维 syndrome 数组，shape = (n_nodes * dim,)
        topo: 超立方体拓扑
        reverse_map: 预计算的反向索引映射（可选，不传则现场计算）
        feature_mode: 特征模式，"bidirectional" 或 "unidirectional"

    Returns:
        节点特征矩阵，shape = (n_nodes, 2*dim) 或 (n_nodes, dim)
    """
    if reverse_map is None:
        reverse_map = build_reverse_index_map(topo)
    features = batch_syndrome_to_features(
        syndrome.reshape(1, -1), topo, reverse_map, feature_mode=feature_mode
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


# ==================== 加权 PMC 统计特征（来源：学长专利优化思路）====================
# 在原始双向特征基础上，拼接 3 个聚合统计量，作为 GAT-WF（Weighted Features）变体的输入

def extract_weighted_features(X: np.ndarray, topo) -> np.ndarray:
    """
    批量提取加权 PMC 统计特征（向量化实现）

    来源：学长 patent-optimization/run_single_sample_vis.py extract_features()
    适配：改为批量向量化版本，兼容永久性故障 / 超立方体场景

    三个特征（每个节点）：
    - Accused（被控诉比例）:  A(u) = accused_count(u) / deg(u)
    - Accusing（控诉比例）:   C(u) = accusing_count(u) / deg(u)
    - WAccused（加权被控诉）: WA(u) = sum(w(v) * T(v,u)) / deg(u)
      其中 w(v) = max(0, 1 - A(v))，即测试者的可靠性权重

    Args:
        X: syndrome 数组，shape = (n_samples, syndrome_size)
        topo: 超立方体拓扑

    Returns:
        加权统计特征，shape = (n_samples, n_nodes, 3)
    """
    n_samples = X.shape[0]
    n_nodes = topo.n_nodes
    dim = topo.dim

    # X reshape 为 (n_samples, n_nodes, dim)：X[s, u, k] = T(u, neighbors[u][k])
    X_r = X.reshape(n_samples, n_nodes, dim)

    # Accusing: 节点 u 控诉他人的比例 = sum(T(u, v)) / dim
    # shape: (n_samples, n_nodes)
    accusing = X_r.sum(axis=2) / dim

    # Accused: 节点 u 被他人控诉的比例
    # 需要知道：对于节点 u，谁在测试 u？即邻居 v 测试 u 的结果 T(v, u)
    # 用 reverse_map 向量化提取：reverse_map[u, k] = 邻居 v 在自己的测试列表中测试 u 的 syndrome 位置
    # 这里直接使用 batch_syndrome_to_features 已有逻辑中的 tested_by 部分
    # shape: tested_by[s, u, k] = T(neighbors[u][k], u)
    reverse_map = build_reverse_index_map(topo)
    flat_indices = reverse_map.flatten()          # (n_nodes * dim,)
    tested_by = X[:, flat_indices].reshape(n_samples, n_nodes, dim)  # (n_samples, n_nodes, dim)

    # Accused: (n_samples, n_nodes)
    accused = tested_by.sum(axis=2) / dim

    # WAccused: 对 u 的加权被控诉
    # T(v, u) = tested_by[s, u, k]，对应权重 w(v) = max(0, 1 - accused[s, v])
    # 需要把 accusing[s, neighbors[u][k]] 对应为每条边的权重
    # 利用 index_map：index_map[v, bit] = u * dim + bit（v 测 u 在 syndrome 中的偏移量，u = v ^ (1<<bit)）
    # 等价地，反向：对节点 u 的第 k 个邻居 v = neighbors[u][k]
    # 向量化：构建 (n_nodes, dim) 的邻居索引，neighbor_ids[u, k] = neighbors[u][k]
    neighbor_ids = np.array([topo.get_neighbors(u) for u in range(n_nodes)], dtype=np.int64)
    # shape: (n_samples, n_nodes, dim) — 每条 T(v,u) 对应的测试者 v 的 被控诉比例
    accuser_accused = accused[:, neighbor_ids]   # (n_samples, n_nodes, dim)
    # 可靠性权重 w(v) = max(0, 1 - A(v))，A(v) 为测试者 v 的被控诉比例
    # 若测试者自身被大量指控为故障，其测试结果不可靠，权重降低
    weights = np.maximum(0.0, 1.0 - accuser_accused)
    # 加权被控诉：sum(w(v) * T(v,u)) / dim
    waccused = (weights * tested_by).sum(axis=2) / dim

    # 拼接为 (n_samples, n_nodes, 3)
    feat = np.stack([accused, accusing, waccused], axis=2).astype(np.float32)
    return feat


def batch_syndrome_to_features_wf(X: np.ndarray, topo,
                                   reverse_map: np.ndarray) -> np.ndarray:
    """
    批量提取 GAT-WF 使用的扩展节点特征：双向特征（2d 维）+ 加权统计特征（3 维）

    来源：融合原 GAT 双向特征和学长加权 PMC 统计特征的思路

    Args:
        X: syndrome 数组，shape = (n_samples, syndrome_size)
        topo: 超立方体拓扑
        reverse_map: 预计算的反向索引映射

    Returns:
        扩展特征，shape = (n_samples, n_nodes, 2*dim + 3)
    """
    # 双向特征: (n_samples, n_nodes, 2*dim)
    bidir = batch_syndrome_to_features(X, topo, reverse_map, feature_mode="bidirectional")
    # 加权统计特征: (n_samples, n_nodes, 3)
    wf = extract_weighted_features(X, topo)
    # 拼接: (n_samples, n_nodes, 2*dim + 3)
    return np.concatenate([bidir, wf], axis=2).astype(np.float32)


def syndrome_to_node_features_wf(syndrome: np.ndarray, topo,
                                  reverse_map: np.ndarray = None) -> torch.Tensor:
    """
    单样本版本：从 syndrome 提取 GAT-WF 扩展特征

    Args:
        syndrome: 一维 syndrome 数组，shape = (n_nodes * dim,)
        topo: 超立方体拓扑
        reverse_map: 预计算的反向索引映射（可选）

    Returns:
        节点特征矩阵，shape = (n_nodes, 2*dim + 3)
    """
    if reverse_map is None:
        reverse_map = build_reverse_index_map(topo)
    features = batch_syndrome_to_features_wf(
        syndrome.reshape(1, -1), topo, reverse_map
    )[0]
    return torch.tensor(features, dtype=torch.float)
