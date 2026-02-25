"""
本包职责：提供数据生成、保存、加载和格式转换功能

包结构：
- generator.py: 生成故障诊断数据集
- dataset.py: 数据集的持久化（保存/加载）
- converter.py: syndrome 数据转换为 PyG 图数据格式

对外接口：
从本包可导入：
- generate_data: 生成并划分训练/验证/测试数据集
- save_dataset: 保存数据集到磁盘
- load_dataset: 从磁盘加载数据集
- build_edge_index: 从拓扑构建 PyG 边索引
- build_reverse_index_map: 预计算反向测试索引映射
- syndrome_to_node_features: 从 syndrome 提取节点特征
- batch_syndrome_to_features: 批量提取节点特征（向量化）
- create_dataloader: 批量转换数据集为 PyG DataLoader

⚠️ 提醒：包内文件变化时必须更新本文件
"""

__all__ = [
    'generate_data', 'save_dataset', 'load_dataset',
    'build_edge_index', 'build_reverse_index_map',
    'syndrome_to_node_features', 'batch_syndrome_to_features',
    'create_dataloader'
]

from .generator import generate_data
from .dataset import save_dataset, load_dataset
from .converter import (build_edge_index, build_reverse_index_map,
                        syndrome_to_node_features, batch_syndrome_to_features,
                        create_dataloader)
