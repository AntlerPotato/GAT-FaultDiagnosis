"""
本包职责：提供故障诊断模型的实现，包括抽象基类、BPNN 模型、GAT 模型、GNN 模型和 GATWF 变体

包结构：
- base.py:   诊断模型的抽象基类 BaseModel
- bpnn.py:   反向传播神经网络模型 BPNN
- gat.py:    图注意力网络模型 GAT
- gat_wf.py: GAT-WF 变体（融合加权 PMC 统计特征，实验性）
- gnn.py:    图神经网络模型 GNN（学长专利实验用）

对外接口：
从本包可导入：
- BaseModel: 诊断模型的抽象基类
- BPNN: 反向传播神经网络模型
- GAT: 图注意力网络模型
- GATWF: 融合加权 PMC 统计特征的 GAT 变体（实验性）
- GNN: 图神经网络模型（专利实验用）

⚠️ 提醒：包内文件变化时必须更新本文件
"""

__all__ = ['BaseModel', 'BPNN', 'GAT', 'GATWF', 'GNN']

from .base import BaseModel
from .bpnn import BPNN
from .gat import GAT
from .gat_wf import GATWF
from .gnn import GNN

