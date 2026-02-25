"""
本包职责：提供故障诊断模型的实现，包括抽象基类、BPNN 模型和 GAT 模型

包结构：
- base.py: 诊断模型的抽象基类 BaseModel
- bpnn.py: 反向传播神经网络模型 BPNN
- gat.py: 图注意力网络模型 GAT

对外接口：
从本包可导入：
- BaseModel: 诊断模型的抽象基类
- BPNN: 反向传播神经网络模型
- GAT: 图注意力网络模型

⚠️ 提醒：包内文件变化时必须更新本文件
"""

__all__ = ['BaseModel', 'BPNN', 'GAT']

from .base import BaseModel
from .bpnn import BPNN
from .gat import GAT

