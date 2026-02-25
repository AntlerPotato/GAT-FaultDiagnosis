"""
本包职责：提供网络拓扑结构的实现

包结构：
- base.py: 网络拓扑的抽象基类 BaseTopology
- hypercube.py: N维超立方体网络拓扑 Hypercube

对外接口：
从本包可导入：
- BaseTopology: 网络拓扑的抽象基类
- Hypercube: N维超立方体网络拓扑

⚠️ 提醒：包内文件变化时必须更新本文件
"""

__all__ = ['BaseTopology', 'Hypercube']

from .base import BaseTopology
from .hypercube import Hypercube

