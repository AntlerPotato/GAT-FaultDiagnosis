"""
Input:
  - 标准库: abc
  - 第三方: 无
Output:
  - BaseTopology: 网络拓扑的抽象基类
Position: 定义网络拓扑的接口规范，所有具体拓扑必须继承此类

⚠️ 提醒：修改本文件后务必：
   1. 更新上方 Input/Output/Position 说明
   2. 更新所属文件夹的 README.md（如果影响模块结构）
   3. 更新根目录 README.md（如果影响核心架构）
"""

from abc import ABC, abstractmethod


class BaseTopology(ABC):
    """
    网络拓扑的基类
    
    所有具体的拓扑结构（如 Hypercube, Torus 等）都需要继承这个类，
    并实现 n_nodes 属性和 get_neighbors 方法。
    """
    
    @property
    @abstractmethod
    def n_nodes(self) -> int:
        """返回网络中的节点总数"""
        pass
    
    @abstractmethod
    def get_neighbors(self, node: int) -> list:
        """返回指定节点的所有邻居节点列表"""
        pass
    
    def get_all_edges(self) -> list:
        """
        返回网络中所有的有向边 (u, v)
        
        在 PMC 模型中，边 (u, v) 表示节点 u 测试节点 v
        """
        edges = []
        for u in range(self.n_nodes):
            for v in self.get_neighbors(u):
                edges.append((u, v))
        return edges
