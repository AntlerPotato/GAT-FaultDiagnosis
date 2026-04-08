"""基于超立方体拓扑的图神经网络 (GNN) 诊断模型

来源：学长项目 AI4FaultDiagnosis-main/models/gnn.py
用途：专利优化实验（patent-optimization/）中使用的 GNN 基线模型

设计思想：
- 将 PMC 模型下的 syndrome（一维向量）重排为"每个节点的按维度测试特征"
- 利用超立方体的邻接关系，做两层消息聚合（自特征 + 邻居特征）
- 通过 Sigmoid 输出每个节点为故障的概率

与 BPNN 的对比：
- BPNN 使用平铺特征的 MLP，不显式利用拓扑结构
- GNN 显式编码邻接，并在每层将邻居信息聚合到节点表征中，更适合 PMC 的结构化测试场景

Input:
  - 标准库: 无
  - 第三方: numpy, torch (torch.nn, torch.optim)
  - 本地: .base.BaseModel, utils.get_logger
Output:
  - GNN: 图神经网络诊断模型
Position: 学长专利实验中的 GNN 基线/优化模型，供 patent-optimization/ 脚本使用

⚠️ 提醒：修改本文件后务必：
   1. 更新上方 Input/Output/Position 说明
   2. 更新所属文件夹的 README.md（如果影响模块结构）
   3. 更新根目录 README.md（如果影响核心架构）
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .base import BaseModel
from utils import get_logger


class GNN(BaseModel):
    def __init__(self, input_size: int, output_size: int, hidden_sizes: list = None):
        """初始化网络结构

        Args:
            input_size: 输入维度 = syndrome 大小 = 节点数 × 维度
            output_size: 输出维度 = 节点数
            hidden_sizes: 两层隐藏的大小，默认 [节点数*4, 节点数*2]
        """
        self.n_nodes = output_size
        self.dim = input_size // output_size

        # 兼容处理：检查输入维度是否包含额外特征
        self.extra_features = False
        self.real_dim = int(np.log2(self.n_nodes))  # 真实的超立方体维度

        # 如果计算出的 dim > real_dim，说明包含了额外特征
        if self.dim > self.real_dim:
            self.extra_features = True
            self.n_extra = self.dim - self.real_dim
            self.dim = self.real_dim
            input_dim = self.dim + self.n_extra
        else:
            input_dim = self.dim

        # 构建索引映射：将平铺的一维 syndrome 映射为 [节点, 维度] 的特征
        index_map = torch.zeros((self.n_nodes, self.dim), dtype=torch.long)
        for v in range(self.n_nodes):
            for bit in range(self.dim):
                u = v ^ (1 << bit)
                index_map[v, bit] = u * self.dim + bit
        if hidden_sizes is None:
            hidden_sizes = [output_size * 4, output_size * 2]

        # 构建邻接矩阵：超立方体按翻转一个比特位相邻
        adj = torch.zeros((self.n_nodes, self.n_nodes), dtype=torch.float32)
        for u in range(self.n_nodes):
            for bit in range(self.dim):
                v = u ^ (1 << bit)
                adj[u, v] = 1.0
        # 归一化邻接：按度做平均，使消息聚合为"邻居平均"
        deg = adj.sum(dim=1, keepdim=True)
        adj_norm = adj / torch.clamp(deg, min=1.0)
        self.index_map = index_map
        self.adj_norm = adj_norm

        h1, h2 = hidden_sizes[0], hidden_sizes[1]
        # 两层"自特征 + 邻居特征"线性变换
        self.self1 = nn.Linear(input_dim, h1)
        self.neigh1 = nn.Linear(input_dim, h1)
        self.self2 = nn.Linear(h1, h2)
        self.neigh2 = nn.Linear(h1, h2)
        # 输出层：每个节点一个标量，后接 Sigmoid 得到概率
        self.out = nn.Linear(h2, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

        # 创建 nn.Module 容器以兼容 BaseModel.save_model/load_model
        self.network = nn.ModuleDict({
            'self1': self.self1,
            'neigh1': self.neigh1,
            'self2': self.self2,
            'neigh2': self.neigh2,
            'out': self.out,
            'dropout': self.dropout,
        })

        self.optimizer = optim.Adam(self.network.parameters(), lr=0.001)
        self.criterion = nn.BCELoss()

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：节点特征构造 → 邻居聚合 → 两层融合 → 概率输出"""
        B = x.shape[0]
        device = next(self.self1.parameters()).device

        # 1. 提取基础 Syndrome 特征
        syndrome_size = self.n_nodes * self.dim
        x_syndrome = x[:, :syndrome_size]

        # 将平铺输入重排为每个节点的按维度测试特征
        idx = self.index_map.view(1, -1).to(device).expand(B, -1)
        node_feats = x_syndrome.gather(1, idx).view(B, self.n_nodes, self.dim)

        # 2. 如果有额外特征，提取并拼接
        if self.extra_features:
            x_extra = x[:, syndrome_size:]
            node_extra = x_extra.view(B, self.n_nodes, self.n_extra)
            node_feats = torch.cat([node_feats, node_extra], dim=2)

        # 邻居平均的消息聚合矩阵（批次扩展）
        A = self.adj_norm.to(device).unsqueeze(0).expand(B, -1, -1)
        # 第1层：自特征 + 邻居聚合特征 → ReLU → Dropout
        s1 = self.self1(node_feats)
        n1_raw = torch.bmm(A, node_feats)
        n1 = self.neigh1(n1_raw)
        h1 = self.relu(s1 + n1)
        h1 = self.dropout(h1)
        # 第2层：在 h1 上重复自/邻居融合 → ReLU → Dropout
        s2 = self.self2(h1)
        n2_raw = torch.bmm(A, h1)
        n2 = self.neigh2(n2_raw)
        h2 = self.relu(s2 + n2)
        h2 = self.dropout(h2)
        # 输出概率：每节点一个标量，压缩掉最后一维
        out = torch.sigmoid(self.out(h2)).squeeze(-1)
        return out

    def train(self, train_data: tuple, val_data: tuple, epochs: int = 100,
              batch_size: int = 64) -> list:
        """训练流程：DataLoader 分批 → 前向/反向 → Adam 更新 → 验证损失

        Args:
            train_data: (X_train, Y_train)
            val_data: (X_val, Y_val)
            epochs: 训练轮数
            batch_size: 批次大小

        Returns:
            空列表（兼容接口）
        """
        logger = get_logger()
        X_train, Y_train = torch.FloatTensor(train_data[0]), torch.FloatTensor(train_data[1])
        X_val, Y_val = torch.FloatTensor(val_data[0]), torch.FloatTensor(val_data[1])
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_train, Y_train),
            batch_size=batch_size,
            shuffle=True
        )
        for epoch in range(epochs):
            self.dropout.train()
            train_loss = 0.0
            for bx, by in train_loader:
                self.optimizer.zero_grad()
                preds = self._forward(bx)
                loss = self.criterion(preds, by)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            with torch.no_grad():
                self.dropout.eval()
                val_preds = self._forward(X_val)
                val_loss = self.criterion(val_preds, Y_val).item()
            if (epoch + 1) % 20 == 0:
                avg_train_loss = train_loss / max(1, len(train_loader))
                logger.info(f"Epoch [{epoch+1}/{epochs}] Train: {avg_train_loss:.4f}, Val: {val_loss:.4f}")
        return []

    def predict(self, x: np.ndarray) -> np.ndarray:
        """预测每个节点的故障概率（范围 0~1）"""
        with torch.no_grad():
            inp = torch.FloatTensor(x)
            if inp.dim() == 1:
                inp = inp.unsqueeze(0)
            return self._forward(inp).squeeze().numpy()
