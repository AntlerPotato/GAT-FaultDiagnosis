"""
Input:
  - 标准库: 无
  - 第三方: numpy, torch (torch.nn, torch.optim), torch_geometric (GATConv)
  - 本地: .base.BaseModel, data.converter (build_edge_index, build_reverse_index_map, syndrome_to_node_features, batch_syndrome_to_features), utils.logger.get_logger
Output:
  - GAT: 图注意力网络模型，用于从 syndrome 预测故障节点
Position: 实现 GAT 模型，继承自 BaseModel，利用图拓扑结构进行故障诊断

⚠️ 提醒：修改本文件后务必：
   1. 更新上方 Input/Output/Position 说明
   2. 更新所属文件夹的 README.md（如果影响模块结构）
   3. 更新根目录 README.md（如果影响核心架构）
"""

import gc

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GATConv
from torch_geometric.utils import add_self_loops as pyg_add_self_loops
from .base import BaseModel
from data.converter import build_edge_index, build_reverse_index_map, syndrome_to_node_features, batch_syndrome_to_features
from utils import get_logger


class FocalLoss(nn.Module):
    """
    Focal Loss：聚焦难分类样本，缓解类别不平衡

    公式：FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    - α：类别权重，控制正负样本的重要性
    - γ：聚焦参数，γ 越大越关注难分类样本
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: 模型输出 logits，shape = (N, 2)
            targets: 真实标签，shape = (N,)，值为 0 或 1
        """
        ce_loss = nn.functional.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        pt = pt.clamp(min=1e-7, max=1.0 - 1e-7)  # 数值稳定性
        # 为不同类别分配不同权重
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class GATDiagnosis(nn.Module):
    """
    GAT 故障诊断网络

    架构（参考 FaultGAT 论文）：
        输入特征 F ∈ R^{n_nodes × d_feature}
            ↓
        特征变换层: Linear → LayerNorm → LeakyReLU
            ↓
        GAT Layer 1（中间层）: 多头注意力 (K=8, hidden=64, concat=True) → BatchNorm → ReLU → Dropout
            ↓
        GAT Layer 2（最终层）: 多头注意力 (K=8, out=2, concat=False) → 直接输出 2 分类 logits
    """

    def __init__(self, in_features: int, n_heads: int = 8,
                 hidden_dim: int = 64, dropout: float = 0.3):
        """
        Args:
            in_features: 输入节点特征维度（2 * dimension）
            n_heads: 注意力头数
            hidden_dim: 每个头的隐藏维度
            dropout: Dropout 率
        """
        super().__init__()

        # 特征变换层：将原始特征映射到更高维空间
        self.feature_transform = nn.Sequential(
            nn.Linear(in_features, hidden_dim * n_heads),
            nn.LayerNorm(hidden_dim * n_heads),
            nn.LeakyReLU(0.2)
        )

        # GAT Layer 1：多头注意力
        # 输入维度 = hidden_dim * n_heads，输出每个头 hidden_dim
        # add_self_loops=False：自环在外部预添加到 edge_index 中，
        # 避免 GATConv 每次 forward 都调用 remove/add_self_loops 创建新张量
        self.gat1 = GATConv(
            in_channels=hidden_dim * n_heads,
            out_channels=hidden_dim,
            heads=n_heads,
            dropout=dropout,
            concat=True,  # 拼接多头输出，输出维度 = hidden_dim * n_heads
            add_self_loops=False
        )
        self.bn1 = nn.BatchNorm1d(hidden_dim * n_heads)
        self.dropout1 = nn.Dropout(dropout)

        # GAT Layer 2（最终层）：多头注意力，直接输出 2 分类 logits
        # concat=False 对多头结果取平均，输出维度 = out_channels = 2
        self.gat2 = GATConv(
            in_channels=hidden_dim * n_heads,
            out_channels=2,
            heads=n_heads,
            dropout=dropout,
            concat=False,  # 最终层对多头取平均（论文做法）
            add_self_loops=False
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 节点特征，shape = (n_nodes, in_features)
            edge_index: 边索引，shape = (2, n_edges)

        Returns:
            logits: shape = (n_nodes, 2)
        """
        # 特征变换
        x = self.feature_transform(x)

        # GAT Layer 1（中间层）
        x = self.gat1(x, edge_index)
        x = self.bn1(x)
        x = nn.functional.relu(x)
        x = self.dropout1(x)

        # GAT Layer 2（最终层，直接输出 logits）
        logits = self.gat2(x, edge_index)
        return logits


class GAT(BaseModel):
    """
    GAT 故障诊断模型（封装类）

    继承 BaseModel，提供与 BPNN 兼容的 train() 和 predict() 接口。
    内部自动处理 syndrome → PyG Data 的转换。
    """

    def __init__(self, topo, n_heads: int = 8, hidden_dim: int = 64,
                 dropout: float = 0.3, lr: float = 0.002):
        """
        Args:
            topo: 超立方体拓扑对象（需要用于构建图结构）
            n_heads: 注意力头数
            hidden_dim: 隐藏维度
            dropout: Dropout 率
            lr: 学习率
        """
        self.topo = topo
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 预添加自环，避免 GATConv 每次 forward 都 remove/add_self_loops
        edge_index_raw = build_edge_index(topo)
        edge_index_with_loops, _ = pyg_add_self_loops(edge_index_raw, num_nodes=topo.n_nodes)
        self.edge_index = edge_index_with_loops.to(self.device)
        self.reverse_map = build_reverse_index_map(topo)
        in_features = 2 * topo.dim  # 节点特征维度

        # 预缓存批次 edge_index，避免训练中反复创建张量
        # key = batch_size，value = 拼接后的 edge_index
        self._edge_cache: dict[int, torch.Tensor] = {}

        self.network = GATDiagnosis(
            in_features=in_features,
            n_heads=n_heads,
            hidden_dim=hidden_dim,
            dropout=dropout
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.network.parameters(), lr=lr, weight_decay=5e-4
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=15
        )
        # 损失函数说明：
        # 原设计（FaultGAT 论文 §III-C）使用 Focal Loss（α=0.25, γ=2.0），
        # 针对间歇性故障场景下的类别不平衡问题。
        # 但在永久性故障场景中，类别不平衡程度较温和，Focal Loss 的聚焦机制
        # 反而会抑制梯度，导致收敛不稳定。
        # 改用论文方法部分（§III-D）描述的标准交叉熵损失后效果更好。
        # ⚠️ 待与导师讨论此处的损失函数选择。
        # self.criterion = FocalLoss(alpha=0.25, gamma=2.0)  # 原设计，暂不使用
        self.criterion = nn.CrossEntropyLoss()

    def _get_batch_edge_index(self, bs: int) -> torch.Tensor:
        """
        获取指定 batch_size 的批次 edge_index（带缓存）

        对同构图批处理：将 bs 个单图的 edge_index 拼接为一个大图，
        通过偏移节点索引避免不同图的节点混淆。

        Args:
            bs: 批次中的图数量

        Returns:
            拼接后的 edge_index，shape = (2, bs * n_edges_per_graph)
        """
        if bs not in self._edge_cache:
            n_nodes = self.topo.n_nodes
            single_edge = self.edge_index
            offsets = torch.arange(bs, device=self.device).unsqueeze(1) * n_nodes
            edges = single_edge.unsqueeze(0).expand(bs, -1, -1) + offsets.unsqueeze(1)
            self._edge_cache[bs] = edges.permute(1, 0, 2).reshape(2, -1)
        return self._edge_cache[bs]

    def train(self, train_data: tuple, val_data: tuple, epochs: int = 200,
              batch_size: int = 64, patience: int = 50):
        """
        训练 GAT 模型（使用手动批处理，避免 PyG DataLoader 开销）

        Args:
            train_data: (X_train, Y_train)
            val_data: (X_val, Y_val)
            epochs: 训练轮数
            batch_size: 批次大小
            patience: Early Stopping 耐心值
        """
        logger = get_logger()
        n_nodes = self.topo.n_nodes
        device = self.device

        # 预计算所有特征（向量化，一次性完成），并移至 device
        from data.converter import batch_syndrome_to_features
        train_X = torch.tensor(
            batch_syndrome_to_features(train_data[0], self.topo, self.reverse_map),
            dtype=torch.float, device=device
        )
        train_Y = torch.tensor(train_data[1], dtype=torch.long, device=device)
        val_X = torch.tensor(
            batch_syndrome_to_features(val_data[0], self.topo, self.reverse_map),
            dtype=torch.float, device=device
        )
        val_Y = torch.tensor(val_data[1], dtype=torch.long, device=device)

        n_train = len(train_data[0])
        best_val_f1 = 0.0
        best_state = None
        no_improve = 0

        for epoch in range(epochs):
            # ========== 训练阶段 ==========
            self.network.train()
            perm = torch.randperm(n_train, device=device)
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for start in range(0, n_train, batch_size):
                idx = perm[start:start + batch_size]
                bs = len(idx)

                # 拼接为大图：(bs * n_nodes, features)
                x_batch = train_X[idx].reshape(bs * n_nodes, -1)
                y_batch = train_Y[idx].reshape(bs * n_nodes)
                edge_batch = self._get_batch_edge_index(bs)

                self.optimizer.zero_grad()
                logits = self.network(x_batch, edge_batch)
                loss = self.criterion(logits, y_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
                self.optimizer.step()

                train_loss += loss.item() * bs
                preds = logits.detach().argmax(dim=1)
                train_correct += (preds == y_batch).sum().item()
                train_total += y_batch.size(0)
                # 显式释放计算图中间变量，防止 CPU 内存碎片累积
                del logits, loss, preds

            avg_train_loss = train_loss / n_train

            # ========== 验证阶段 ==========
            val_f1, val_acc = self._evaluate_tensors(val_X, val_Y, batch_size)

            self.scheduler.step(val_f1)

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_state = {k: v.clone() for k, v in self.network.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if (epoch + 1) % 20 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                train_acc = train_correct / train_total if train_total > 0 else 0
                logger.info(
                    f"Epoch [{epoch+1}/{epochs}] "
                    f"Loss: {avg_train_loss:.4f}, "
                    f"Train Acc: {train_acc*100:.1f}%, "
                    f"Val Acc: {val_acc*100:.1f}%, "
                    f"Val F1: {val_f1*100:.1f}%, "
                    f"LR: {current_lr:.6f}"
                )

            # 每个 epoch 强制 GC，防止 CPU 模式下内存持续增长
            gc.collect()

            if no_improve >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}, best Val F1: {best_val_f1*100:.2f}%")
                break

        if best_state is not None:
            self.network.load_state_dict(best_state)
            logger.info(f"Restored best model with Val F1: {best_val_f1*100:.2f}%")

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        预测节点故障概率

        与 BPNN.predict() 接口兼容：输入单个 syndrome，输出概率数组。

        Args:
            x: 单个 syndrome，shape = (syndrome_size,)

        Returns:
            每个节点的故障概率，shape = (n_nodes,)
        """
        self.network.eval()
        with torch.no_grad():
            node_features = syndrome_to_node_features(x, self.topo, self.reverse_map)
            node_features = node_features.to(self.device)
            logits = self.network(node_features, self.edge_index)
            # softmax 取故障类（index=1）的概率
            probs = torch.softmax(logits, dim=1)[:, 1]
            return probs.cpu().numpy()

    def _evaluate_tensors(self, X: torch.Tensor, Y: torch.Tensor,
                          batch_size: int = 64) -> tuple:
        """
        在预计算的特征张量上评估模型

        Args:
            X: 特征张量，shape = (n_samples, n_nodes, features)
            Y: 标签张量，shape = (n_samples, n_nodes)
            batch_size: 批次大小

        Returns:
            (f1_score, accuracy) 元组
        """
        self.network.eval()
        n_nodes = self.topo.n_nodes
        all_preds = []
        all_labels = []

        with torch.inference_mode():
            for start in range(0, len(X), batch_size):
                x_batch = X[start:start + batch_size]
                y_batch = Y[start:start + batch_size]
                bs = len(x_batch)

                x_flat = x_batch.reshape(bs * n_nodes, -1)
                y_flat = y_batch.reshape(bs * n_nodes)
                edge_batch = self._get_batch_edge_index(bs)

                logits = self.network(x_flat, edge_batch)
                preds = logits.argmax(dim=1)
                all_preds.append(preds)
                all_labels.append(y_flat)

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        correct = (all_preds == all_labels).sum().item()
        total = all_labels.size(0)
        accuracy = correct / total

        tp = ((all_preds == 1) & (all_labels == 1)).sum().item()
        fp = ((all_preds == 1) & (all_labels == 0)).sum().item()
        fn = ((all_preds == 0) & (all_labels == 1)).sum().item()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return f1, accuracy
