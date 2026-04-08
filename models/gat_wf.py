"""
GAT-WF（Weighted Features）：融合学长加权 PMC 统计特征的 GAT 变体

注意：本文件是实验性变体，对应原始 models/gat.py 保持不变。
      如果实验证明有效，可替换原始文件；否则删除本文件。

改动点（相对于 gat.py）：
- 节点特征维度从 2d 扩展为 2d+3（拼接 Accused / Accusing / WAccused 三个统计量）
- 使用 data.converter 中新增的 batch_syndrome_to_features_wf / syndrome_to_node_features_wf
- 其余架构、超参数、训练逻辑与 gat.py 完全一致

来源背景：
  学长专利优化思路（patent-optimization/run_single_sample_vis.py）：
  在原始 syndrome 基础上增加加权 PMC 统计特征，可为 GNN 提供更强的局部故障信号。
  本变体将该思路移植到 GAT 框架下，验证其对 GAT 效果的影响。

Input:
  - 标准库: 无
  - 第三方: numpy, torch (torch.nn, torch.optim), torch_geometric (GATConv)
  - 本地: .gat.GATDiagnosis（复用原始网络结构）, .base.BaseModel,
          data.converter (build_edge_index, build_reverse_index_map,
                          batch_syndrome_to_features_wf, syndrome_to_node_features_wf),
          utils.logger.get_logger
Output:
  - GATWF: 融合加权特征的 GAT 故障诊断模型
Position: GAT-WF 实验变体，供与原始 GAT 对比，验证加权统计特征的有效性

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
from torch_geometric.utils import add_self_loops as pyg_add_self_loops
from .base import BaseModel
from .gat import GATDiagnosis  # 复用完全相同的网络结构，只有输入维度不同
from data.converter import (
    build_edge_index, build_reverse_index_map,
    batch_syndrome_to_features_wf, syndrome_to_node_features_wf
)
from utils import get_logger


class GATWF(BaseModel):
    """
    GAT-WF 故障诊断模型（融合加权 PMC 统计特征的变体）

    与原始 GAT 的唯一差异：
    - 输入节点特征维度 = 2*dim + 3（双向特征 + Accused/Accusing/WAccused）
    - 其余架构、超参数、训练逻辑完全一致

    继承 BaseModel，提供与 BPNN/GAT 兼容的 train() 和 predict() 接口。
    """

    def __init__(self, topo, n_heads: int = 8, hidden_dim: int = 64,
                 dropout: float = 0.3, lr: float = 0.002,
                 n_layers: int = 2, no_regularization: bool = False):
        """
        Args:
            topo: 超立方体拓扑对象
            n_heads: 注意力头数
            hidden_dim: 隐藏维度
            dropout: Dropout 率
            lr: 学习率
            n_layers: GAT 层数（1 或 2）
            no_regularization: 是否关闭 BatchNorm 和 Dropout
        """
        self.topo = topo
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 预添加自环
        edge_index_raw = build_edge_index(topo)
        edge_index_with_loops, _ = pyg_add_self_loops(edge_index_raw, num_nodes=topo.n_nodes)
        self.edge_index = edge_index_with_loops.to(self.device)
        self.reverse_map = build_reverse_index_map(topo)

        # 输入特征维度 = 2*dim + 3（双向 + 加权统计）
        in_features = 2 * topo.dim + 3

        self._edge_cache: dict[int, torch.Tensor] = {}

        self.network = GATDiagnosis(
            in_features=in_features,
            n_heads=n_heads,
            hidden_dim=hidden_dim,
            dropout=dropout,
            n_layers=n_layers,
            no_regularization=no_regularization
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.network.parameters(), lr=lr, weight_decay=5e-4
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=15
        )
        self.criterion = nn.CrossEntropyLoss()

    def _get_batch_edge_index(self, bs: int) -> torch.Tensor:
        """获取指定 batch_size 的批次 edge_index（带缓存）"""
        if bs not in self._edge_cache:
            n_nodes = self.topo.n_nodes
            single_edge = self.edge_index
            offsets = torch.arange(bs, device=self.device).unsqueeze(1) * n_nodes
            edges = single_edge.unsqueeze(0).expand(bs, -1, -1) + offsets.unsqueeze(1)
            self._edge_cache[bs] = edges.permute(1, 0, 2).reshape(2, -1)
        return self._edge_cache[bs]

    def train(self, train_data: tuple, val_data: tuple, epochs: int = 200,
              batch_size: int = 64, patience: int = 50) -> list:
        """
        训练 GATWF 模型

        Args:
            train_data: (X_train, Y_train)
            val_data: (X_val, Y_val)
            epochs: 训练轮数
            batch_size: 批次大小
            patience: Early Stopping 耐心值

        Returns:
            每 epoch 的损失记录列表
        """
        logger = get_logger()
        loss_history = []
        n_nodes = self.topo.n_nodes
        device = self.device

        # 预计算所有扩展特征（双向 + 加权统计）
        train_X = torch.tensor(
            batch_syndrome_to_features_wf(train_data[0], self.topo, self.reverse_map),
            dtype=torch.float, device=device
        )
        train_Y = torch.tensor(train_data[1], dtype=torch.long, device=device)
        val_X = torch.tensor(
            batch_syndrome_to_features_wf(val_data[0], self.topo, self.reverse_map),
            dtype=torch.float, device=device
        )
        val_Y = torch.tensor(val_data[1], dtype=torch.long, device=device)

        n_train = len(train_data[0])
        best_val_f1 = 0.0
        best_state = None
        no_improve = 0

        for epoch in range(epochs):
            self.network.train()
            perm = torch.randperm(n_train, device=device)
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for start in range(0, n_train, batch_size):
                idx = perm[start:start + batch_size]
                bs = len(idx)

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
                del logits, loss, preds

            avg_train_loss = train_loss / n_train
            val_f1, val_acc = self._evaluate_tensors(val_X, val_Y, batch_size)
            self.scheduler.step(val_f1)

            loss_history.append({
                "epoch": epoch + 1,
                "train_loss": round(avg_train_loss, 6),
                "val_f1": round(val_f1, 6)
            })

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

            gc.collect()

            if no_improve >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}, best Val F1: {best_val_f1*100:.2f}%")
                break

        if best_state is not None:
            self.network.load_state_dict(best_state)
            logger.info(f"Restored best model with Val F1: {best_val_f1*100:.2f}%")

        return loss_history

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        预测节点故障概率

        Args:
            x: 单个 syndrome，shape = (syndrome_size,)

        Returns:
            每个节点的故障概率，shape = (n_nodes,)
        """
        self.network.eval()
        with torch.no_grad():
            node_features = syndrome_to_node_features_wf(x, self.topo, self.reverse_map)
            node_features = node_features.to(self.device)
            logits = self.network(node_features, self.edge_index)
            probs = torch.softmax(logits, dim=1)[:, 1]
            return probs.cpu().numpy()

    def _evaluate_tensors(self, X: torch.Tensor, Y: torch.Tensor,
                          batch_size: int = 64) -> tuple:
        """在预计算的特征张量上评估模型，返回 (f1, accuracy)"""
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
