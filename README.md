# GAT-FaultDiagnosis

基于图注意力网络（GAT）的超立方体互连网络故障诊断

## 项目简介

在 PMC 测试模型下，使用 GAT 和 BPNN 两种神经网络实现超立方体拓扑的系统级故障诊断。输入测试综合征（syndrome），输出故障节点集合。

- **BPNN 基线**：借鉴 Elhadef & Nayak (2012) 思路，在 PMC 模型下实现
- **GAT 模型**：利用图注意力机制处理超立方体拓扑结构，提升高维度、高故障率场景下的诊断性能

## 整体流程

```
┌─────────────────────────────────────────────────────────────────┐
│                         数据生成                                 │
│  随机选择故障节点 → PMC 模型生成 syndrome → 划分 train/val/test    │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│                      数据格式转换（GAT）                          │
│  syndrome → 节点特征矩阵 + 超立方体边索引 → PyG Data 对象          │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│                         模型训练                                 │
│  BPNN: syndrome → 全连接网络 → 节点故障概率                       │
│  GAT:  节点特征 + 图结构 → 图注意力网络 → 节点分类                 │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│                         评估与记录                               │
│  计算 Accuracy/Precision/Recall/F1 + 推理时延 → 保存 JSON 记录    │
└─────────────────────────────────────────────────────────────────┘
```

## 环境要求

- Python >= 3.10
- PyTorch
- PyTorch Geometric
- NumPy, Matplotlib, NetworkX

## 快速开始

```bash
# 1. 克隆 & 进入项目
git clone https://github.com/AntlerPotato/GAT-FaultDiagnosis.git
cd GAT-FaultDiagnosis

# 2. 创建虚拟环境
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# 3. 安装依赖
pip install -r requirements.txt

# 4. 运行
python main.py -d 4 -f 0.25 -n 5000 -e 200 -m both
```

## 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-d, --dimension` | 超立方体维度（节点数 = 2^d） | 4 |
| `-f, --faults` | 故障数（整数）或故障率（小数） | 0.25 |
| `-n, --n_samples` | 总样本数 | 5000 |
| `-e, --epochs` | 训练轮数 | 200 |
| `-m, --model` | 模型选择：bpnn / gat / both | bpnn |
| `--save NAME` | 保存数据集 | - |
| `--load NAME` | 加载数据集 | - |
| `--visualize PATH` | 可视化单个 syndrome | - |

**示例**:

```bash
# BPNN 训练
python main.py -d 4 -f 0.25 -n 5000 -e 200

# GAT 训练
python main.py -d 6 -f 0.25 -n 5000 -e 200 -m gat

# BPNN vs GAT 对比（结果自动保存到 TrainingRecords/）
python main.py -d 6 -f 0.25 -n 5000 -e 200 -m both

# 保存/加载数据集
python main.py -n 5000 --save my_data
python main.py --load my_data -m gat

# 可视化单个 syndrome
python main.py --visualize datasets/my_data/xxx/train/1.npz
```

## 项目结构

```
GAT-FaultDiagnosis/
├── topologies/          # 网络拓扑 + PMC syndrome 生成
│   ├── __init__.py
│   ├── base.py          # 拓扑抽象基类 (BaseTopology)
│   └── hypercube.py     # N维超立方体拓扑 (Hypercube)
├── models/              # 诊断模型
│   ├── __init__.py
│   ├── base.py          # 模型抽象基类 (BaseModel)
│   ├── bpnn.py          # BPNN 模型（基线）
│   └── gat.py           # GAT 模型（2层, 8头, 64维隐藏层）
├── data/                # 数据生成与管理
│   ├── __init__.py
│   ├── generator.py     # 训练/验证/测试数据集生成
│   ├── dataset.py       # 数据集持久化（保存/加载）
│   └── converter.py     # syndrome → PyG Data 格式转换
├── evaluation/          # 评估模块
│   ├── __init__.py
│   └── metrics.py       # Accuracy / Precision / Recall / F1
├── utils/               # 工具
│   ├── __init__.py
│   ├── logger.py        # 日志配置
│   └── visualizer.py    # Syndrome 可视化
├── papers/              # 参考论文
├── datasets/            # 保存的数据集（运行时生成，不入库）
├── TrainingRecords/     # 实验记录 JSON（按时间戳子文件夹组织，不入库）
├── main.py              # 项目入口
├── requirements.txt     # 项目依赖
├── CLAUDE.md            # Claude Code 项目规范
└── .gitignore
```

## 核心概念

### PMC 模型

| 测试者 | 被测者 | 结果 |
|--------|--------|------|
| 正常 | 正常 | 0 |
| 正常 | 故障 | 1 |
| 故障 | 任意 | 不可靠 |

### 超立方体

- N 维有 2^N 个节点，每个节点 N 个邻居
- 节点二进制编号只差一位即为邻居
- 例：节点 5 (101) 的邻居是 4 (100), 7 (111), 1 (001)

### GAT 节点特征

节点 u 的特征向量为其双向测试结果拼接：
- 前半部分：u 测试各邻居的结果
- 后半部分：各邻居测试 u 的结果
- 特征维度 = 2d（d 为超立方体维度）

## 实验记录

每次训练自动保存 JSON 到 `TrainingRecords/{时间戳}/`，包含：
- 实验配置（维度、故障率、样本数、epoch、随机种子）
- 评估指标（Accuracy、Precision、Recall、F1）
- 效率指标（模型参数量、训练耗时、推理时延）
- 完整 Loss 曲线（每 epoch 的 loss 数据）

## 扩展指南

### 添加新拓扑

```python
# topologies/torus.py
class Torus(BaseTopology):
    @property
    def n_nodes(self) -> int: ...
    @property
    def syndrome_size(self) -> int: ...
    def get_neighbors(self, node: int) -> list: ...
    def generate_PMC_syndrome(self, faulty_nodes: set): ...
```

### 添加新模型

```python
# models/new_model.py
class NewModel(BaseModel):
    def train(self, train_data, val_data, epochs): ...
    def predict(self, x): ...
```

## License

MIT
