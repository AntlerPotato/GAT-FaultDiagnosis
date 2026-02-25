# AI4FaultDiagnosis

基于神经网络的系统级故障诊断

## 项目简介

使用 BPNN（反向传播神经网络）实现 PMC 模型下的故障诊断。输入测试综合征，输出故障节点集合。

**参考论文**: *Comparison-Based System-Level Fault Diagnosis: A Neural Network Approach* (Elhadef & Nayak, 2012)

## 整体流程

```
┌─────────────────────────────────────────────────────────────────┐
│                         数据生成                                 │
│  随机选择故障节点 → PMC 模型生成 syndrome → 划分 train/val/test   │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│                         模型训练                                 │
│  syndrome (输入) → BPNN → 故障概率 (输出) → 与标签比较 → 更新权重  │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│                         模型测试                                 │
│  syndrome → BPNN → 概率 > 0.5 判定故障 → 计算 Acc/Prec/Recall    │
└─────────────────────────────────────────────────────────────────┘
```

### 1. 数据生成

```
输入: 超立方体维度 d, 最大故障数 t, 样本数 n

对于每个样本:
    1. 随机选择 1~t 个故障节点
    2. 按 PMC 规则生成 syndrome (每个节点测试其邻居)
    3. 生成标签 (故障=1, 正常=0)

输出: X (syndrome), Y (标签), 按 80/10/10 划分
```

### 2. 模型训练

```
BPNN 结构:
    输入层 [syndrome_size] → 隐藏层 [n*4, n*2] → 输出层 [n_nodes]

训练循环:
    for epoch in epochs:
        前向传播 → 计算 BCE Loss → 反向传播 → 更新权重
        验证集评估 (监控过拟合)
```

### 3. 模型测试

```
评估指标:
    Accuracy  = 完全正确识别所有故障节点的比例
    Precision = 预测故障中真正故障的比例  
    Recall    = 真正故障中被正确预测的比例
```

## 环境要求

- Python >= 3.10
- macOS / Linux / Windows

## 快速开始

```bash
# 1. 克隆 & 进入项目
git clone https://github.com/your-repo/AI4FaultDiagnosis.git
cd AI4FaultDiagnosis

# 2. 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 运行
python main.py
```

## 命令行参数

| 参数                 | 说明                           | 默认值 |
| -------------------- | ------------------------------ | ------ |
| `-d, --dimension`  | 超立方体维度（节点数 = 2^d）   | 4      |
| `-f, --faults`     | 故障数（整数）或故障率（小数） | 0.25   |
| `-n, --n_samples`  | 总样本数                       | 1000   |
| `-e, --epochs`     | 训练轮数                       | 100    |
| `--save NAME`      | 保存数据集                     | -      |
| `--load NAME`      | 加载数据集                     | -      |
| `--visualize PATH` | 可视化单个 syndrome           | -      |

**示例**:

```bash
python main.py -d 5 -f 6 -n 5000           # 训练
python main.py -n 2000 --save my_data      # 保存数据集
python main.py --load my_data              # 加载数据集
python main.py --visualize datasets/my_data/xxx/train/1.npz  # 可视化
```

## 项目结构

```
AI4FaultDiagnosis/
├── topologies/          # 网络拓扑 + PMC syndrome 生成
│   ├── __init__.py      # 包初始化，导出 BaseTopology, Hypercube
│   ├── base.py          # 网络拓扑的抽象基类
│   └── hypercube.py     # N维超立方体网络拓扑
├── models/              # 诊断模型
│   ├── __init__.py      # 包初始化，导出 BaseModel, BPNN
│   ├── base.py          # 诊断模型的抽象基类
│   └── bpnn.py          # 反向传播神经网络模型
├── data/                # 数据生成与管理
│   ├── __init__.py      # 包初始化，导出数据相关函数
│   ├── generator.py     # 生成 train/val/test 数据集
│   └── dataset.py       # 数据集的持久化（保存/加载）
├── evaluation/          # 评估模块
│   ├── __init__.py      # 包初始化，导出评估函数
│   └── metrics.py       # 模型评估指标计算
├── utils/               # 工具
│   ├── __init__.py      # 包初始化，导出工具函数
│   ├── logger.py        # 日志配置和获取
│   └── visualizer.py    # Syndrome 可视化
├── papers/              # 参考论文
│   ├── Comparison-Based_System-Level_Fault_Diagnosis_A_Neural_Network_Approach.pdf
│   └── Comparison-Based_System-Level_Fault_Diagnosis_A_Neural_Network_Approach.txt
├── datasets/            # 保存的数据集（运行时生成）
├── main.py              # 项目入口文件
├── requirements.txt     # 项目依赖
└── .gitignore           # Git 忽略配置
```

## 核心概念

### PMC 模型

| 测试者 | 被测者 | 结果   |
| ------ | ------ | ------ |
| 正常   | 正常   | 0      |
| 正常   | 故障   | 1      |
| 故障   | 任意   | 不可靠 |

### 超立方体

- N 维有 2^N 个节点，每个节点 N 个邻居
- 节点二进制编号只差一位即为邻居
- 例：节点 5 (101) 的邻居是 4 (100), 7 (111), 1 (001)

### 可视化

| 元素        | 含义       |
| ----------- | ---------- |
| 🔴 红色节点 | 故障       |
| 🟢 绿色节点 | 正常       |
| 红色虚线    | 测试结果=1 |
| 绿色实线    | 测试结果=0 |
| 灰色点线    | 不可靠     |

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
# models/gnn.py
class GNN(BaseModel):
    def train(self, train_data, val_data, epochs): ...
    def predict(self, x): ...
```

## License

MIT
