# CLAUDE.md

本文件为 Claude Code 提供项目上下文和编码规范。

## 毕设背景

- **作者身份**：南京邮电大学本科生，正在完成毕业设计
- **论文题目**：《基于图注意力机制的互连网络故障诊断研究》
- **研究场景**：超立方体（Hypercube）互连网络拓扑下的**永久性**故障诊断（最理想、最简单的场景）
- **基线方案**：借鉴 Elhadef 2012 思路，在 PMC 模型下实现的 BPNN 基线（诊断准确率已达 99%+）
- **改进方案**：使用 GAT（Graph Attention Network，图注意力网络）替换 BPNN，提升故障诊断性能
- **最终交付物**：通过多维度数据与对比分析，论证 GAT 优于 BPNN
- **关键参考文献**：
  - Elhadef & Nayak, 2012 — *Comparison-Based System-Level Fault Diagnosis: A Neural Network Approach*（BPNN 思路来源，注意：原文使用 SCM/GCM 比较模型，本项目在 PMC 模型下借鉴其思路）
  - 孙雪丽 — GAT 相关论文（改进方案核心参考）
- **技术约束**：模型必须基于 GAT，突出 GAT 优势并缓解其不足，**不涉及强化学习**
- **论文引用约束**：孙雪丽的 FaultGAT 论文正在发表中，**论文正文中不得出现 FaultGAT 名称及孙雪丽的引用**。CLAUDE.md 和代码注释中可保留作为内部技术参考，但所有面向论文的输出必须避免提及。

### BPNN 基线与 Elhadef 2012 论文的关系（Phase 1 核查结论）

在进入 Phase 2 之前，对 BPNN 基线代码与 Elhadef & Nayak (2012) 论文进行了逐项核查，确认以下事实：

1. **诊断模型不同**：Elhadef 2012 使用的是比较模型（SCM/GCM），本项目使用的是 PMC 测试模型。两者的 syndrome 语义完全不同，不存在"在 PMC 下复现 Elhadef 方法"的说法。
2. **BPNN 基线定位**：借鉴 Elhadef 2012 提出的"用 BPNN 做系统级故障诊断"这一思路，在 PMC 模型下独立实现 BPNN 基线。不是对原论文的复现。
3. **网络结构与超参数差异**：论文中的隐藏层公式（$2n+1$ 等）针对 SCM/GCM syndrome 设计，PMC syndrome 维度不同（$n \times d$ vs $\binom{n}{2}$），网络结构需要重新调整。当前代码使用 Adam、BatchNorm、ReduceLROnPlateau 等现代技巧，使基线更强，有利于 GAT 对比实验的说服力。
4. **阈值方向**：论文中 output $\leq 0.5$ 表示故障（标签编码不同），代码中 1 = 故障、$>0.5$ 判定为故障，逻辑自洽，无问题。
5. **论文写作建议**：引用 Elhadef 2012 时应说明"借鉴其思路"，不要说"复现了 Elhadef 的方法"，避免答辩时被质疑复现准确性。

> **结论**：当前 BPNN 基线代码作为 Phase 2 GAT 对比实验的基线是合格的，无需修改。

## 项目阶段

| 阶段 | 内容 | 状态 |
| --- | --- | --- |
| Phase 1 | BPNN 基线实现 | ✅ 已完成 |
| Phase 2 | GAT 模型实现，替换 BPNN | ✅ 基本完成（剩余：requirements.txt、消融实验配置） |
| Phase 3 | BPNN vs GAT 多维度对比实验与分析 | 🔧 进行中（代码准备已完成，待跑实验） |
| Phase 4 | 论文撰写 | 🔧 同步进行 |

> **注意**：当前代码库克隆自指导老师提供的 GitHub 项目并做了改进。BPNN 基线诊断准确率已达 99%+，GAT 模型已实现并验证通过（4D 99%+，6D 94% 大幅领先 BPNN 77%）。

## ★★ 隐私文件管理规则

项目中有部分文件仅本地使用，不推送到 GitHub。这些文件通过 `.git/info/exclude` 排除（而非 `.gitignore`，因为 `.gitignore` 会被推送到远程）。

**当前已排除的文件**（见 `.git/info/exclude`）：
- `毕设提交文件/`、`汇报消息/` — 毕设提交材料和与导师的沟通记录
- `FAULTGAT_中文翻译.md`、`FaultGAT论文精读笔记.md` — 内部论文参考
- `PROGRESS.md`、`各阶段子任务计划表.md` — 项目进度跟踪（含隐私）
- `代码解释和终端命令.md`、`代码涉及公式.md`、`第三次交流信息存储.md`、`论文格式要求.md` — 个人笔记

**规则**：如果之后新增了不应推送到 GitHub 的本地文件（如个人笔记、导师沟通记录等），应将其路径添加到 `.git/info/exclude` 而非 `.gitignore`。

## ★★ 关键规则：模型与算法验证

在进行任何实验设计、代码实现或深入讨论之前，**必须**先与用户确认并对齐以下内容：

1. 所涉及的模型或算法的**精确定义**、**结构**和**行为**
2. 不同资料来源可能对同一模型有不同描述，**不要假设自己的理解是正确的**
3. 始终先与用户验证，避免下游错误累积

## 项目概述

基于神经网络的系统级故障诊断项目。

- **当前（Phase 1）**：使用 BPNN 实现 PMC 模型下的故障诊断，输入测试综合征（syndrome），输出故障节点集合
- **目标（Phase 2）**：使用 GAT 替换 BPNN，利用图结构信息提升诊断准确率

## 技术栈

- Python >= 3.10
- PyTorch（BPNN 模型 / GAT 模型）
- PyTorch Geometric（GAT 实现，Phase 2 引入）
- NumPy（数据处理）
- Matplotlib / NetworkX（可视化）

## 项目结构

```
GAT-FaultDiagnosis/
├── topologies/          # 网络拓扑 + PMC syndrome 生成
│   ├── base.py          # 拓扑抽象基类 (BaseTopology)
│   └── hypercube.py     # N维超立方体拓扑 (Hypercube)
├── models/              # 诊断模型
│   ├── base.py          # 模型抽象基类 (BaseModel)
│   ├── bpnn.py          # BPNN 模型（Phase 1 基线）
│   └── gat.py           # GAT 模型（Phase 2）
├── data/                # 数据生成与管理
│   ├── generator.py     # 训练/验证/测试数据集生成
│   ├── dataset.py       # 数据集持久化（保存/加载）
│   └── converter.py     # syndrome → PyG Data 格式转换
├── evaluation/          # 评估模块
│   └── metrics.py       # Accuracy / Precision / Recall / F1 计算
├── utils/               # 工具
│   ├── logger.py        # 日志配置
│   └── visualizer.py    # Syndrome 可视化
├── papers/              # 参考论文（PDF + TXT）
├── datasets/            # 运行时生成的数据集（不推送）
├── TrainingRecords/     # 实验记录 JSON（不推送，按时间戳子文件夹组织）
└── main.py              # 项目入口（支持 BPNN/GAT/对比模式）
```

## 常用命令

```bash
# BPNN 训练（默认）
python main.py -d 4 -f 0.25 -n 1000 -e 100

# GAT 训练
python main.py -d 4 -f 0.25 -n 1000 -e 100 -m gat

# BPNN vs GAT 对比
python main.py -d 4 -f 0.25 -n 1000 -e 100 -m both

# 自定义参数
python main.py -d 5 -f 6 -n 5000

# 数据集保存/加载
python main.py -n 2000 --save my_data
python main.py --load my_data -m gat

# 可视化单个 syndrome
python main.py --visualize datasets/my_data/xxx/train/1.npz
```

## 核心概念

- **PMC 模型**：正常节点测试正常节点=0，正常测试故障=1，故障节点测试结果不可靠
- **超立方体**：N 维有 $2^N$ 个节点，节点二进制编号只差一位即为邻居
- **Syndrome**：所有测试结果的集合，是诊断模型的输入
- **GAT（Graph Attention Network）**：基于注意力机制的图神经网络，能够为不同邻居节点分配不同的注意力权重，天然适合处理图结构数据（如超立方体拓扑）

## GAT 技术细节（Phase 2）

> 核心参考：孙雪丽 FaultGAT 论文。注意：FaultGAT 处理间歇性故障（多轮测试），本项目处理永久性故障（单轮测试），特征设计有所不同。

### 永久性故障适配

FaultGAT 的节点特征基于多轮测试的比例统计 $P_{u,v} = \frac{1}{R}\sum_{r=1}^{R} T_{u,v}^{(r)}$。本项目为永久性故障（单轮测试），节点特征设计为：

- 节点 $u$ 的特征向量：$f_u = [\sigma(u, v_1), \ldots, \sigma(u, v_d), \sigma(v_1, u), \ldots, \sigma(v_d, u)]$
- 前半部分：$u$ 测试各邻居的结果；后半部分：各邻居测试 $u$ 的结果
- 特征维度 = $2d$（$d$ 为超立方体维度）

### GAT 核心公式

**注意力系数计算**（每个头 $k$，层 $l$）：

$$e_{uv}^{(k)} = \text{LeakyReLU}(\mathbf{a}_k^T [\mathbf{W}_k^{(l)} \mathbf{h}_u^{(l-1)} \| \mathbf{W}_k^{(l)} \mathbf{h}_v^{(l-1)}])$$

**注意力权重归一化**（Softmax）：

$$\alpha_{uv}^{(k)} = \frac{\exp(e_{uv}^{(k)})}{\sum_{m \in \mathcal{N}(u)} \exp(e_{um}^{(k)})}$$

**邻居信息聚合**：

$$\mathbf{h}_u^{(k,l)} = \sum_{v \in \mathcal{N}(u)} \alpha_{uv}^{(k)} \mathbf{W}_k^{(l)} \mathbf{h}_v^{(l-1)}$$

**多头拼接**（中间层）：

$$\mathbf{H}^{(l)} = \|_{k=1}^{K} \mathbf{h}^{(k,l)}$$

### 模型架构

```
输入特征 F ∈ R^{n_nodes × 2d}
    ↓
特征变换层: Linear(2d → hidden*heads) → LayerNorm → LeakyReLU
    ↓
GAT Layer 1（中间层）: GATConv(K=8 heads, hidden=64, concat=True) → BatchNorm → ReLU → Dropout(0.3)
    ↓
GAT Layer 2（最终层）: GATConv(K=8 heads, out=2, concat=False) → 输出 logits
    ↓
Softmax → 节点分类
```

### 参数设置

| 参数 | 值 | 来源 |
|------|-----|------|
| GAT 层数 | 2 | FaultGAT |
| 注意力头数 $K$ | 8 | FaultGAT |
| 隐藏维度 | 64 | FaultGAT |
| Dropout | 0.3 | FaultGAT |
| 优化器 | Adam (weight_decay=5e-4) | FaultGAT |
| 初始学习率 | 0.002 | FaultGAT |
| 损失函数 | CrossEntropyLoss（Focal Loss 在永久性故障场景下效果不佳，已弃用） | 自定义 |
| 特征变换 | LayerNorm + LeakyReLU | FaultGAT |
| 中间层归一化 | BatchNorm | FaultGAT |
| Early Stopping | 基于 val F1，patience=50 | FaultGAT（patience 调大） |
| 梯度裁剪 | max_norm=1.0 | FaultGAT |
| LR 调度 | ReduceLROnPlateau (factor=0.5, patience=15) | 自定义 |

### Focal Loss 公式

$$FL(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

- $\alpha$：类别权重，控制正负样本的重要性
- $\gamma$：聚焦参数，$\gamma$ 越大越关注难分类样本

## 对话风格与格式要求

以下规则适用于 Claude Code 与用户的交互：

- **默认使用中文**输出，使用中文标点（，。（）：；）
- 使用 Markdown 或 LaTeX 显示数学公式，所有数学内容必须用 `$` 或 `$$` 包裹
- 语气正式且友好
- 回答简洁，除非用户要求详细展开
- 介绍技术概念时，先给定义，再给具体易懂的解释
- 用句号或逗号替代破折号（——）
- **仅限毕设相关问题**，与毕设无关的问题应礼貌拒绝并提醒用户聚焦
- Claude 应具备独立判断能力，如果用户陈述有误，应指出并提供准确信息，而不是附和

## 编码规范

### 文档分形结构（三层）

本项目采用三层文档结构，修改代码时必须同步维护：

1. **根目录 README.md** — 项目整体架构、功能模块、技术栈、快速开始
2. **文件夹 README.md** — 该文件夹的职责、文件清单表格、依赖关系
3. **文件头部注释** — 每个 .py 文件的模块级 docstring

### Python 文件头部 docstring（必须）

每个 .py 文件开头必须包含：

```python
"""
Input: [依赖 - 标准库 | 第三方库 | 本地模块 | 配置文件 | 环境变量]
Output: [对外提供的函数/类/常量]
Position: [在系统中的定位，一句话]

⚠️ 提醒：修改本文件后务必：
   1. 更新上方 Input/Output/Position 说明
   2. 更新所属文件夹的 README.md（如果影响模块结构）
   3. 更新根目录 README.md（如果影响核心架构）
"""
```

### `__init__.py` 要求

- 必须声明 `__all__` 列表
- 包含包职责说明、包结构、对外接口
- 包内文件变化时必须同步更新

### 函数和类 docstring

```python
def function_name(param1: str, param2: int = 0) -> dict:
    """
    [必填] 一句话说明功能

    Args:
        param1: 参数说明
        param2: 参数说明，默认值为0

    Returns:
        [有返回值时必填] 返回值说明

    Raises:
        [显式抛出异常时必填] 异常说明
    """
```

### 导入规范（PEP 8 顺序）

```python
# 标准库
import os

# 第三方库
import numpy as np

# 本地模块
from utils.helper import format_data
```

各部分之间空一行。优先使用绝对导入。

### 类型注解

所有公开函数必须添加参数和返回值的类型注解。

## 变更检查清单

### 新增文件时

- [ ] 添加模块级 docstring（Input/Output/Position）
- [ ] 更新 `__init__.py` 和 `__all__`
- [ ] 更新所属文件夹 README.md
- [ ] 如新功能模块，更新根目录 README.md
- [ ] 如引入新依赖，更新 requirements.txt

### 修改文件时

- [ ] 检查 Input/Output/Position 是否需要更新
- [ ] 如导出接口变化，更新 `__init__.py`
- [ ] 修改函数签名时同步更新 docstring

### 删除文件时

- [ ] 从 `__init__.py` 删除相关导入和 `__all__` 条目
- [ ] 从文件夹 README.md 删除文件信息
- [ ] 全局搜索确认无其他文件依赖此文件
- [ ] 如核心模块，更新根目录 README.md

## ★★ 关键规则：同步更新子任务计划表

`各阶段子任务计划表.md` 是本项目的核心进度文档，**不仅供 Claude Code 参考，也是用户撰写毕业论文正文时的重要依据**。

因此，在以下情况发生时，**必须**同步更新 `各阶段子任务计划表.md`：

1. 与用户讨论后，编码计划、实现方案或任务拆分发生变化
2. 某个子任务完成（将 `- [ ]` 改为 `- [x]`）
3. 新增了原计划中未列出的子任务
4. 发现某个子任务不再需要（标注删除原因）
5. 新建文件清单发生变化（新增或移除待创建的文件）

> **注意**：这一要求独立于 Claude Code 内部的 TodoWrite 工具。TodoWrite 用于会话内的临时任务跟踪，而 `各阶段子任务计划表.md` 是持久化的、用户可见的项目进度记录。两者都需要维护，不可互相替代。

## Phase 3 实验设计（已与导师确认）

> 导师指导：实验范围先暂定，结果出来后根据控制变量法分析调整。需设计消融实验表明指标的合理性、有效性和正确性。

### 控制变量法：四组实验

| 实验 | 变量 | 固定条件 | 目的 |
| --- | --- | --- | --- |
| 维度扩展性 | $d=4,5,6,7$（+8） | 故障率 25%，样本 5000 | GAT 随维度增大优势放大 |
| 故障率鲁棒性 | 10%~50%（步长 10%） | $d=6$，样本 5000 | GAT 高故障率下更稳健 |
| 样本效率 | 500, 1000, 2000, 5000, 10000 | $d=6$，故障率 25% | GAT 数据利用效率更高 |
| 消融实验 | 每次移除一个组件 | $d=6$，故障率 25%，样本 5000 | 验证各组件有效性 |

### 消融实验项

采用"每次移除一个"策略（完整模型 + 4 个变体 = 5 次实验）：

1. 多头注意力（8头）→ 单头注意力（1头）
2. 双向特征（$2d$）→ 单向特征（$d$）
3. 2 层 GAT → 1 层 GAT
4. 去掉正则化（BatchNorm + Dropout）

### 评价指标

- 分类性能：Accuracy、Precision、Recall、F1-Score
- 效率指标：推理时延（单个 syndrome，多次取平均）
- 辅助指标：训练耗时、模型参数量、训练收敛曲线

## 扩展约定

### 添加新拓扑

继承 `BaseTopology`，实现 `n_nodes`、`syndrome_size`、`get_neighbors`、`generate_PMC_syndrome`。

### 添加新模型

继承 `BaseModel`，实现 `train`、`predict`。
