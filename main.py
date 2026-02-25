"""
Input:
  - 标准库: argparse, random
  - 第三方: numpy, torch
  - 本地: topologies.Hypercube, models (BPNN, GAT), data (generate_data, save_dataset, load_dataset), evaluation.evaluate, utils (setup_logger, visualize_syndrome)
Output:
  - main: 主函数，执行故障诊断的完整流程
  - parse_args: 解析命令行参数
  - train_and_evaluate: 训练并评估单个模型
Position: 项目的入口文件，协调整个故障诊断流程，支持 BPNN/GAT/对比模式

⚠️ 提醒：修改本文件后务必：
   1. 更新上方 Input/Output/Position 说明
   2. 更新所属文件夹的 README.md（如果影响模块结构）
   3. 更新根目录 README.md（如果影响核心架构）
"""

import argparse
import random
import numpy as np
import torch
from topologies import Hypercube
from models import BPNN, GAT
from data import generate_data, save_dataset, load_dataset
from evaluation import evaluate
from utils import setup_logger, visualize_syndrome


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数
    
    Returns:
        包含命令行参数的 Namespace 对象
    """
    parser = argparse.ArgumentParser(description="Neural Network Fault Diagnosis under PMC Model")
    parser.add_argument("-d", "--dimension", type=int, default=4,
                        help="超立方体维度，节点数 = 2^d (default: 4)")
    parser.add_argument("-f", "--faults", type=str, default="0.25",
                        help="故障数：整数表示具体数目，小数表示故障率 (default: 0.25)")
    parser.add_argument("-n", "--n_samples", type=int, default=5000,
                        help="总样本数 (default: 5000)")
    parser.add_argument("-e", "--epochs", type=int, default=200,
                        help="训练轮数 (default: 200)")
    parser.add_argument("-m", "--model", type=str, default="bpnn",
                        choices=["bpnn", "gat", "both"],
                        help="模型选择：bpnn / gat / both (default: bpnn)")
    parser.add_argument("--load", type=str, default=None,
                        help="加载已有数据集（数据集名称）")
    parser.add_argument("--save", type=str, default=None,
                        help="保存数据集（数据集名称）")
    parser.add_argument("--visualize", type=str, default=None,
                        help="可视化单个 syndrome 文件路径")
    return parser.parse_args()


def train_and_evaluate(model_name: str, model, train_data: tuple,
                       val_data: tuple, test_data: tuple,
                       epochs: int, logger) -> dict:
    """
    训练并评估单个模型

    Args:
        model_name: 模型名称（用于日志输出）
        model: 模型实例（BPNN 或 GAT）
        train_data: 训练数据 (X, Y)
        val_data: 验证数据 (X, Y)
        test_data: 测试数据 (X, Y)
        epochs: 训练轮数
        logger: 日志对象

    Returns:
        评估结果字典
    """
    logger.info(f"--- Training {model_name} ---")
    model.train(train_data, val_data, epochs)

    logger.info(f"Evaluating {model_name} on test set...")
    results = evaluate(model, test_data)

    logger.info(f"=== {model_name} Results ===")
    logger.info(f"Accuracy:  {results['accuracy']*100:.2f}%")
    logger.info(f"Precision: {results['precision']*100:.2f}%")
    logger.info(f"Recall:    {results['recall']*100:.2f}%")
    logger.info(f"F1-Score:  {results['f1']*100:.2f}%")

    return results


def main() -> None:
    """
    主函数：执行故障诊断的完整流程

    流程图：
        参数解析
            ↓
        [可视化模式?] → 是 → 可视化 syndrome → 结束
            ↓ 否
        创建超立方体拓扑
            ↓
        [加载数据?] → 是 → 从磁盘加载
            ↓ 否
        生成新数据 → [保存?] → 保存到磁盘
            ↓
        [模型选择]
            ├─ bpnn → 训练 BPNN → 评估
            ├─ gat  → 训练 GAT  → 评估
            └─ both → 训练两者  → 对比输出
    """
    # ==================== 初始化 ====================
    args = parse_args()
    logger = setup_logger()

    # 设置随机种子以确保可重复性
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ==================== 可视化模式 ====================
    if args.visualize:
        logger.info(f"Visualizing: {args.visualize}")
        output = visualize_syndrome(args.visualize, args.dimension)
        logger.info(f"Saved: {output}")
        return

    # ==================== 解析参数 ====================
    dimension = args.dimension
    n_nodes = 2 ** dimension

    fault_val = float(args.faults)
    if fault_val < 1:
        max_faults = max(1, int(n_nodes * fault_val))
    else:
        max_faults = int(fault_val)

    logger.info(f"=== Fault Diagnosis for {dimension}-D Hypercube (model: {args.model}) ===")

    # ==================== 创建拓扑 ====================
    topo = Hypercube(dimension)

    # ==================== 数据准备 ====================
    if args.load:
        logger.info(f"Loading dataset: {args.load}")
        train_data, val_data, test_data, metadata = load_dataset(args.load)
        n_nodes = metadata.get("n_nodes", n_nodes)
        max_faults = metadata.get("max_faults", max_faults)
        logger.info(f"Loaded - Train: {len(train_data[0])}, Val: {len(val_data[0])}, Test: {len(test_data[0])}")
    else:
        logger.info(f"Generating data (train/val/test = 80/10/10)...")
        train_data, val_data, test_data = generate_data(topo, max_faults, args.n_samples)
        logger.info(f"Train: {len(train_data[0])}, Val: {len(val_data[0])}, Test: {len(test_data[0])}")

        if args.save:
            metadata = {"n_nodes": n_nodes, "max_faults": max_faults, "dimension": dimension}
            path = save_dataset(train_data, val_data, test_data, args.save, metadata)
            logger.info(f"Dataset saved: {path}")

    logger.info(f"Nodes: {n_nodes}, Max faults: {max_faults}")

    # ==================== 模型训练与评估 ====================
    if args.model == "bpnn":
        model = BPNN(input_size=topo.syndrome_size, output_size=n_nodes)
        train_and_evaluate("BPNN", model, train_data, val_data, test_data, args.epochs, logger)

    elif args.model == "gat":
        model = GAT(topo=topo)
        train_and_evaluate("GAT", model, train_data, val_data, test_data, args.epochs, logger)

    elif args.model == "both":
        # 依次训练 BPNN 和 GAT，输出对比结果
        bpnn_model = BPNN(input_size=topo.syndrome_size, output_size=n_nodes)
        bpnn_results = train_and_evaluate("BPNN", bpnn_model, train_data, val_data, test_data, args.epochs, logger)

        gat_model = GAT(topo=topo)
        gat_results = train_and_evaluate("GAT", gat_model, train_data, val_data, test_data, args.epochs, logger)

        # 对比输出
        logger.info("=== BPNN vs GAT Comparison ===")
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            b = bpnn_results[metric] * 100
            g = gat_results[metric] * 100
            diff = g - b
            sign = "+" if diff >= 0 else ""
            logger.info(f"{metric.capitalize():>10}: BPNN={b:.2f}%  GAT={g:.2f}%  ({sign}{diff:.2f}%)")

        # === Phase 3 对比可视化（暂时注释）===
        # 以下代码用于生成 BPNN vs GAT 的对比图表。
        # 当 Phase 3 正式开始时，取消注释即可使用。
        # 位置标记：main.py → main() → both 分支末尾
        #
        # import matplotlib.pyplot as plt
        #
        # # 1. 指标柱状图对比
        # metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        # bpnn_vals = [bpnn_results[k]*100 for k in ['accuracy','precision','recall','f1']]
        # gat_vals = [gat_results[k]*100 for k in ['accuracy','precision','recall','f1']]
        # x = np.arange(len(metrics_names))
        # width = 0.35
        # fig, ax = plt.subplots(figsize=(10, 6))
        # ax.bar(x - width/2, bpnn_vals, width, label='BPNN')
        # ax.bar(x + width/2, gat_vals, width, label='GAT')
        # ax.set_ylabel('Score (%)')
        # ax.set_title(f'BPNN vs GAT - {dimension}D Hypercube')
        # ax.set_xticks(x)
        # ax.set_xticklabels(metrics_names)
        # ax.legend()
        # ax.set_ylim(0, 105)
        # plt.tight_layout()
        # plt.savefig(f'comparison_{dimension}d.png', dpi=150)
        # logger.info(f"Comparison chart saved: comparison_{dimension}d.png")
        # plt.close()


if __name__ == "__main__":
    main()
