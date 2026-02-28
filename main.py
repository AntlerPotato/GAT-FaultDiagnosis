"""
Input:
  - 标准库: argparse, random, time, json, os
  - 第三方: numpy, torch
  - 本地: topologies.Hypercube, models (BPNN, GAT), data (generate_data, save_dataset, load_dataset), evaluation.evaluate, utils (setup_logger, visualize_syndrome), utils.attention_viz.plot_attention_boxplot
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
import json
import os
import random
import time
import numpy as np
import torch
from topologies import Hypercube
from models import BPNN, GAT
from data import generate_data, save_dataset, load_dataset
from evaluation import evaluate
from utils import setup_logger, visualize_syndrome
from utils.attention_viz import plot_attention_boxplot


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
    # 消融实验参数（仅影响 GAT 模型）
    parser.add_argument("--n_heads", type=int, default=8,
                        help="GAT 注意力头数 (default: 8)")
    parser.add_argument("--n_layers", type=int, default=2,
                        help="GAT 层数 (default: 2)")
    parser.add_argument("--feature_mode", type=str, default="bidirectional",
                        choices=["bidirectional", "unidirectional"],
                        help="特征模式：bidirectional / unidirectional (default: bidirectional)")
    parser.add_argument("--no_regularization", action="store_true",
                        help="关闭 GAT 的 BatchNorm + Dropout")
    parser.add_argument("--attention", action="store_true",
                        help="训练 GAT 后分析注意力权重分布（需配合 --load 或 -m gat/both）")
    return parser.parse_args()


def train_and_evaluate(model_name: str, model, train_data: tuple,
                       val_data: tuple, test_data: tuple,
                       epochs: int, logger) -> dict:
    """
    训练并评估单个模型，记录训练耗时、推理时延、参数量和 loss 曲线

    Args:
        model_name: 模型名称（用于日志输出）
        model: 模型实例（BPNN 或 GAT）
        train_data: 训练数据 (X, Y)
        val_data: 验证数据 (X, Y)
        test_data: 测试数据 (X, Y)
        epochs: 训练轮数
        logger: 日志对象

    Returns:
        包含评估指标、训练耗时、推理时延、参数量、loss 曲线的结果字典
    """
    # 模型参数量
    n_params = sum(p.numel() for p in model.network.parameters())
    logger.info(f"--- Training {model_name} (params: {n_params:,}) ---")

    # 训练计时
    start_time = time.time()
    loss_history = model.train(train_data, val_data, epochs)
    train_time = time.time() - start_time
    logger.info(f"{model_name} training time: {train_time:.2f}s")

    # 测试集评估
    logger.info(f"Evaluating {model_name} on test set...")
    results = evaluate(model, test_data)

    # 推理时延：对测试集前 100 个样本逐个推理，取平均
    n_latency_samples = min(100, len(test_data[0]))
    latency_times = []
    for i in range(n_latency_samples):
        t0 = time.perf_counter()
        model.predict(test_data[0][i])
        t1 = time.perf_counter()
        latency_times.append(t1 - t0)
    avg_latency_ms = (sum(latency_times) / len(latency_times)) * 1000

    logger.info(f"=== {model_name} Results ===")
    logger.info(f"Accuracy:  {results['accuracy']*100:.2f}%")
    logger.info(f"Precision: {results['precision']*100:.2f}%")
    logger.info(f"Recall:    {results['recall']*100:.2f}%")
    logger.info(f"F1-Score:  {results['f1']*100:.2f}%")
    logger.info(f"Inference latency: {avg_latency_ms:.3f} ms/sample (avg of {n_latency_samples})")

    # 将额外信息附加到结果字典
    results["n_params"] = n_params
    results["train_time_s"] = round(train_time, 2)
    results["avg_latency_ms"] = round(avg_latency_ms, 3)
    results["loss_history"] = loss_history

    return results


def save_experiment_record(record: dict, batch_id: str, logger) -> str:
    """
    将实验记录保存为 JSON 文件到 TrainingRecords/{batch_id}/ 子目录

    同一次运行的所有模型记录共享同一个 batch_id 子文件夹。
    文件名格式：{model}_{dimension}d_f{faults}_n{n_samples}.json

    Args:
        record: 实验记录字典
        batch_id: 批次标识（时间戳），作为子文件夹名
        logger: 日志对象

    Returns:
        保存的文件路径
    """
    batch_dir = os.path.join("TrainingRecords", "raw_data", batch_id)
    os.makedirs(batch_dir, exist_ok=True)
    model_name = record.get("model", "unknown")
    dimension = record.get("dimension", 0)
    faults = record.get("faults", "")
    n_samples = record.get("n_samples", 0)
    filename = f"{model_name}_{dimension}d_f{faults}_n{n_samples}.json"
    filepath = os.path.join(batch_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)

    logger.info(f"Experiment record saved: {filepath}")
    return filepath


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
    # 构建 GAT 变体名称（用于文件命名和 JSON 记录）
    gat_variant = "GAT"
    if args.n_heads != 8:
        gat_variant += f"-{args.n_heads}head"
    if args.n_layers != 2:
        gat_variant += f"-{args.n_layers}layer"
    if args.feature_mode == "unidirectional":
        gat_variant += "-unidir"
    if args.no_regularization:
        gat_variant += "-noreg"

    # GAT 消融配置（记录到 JSON 中，方便后续分析）
    gat_config = {
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "feature_mode": args.feature_mode,
        "no_regularization": args.no_regularization,
    }

    # 实验配置（用于 JSON 记录）
    batch_id = time.strftime("%Y%m%d_%H%M%S")
    experiment_config = {
        "dimension": dimension,
        "n_nodes": n_nodes,
        "max_faults": max_faults,
        "faults": args.faults,
        "n_samples": args.n_samples,
        "epochs": args.epochs,
        "seed": seed,
    }

    def _create_gat_model() -> GAT:
        """根据命令行参数创建 GAT 模型实例"""
        return GAT(
            topo=topo,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            no_regularization=args.no_regularization,
            feature_mode=args.feature_mode,
        )

    if args.model == "bpnn":
        model = BPNN(input_size=topo.syndrome_size, output_size=n_nodes)
        results = train_and_evaluate("BPNN", model, train_data, val_data, test_data, args.epochs, logger)
        record = {**experiment_config, "model": "BPNN", "results": {
            k: v for k, v in results.items() if k != "loss_history"
        }, "loss_history": results["loss_history"]}
        save_experiment_record(record, batch_id, logger)

    elif args.model == "gat":
        model = _create_gat_model()
        results = train_and_evaluate(gat_variant, model, train_data, val_data, test_data, args.epochs, logger)
        record = {**experiment_config, "model": gat_variant, "gat_config": gat_config,
                  "results": {
            k: v for k, v in results.items() if k != "loss_history"
        }, "loss_history": results["loss_history"]}
        save_experiment_record(record, batch_id, logger)

        # 注意力权重分析
        if args.attention:
            logger.info("=== Attention Weight Analysis ===")
            attn_data = model.get_attention_weights(test_data)
            for key, arr in attn_data["by_type"].items():
                if len(arr) > 0:
                    logger.info(f"  {key}: n={len(arr)}, mean={arr.mean():.4f}, std={arr.std():.4f}")
            fig_path = plot_attention_boxplot(
                attn_data, save_dir="figures",
                dimension=dimension, fault_rate=float(args.faults)
            )
            logger.info(f"Attention boxplot saved: {fig_path}")

    elif args.model == "both":
        # 依次训练 BPNN 和 GAT，输出对比结果
        bpnn_model = BPNN(input_size=topo.syndrome_size, output_size=n_nodes)
        bpnn_results = train_and_evaluate("BPNN", bpnn_model, train_data, val_data, test_data, args.epochs, logger)

        gat_model = _create_gat_model()
        gat_results = train_and_evaluate(gat_variant, gat_model, train_data, val_data, test_data, args.epochs, logger)

        # 对比输出
        logger.info(f"=== BPNN vs {gat_variant} Comparison ===")
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            b = bpnn_results[metric] * 100
            g = gat_results[metric] * 100
            diff = g - b
            sign = "+" if diff >= 0 else ""
            logger.info(f"{metric.capitalize():>10}: BPNN={b:.2f}%  {gat_variant}={g:.2f}%  ({sign}{diff:.2f}%)")

        # 保存两个模型的实验记录
        bpnn_record = {**experiment_config, "model": "BPNN", "results": {
            k: v for k, v in bpnn_results.items() if k != "loss_history"
        }, "loss_history": bpnn_results["loss_history"]}
        save_experiment_record(bpnn_record, batch_id, logger)

        gat_record = {**experiment_config, "model": gat_variant, "gat_config": gat_config,
                      "results": {
            k: v for k, v in gat_results.items() if k != "loss_history"
        }, "loss_history": gat_results["loss_history"]}
        save_experiment_record(gat_record, batch_id, logger)

        # 注意力权重分析
        if args.attention:
            logger.info("=== Attention Weight Analysis ===")
            attn_data = gat_model.get_attention_weights(test_data)
            for key, arr in attn_data["by_type"].items():
                if len(arr) > 0:
                    logger.info(f"  {key}: n={len(arr)}, mean={arr.mean():.4f}, std={arr.std():.4f}")
            fig_path = plot_attention_boxplot(
                attn_data, save_dir="figures",
                dimension=dimension, fault_rate=float(args.faults)
            )
            logger.info(f"Attention boxplot saved: {fig_path}")


if __name__ == "__main__":
    main()
