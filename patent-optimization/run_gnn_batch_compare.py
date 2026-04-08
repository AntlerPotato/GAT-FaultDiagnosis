import argparse
import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from topologies import Hypercube
from models import GNN
from data import generate_data
from utils import setup_logger
from run_single_sample_vis import extract_features, visualize_comparison


def evaluate_model(model, X, Y, threshold=0.5):
    graph_correct = 0
    preds_list = []
    labels_list = []
    for x, y in zip(X, Y):
        probs = model.predict(x)
        preds = (probs > threshold).astype(int)
        if np.array_equal(preds, y):
            graph_correct += 1
        preds_list.append(preds)
        labels_list.append(y)
    preds_arr = np.array(preds_list)
    labels_arr = np.array(labels_list)
    y_flat = labels_arr.flatten()
    p_flat = preds_arr.flatten()
    tp = np.sum((p_flat == 1) & (y_flat == 1))
    fp = np.sum((p_flat == 1) & (y_flat == 0))
    fn = np.sum((p_flat == 0) & (y_flat == 1))
    tn = np.sum((p_flat == 0) & (y_flat == 0))
    node_acc = (tp + tn) / len(y_flat)
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    graph_acc = graph_correct / len(X) if len(X) > 0 else 0.0
    return {
        "graph_acc": graph_acc,
        "node_acc": node_acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def build_optimized_inputs(topo, X):
    X_opt = []
    for x in X:
        feat = extract_features(topo, x)
        X_opt.append(np.concatenate([x, feat]))
    return np.array(X_opt, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dimension", type=int, default=6)
    parser.add_argument("-f", "--fault-rate", type=float, default=0.25)  # 改为 0.25（永久性故障场景）
    parser.add_argument("-n", "--n-samples", type=int, default=500)
    parser.add_argument("-e", "--epochs", type=int, default=50)
    args = parser.parse_args()

    logger = setup_logger()
    dimension = args.dimension
    fault_rate = args.fault_rate
    epochs = args.epochs
    n_samples = args.n_samples
    n_nodes = 2**dimension

    topo = Hypercube(dimension)
    max_faults = int(n_nodes * fault_rate)

    logger.info(f"=== Batch Comparison (Dim={dimension}, Fault={fault_rate}) ===")
    logger.info(f"Total samples: {n_samples}, Epochs: {epochs}")

    train_data, val_data, test_data = generate_data(topo, max_faults, n_samples)
    X_train, Y_train = train_data
    X_val, Y_val = val_data
    X_test, Y_test = test_data

    logger.info(
        f"Train: {len(X_train)}, Val: {len(X_val)}, Test (iterations): {len(X_test)}"
    )

    logger.info("Training Ordinary GNN...")
    model_ord = GNN(input_size=topo.syndrome_size, output_size=n_nodes)
    model_ord.train(train_data, val_data, epochs)

    logger.info("Evaluating Ordinary GNN on test set...")
    ord_results = evaluate_model(model_ord, X_test, Y_test)

    logger.info("Preparing optimized inputs...")
    X_train_opt = build_optimized_inputs(topo, X_train)
    X_val_opt = build_optimized_inputs(topo, X_val)
    X_test_opt = build_optimized_inputs(topo, X_test)
    input_size_opt = topo.syndrome_size + n_nodes * 3

    logger.info("Training Optimized GNN...")
    model_opt = GNN(input_size=input_size_opt, output_size=n_nodes)
    model_opt.train((X_train_opt, Y_train), (X_val_opt, Y_val), epochs)

    logger.info("Evaluating Optimized GNN on test set...")
    opt_results = evaluate_model(model_opt, X_test_opt, Y_test)

    logger.info("=== Batch Results (50 test samples expected) ===")
    logger.info(
        f"Graph-level Accuracy  | Ord: {ord_results['graph_acc']*100:.2f}%  "
        f"| Opt: {opt_results['graph_acc']*100:.2f}%"
    )
    logger.info(
        f"Node-level Accuracy   | Ord: {ord_results['node_acc']*100:.2f}%  "
        f"| Opt: {opt_results['node_acc']*100:.2f}%"
    )
    logger.info(
        f"Precision             | Ord: {ord_results['precision']*100:.2f}%  "
        f"| Opt: {opt_results['precision']*100:.2f}%"
    )
    logger.info(
        f"Recall                | Ord: {ord_results['recall']*100:.2f}%  "
        f"| Opt: {opt_results['recall']*100:.2f}%"
    )
    logger.info(
        f"F1-score              | Ord: {ord_results['f1']*100:.2f}%  "
        f"| Opt: {opt_results['f1']*100:.2f}%"
    )

    report_name = f"gnn_batch_compare_{dimension}d_{fault_rate}.txt"
    # 保存到 TrainingRecords/raw_data/{时间戳}/ 与主项目目录规范一致
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    batch_id = time.strftime("%Y%m%d_%H%M%S")
    report_dir = os.path.join(project_root, "TrainingRecords", "raw_data", batch_id)
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, report_name)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(
            f"=== GNN Batch Comparison (Dim={dimension}, Fault={fault_rate}) ===\n\n"
        )
        f.write(
            f"Total samples: {n_samples} "
            f"(Train {len(X_train)}, Val {len(X_val)}, Test {len(X_test)})\n"
        )
        f.write(f"Epochs: {epochs}\n\n")
        f.write(f"{'Metric':<20} | {'Ordinary GNN':<15} | {'Optimized GNN':<15}\n")
        f.write("-" * 60 + "\n")
        f.write(
            f"{'Graph Acc':<20} | {ord_results['graph_acc']*100:>7.2f}%"
            f"{'':<5} | {opt_results['graph_acc']*100:>7.2f}%{'':<5}\n"
        )
        f.write(
            f"{'Node Acc':<20} | {ord_results['node_acc']*100:>7.2f}%"
            f"{'':<5} | {opt_results['node_acc']*100:>7.2f}%{'':<5}\n"
        )
        f.write(
            f"{'Precision':<20} | {ord_results['precision']*100:>7.2f}%"
            f"{'':<5} | {opt_results['precision']*100:>7.2f}%{'':<5}\n"
        )
        f.write(
            f"{'Recall':<20} | {ord_results['recall']*100:>7.2f}%"
            f"{'':<5} | {opt_results['recall']*100:>7.2f}%{'':<5}\n"
        )
        f.write(
            f"{'F1-score':<20} | {ord_results['f1']*100:>7.2f}%"
            f"{'':<5} | {opt_results['f1']*100:>7.2f}%{'':<5}\n"
        )

    logger.info(f"Batch comparison report saved to {report_path}")

    if dimension <= 8:
        logger.info("Generating new dataset for visualization...")
        vis_data, _, _ = generate_data(topo, max_faults, 1, split=(1.0, 0, 0))
        vis_syndrome = vis_data[0][0]
        vis_labels = vis_data[1][0]

        ord_probs_vis = model_ord.predict(vis_syndrome)
        ord_preds_vis = (ord_probs_vis > 0.5).astype(int)

        feat_vis = extract_features(topo, vis_syndrome)
        input_opt_vis = np.concatenate([vis_syndrome, feat_vis])
        opt_probs_vis = model_opt.predict(input_opt_vis)
        opt_preds_vis = (opt_probs_vis > 0.5).astype(int)

        visualize_comparison(
            topo,
            vis_syndrome,
            vis_labels,
            ord_preds_vis,
            opt_preds_vis,
            "gnn_batch_sample_comparison.png",
        )
        logger.info("Visualization saved to gnn_batch_sample_comparison.png")
    else:
        logger.info("Skip visualization for high dimension.")


if __name__ == "__main__":
    main()
