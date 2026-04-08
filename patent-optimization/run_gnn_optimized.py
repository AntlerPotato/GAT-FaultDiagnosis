import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from topologies import Hypercube
from models import GNN
from data import generate_data
from utils import setup_logger

def parse_args():
    parser = argparse.ArgumentParser(description="Feature Optimized GNN for Fault Diagnosis")
    parser.add_argument("-d", "--dimension", type=int, default=4)
    parser.add_argument("-f", "--faults", type=str, default="0.25")
    parser.add_argument("-n", "--n_samples", type=int, default=1000)
    parser.add_argument("-e", "--epochs", type=int, default=50)
    return parser.parse_args()

def extract_features(topo: Hypercube, syndrome: np.ndarray) -> np.ndarray:
    """
    特征提取函数 (与 run_feature_optimized.py 保持一致)
    提取三个特征: Accused, Accusing, Weighted Accused
    """
    n_nodes = topo.n_nodes
    features = np.zeros((n_nodes, 2), dtype=np.float32)
    
    syndrome_idx = 0
    accused_counts = np.zeros(n_nodes)
    accusing_counts = np.zeros(n_nodes)
    
    for u in range(n_nodes):
        neighbors = topo.get_neighbors(u)
        for v in neighbors:
            result = syndrome[syndrome_idx]
            syndrome_idx += 1
            if result == 1:
                accusing_counts[u] += 1
                accused_counts[v] += 1
                
    dim = topo.dim
    norm_accused = accused_counts / dim
    norm_accusing = accusing_counts / dim
    
    # === 加权特征 ===
    weighted_accused_counts = np.zeros(n_nodes)
    syndrome_idx = 0
    for u in range(n_nodes):
        neighbors = topo.get_neighbors(u)
        for v in neighbors:
            result = syndrome[syndrome_idx]
            syndrome_idx += 1
            if result == 1:
                weight_u = max(0, 1.0 - norm_accused[u])
                weighted_accused_counts[v] += weight_u
                
    norm_weighted_accused = weighted_accused_counts / dim

    features[:, 0] = norm_accused
    features[:, 1] = norm_accusing
    
    # 组合成 (N, 3)
    features = np.column_stack((features, norm_weighted_accused))
    
    # 返回展平的特征向量 (N * 3)
    return features.flatten()

def process_dataset(topo: Hypercube, dataset: tuple) -> tuple:
    """将整个数据集的 X 转换为 原始Syndrome + 特征向量"""
    X_raw, Y = dataset
    X_new = []
    
    for i in range(len(X_raw)):
        raw_syndrome = X_raw[i]
        feat = extract_features(topo, raw_syndrome)
        # 拼接: [原始 Syndrome (384维), 统计特征 (192维, N*3)]
        combined = np.concatenate([raw_syndrome, feat])
        X_new.append(combined)
        
    return (np.array(X_new, dtype=np.float32), Y)

def main():
    args = parse_args()
    logger = setup_logger()
    dimension = args.dimension
    
    n_nodes = 2 ** dimension
    fault_val = float(args.faults)
    if fault_val < 1:
        max_faults = max(1, int(n_nodes * fault_val))
    else:
        max_faults = int(fault_val)
        
    logger.info(f"=== Feature Optimized GNN (Hybrid Input) for {dimension}-D Hypercube ===")
    logger.info("Using Hybrid Input: Raw Syndrome + [Accused, Accusing, Weighted Accused]")
    
    topo = Hypercube(dimension)
    
    # ==================== 数据生成与处理 ====================
    logger.info("Generating raw data...")
    raw_train, raw_val, raw_test = generate_data(topo, max_faults, args.n_samples)
    
    logger.info("Extracting features and combining...")
    train_data = process_dataset(topo, raw_train)
    val_data = process_dataset(topo, raw_val)
    test_data = process_dataset(topo, raw_test)
    
    # 计算新的输入维度
    input_size = topo.syndrome_size + (n_nodes * 3)
    
    logger.info(f"Input size increased to {input_size}")
    
    # ==================== 模型训练 ====================
    # GNN 会自动检测 input_size 并启用 extra_features 模式
    model = GNN(input_size=input_size, output_size=n_nodes)
    
    logger.info("Training...")
    model.train(train_data, val_data, args.epochs)
    
    # ==================== 评估 ====================
    logger.info("Evaluating on test set...")
    
    # 手动评估流程以计算完整指标
    X_test, Y_test = test_data
    model.dropout.eval()
    
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test)
        # GNN 的 _forward 返回概率
        probs = model._forward(X_tensor).numpy()
        
    preds = (probs > 0.5).astype(int)
    
    total_samples = len(Y_test)
    graph_correct = 0
    for i in range(total_samples):
        if np.array_equal(preds[i], Y_test[i]):
            graph_correct += 1
            
    Y_flat = Y_test.flatten()
    preds_flat = preds.flatten()
    
    tp = np.sum((preds_flat == 1) & (Y_flat == 1))
    fp = np.sum((preds_flat == 1) & (Y_flat == 0))
    fn = np.sum((preds_flat == 0) & (Y_flat == 1))
    tn = np.sum((preds_flat == 0) & (Y_flat == 0))
    
    node_acc = (tp + tn) / len(Y_flat)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    logger.info("=== Optimized GNN Results ===")
    logger.info(f"Graph-Level Accuracy: {graph_correct / total_samples * 100:.2f}%")
    logger.info(f"Node-Level Accuracy:  {node_acc * 100:.2f}%")
    logger.info(f"Precision:            {precision * 100:.2f}%")
    logger.info(f"Recall:               {recall * 100:.2f}%")

if __name__ == "__main__":
    main()
