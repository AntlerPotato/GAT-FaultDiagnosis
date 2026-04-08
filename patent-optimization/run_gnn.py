import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from topologies import Hypercube
from models import GNN
from data import generate_data
from evaluation import evaluate
from utils import setup_logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dimension", type=int, default=4)
    parser.add_argument("-f", "--faults", type=str, default="0.25")
    parser.add_argument("-n", "--n_samples", type=int, default=1000)
    parser.add_argument("-e", "--epochs", type=int, default=100)
    return parser.parse_args()


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
    logger.info(f"=== GNN Fault Diagnosis for {dimension}-D Hypercube ===")
    topo = Hypercube(dimension)
    logger.info("Generating data (train/val/test = 80/10/10)...")
    train_data, val_data, test_data = generate_data(topo, max_faults, args.n_samples)
    logger.info(f"Train: {len(train_data[0])}, Val: {len(val_data[0])}, Test: {len(test_data[0])}")
    model = GNN(input_size=topo.syndrome_size, output_size=n_nodes)
    logger.info("Training...")
    model.train(train_data, val_data, args.epochs)
    logger.info("Evaluating on test set...")
    results = evaluate(model, test_data)
    logger.info("=== Results ===")
    logger.info(f"Accuracy: {results['accuracy']*100:.2f}%")
    logger.info(f"Precision: {results['precision']*100:.2f}%")
    logger.info(f"Recall: {results['recall']*100:.2f}%")


if __name__ == "__main__":
    main()

