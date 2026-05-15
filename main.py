"""
main.py
-------
Entry-point for training and evaluating LPNE2GGCN on any dataset.

Usage examples:
  python main.py --dataset cora --optimizer adam
  python main.py --dataset all --optimizer both --plot
  python main.py --dataset facebook --model node2vec
"""

import argparse
import os
import sys
import time
import json
import numpy as np
import torch

# Make sure imports resolve whether run from project root or elsewhere
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader     import load_dataset, get_all_datasets
from utils.negative_sampling import prepare_link_prediction_data
from utils.metrics          import compute_metrics
from models.lpne2ggcn       import LPNE2GGCN
from models.node2vec        import Node2VecEmbedder
from models.graphsage       import GraphSAGE, GATBaseline

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="LPNE2GGCN: Link Prediction via Node Embedding + GraphSAGE"
    )
    # Dataset
    parser.add_argument(
        "--dataset", type=str, default="cora",
        choices=["cora", "citeseer", "facebook", "imdb", "dblp", "all"],
        help="Dataset to use (default: cora)",
    )
    # Model
    parser.add_argument(
        "--model", type=str, default="lpne2ggcn",
        choices=["lpne2ggcn", "node2vec", "deepwalk", "gat", "vgae", "seal"],
        help="Model variant (default: lpne2ggcn)",
    )
    # Optimizer
    parser.add_argument(
        "--optimizer", type=str, default="adam",
        choices=["adam", "adadelta", "both"],
        help="Optimizer(s) to use (default: adam)",
    )
    # Architecture
    parser.add_argument("--epochs",        type=int,   default=200)
    parser.add_argument("--hidden_dim",    type=int,   default=128)
    parser.add_argument("--embedding_dim", type=int,   default=64)
    parser.add_argument("--layers",        type=int,   default=3,
                        help="Number of GraphSAGE layers (paper optimum=3)")
    parser.add_argument("--aggregator",    type=str,   default="mean",
                        choices=["mean", "lstm", "pool"])
    parser.add_argument("--dropout",       type=float, default=0.3)
    parser.add_argument("--lr",            type=float, default=0.01)
    # Node2Vec
    parser.add_argument("--p",           type=float, default=1.0,
                        help="Node2Vec return param (paper optimal=1)")
    parser.add_argument("--q",           type=float, default=1.0,
                        help="Node2Vec in-out param (paper optimal=1)")
    parser.add_argument("--walk_length", type=int,   default=10)
    parser.add_argument("--num_walks",   type=int,   default=10)
    # Misc
    parser.add_argument("--seed",       type=int,  default=42)
    parser.add_argument("--save_model", action="store_true",
                        help="Save trained model checkpoint")
    parser.add_argument("--plot",       action="store_true",
                        help="Generate and save comparison charts")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Single-dataset training loop
# ---------------------------------------------------------------------------

def run_single(graph: dict, args, optimizer_name: str) -> dict:
    """
    Train LPNE2GGCN on one dataset with one optimizer.

    Returns
    -------
    dict with accuracy, f1, auc, training_time
    """
    print(f"\n{'='*60}")
    print(f"  Dataset  : {graph['name']}")
    print(f"  Optimizer: {optimizer_name.upper()}")
    print(f"  Layers   : {args.layers}")
    print(f"{'='*60}")

    # --- Prepare data (Algorithm 1) ---
    lp_data = prepare_link_prediction_data(
        graph, test_ratio=0.2, pos_ratio=0.5,
        max_neg_distance=3, seed=args.seed,
    )

    # --- Build and train model ---
    model = LPNE2GGCN(
        embedding_dim=args.embedding_dim,
        hidden_dim   =args.hidden_dim,
        num_layers   =args.layers,
        aggregator   =args.aggregator,
        dropout      =args.dropout,
        p            =args.p,
        q            =args.q,
        walk_length  =args.walk_length,
        num_walks    =args.num_walks,
        optimizer    =optimizer_name,
        lr           =args.lr,
        epochs       =args.epochs,
        seed         =args.seed,
    )

    model.fit(
        G_train      =lp_data["G_train"],
        edge_index   =lp_data["edge_index_train"],
        train_edges  =lp_data["train_edges"],
        train_labels =lp_data["train_labels"],
        node_features=graph["x"].numpy() if graph["x"] is not None else None,
    )

    # --- Evaluate ---
    metrics = model.evaluate(lp_data["test_edges"], lp_data["test_labels"])
    metrics["training_time"] = model.training_time_

    print(f"\n  Results [{graph['name']} | {optimizer_name.upper()}]")
    print(f"    Accuracy : {metrics['accuracy']:.2f}%")
    print(f"    AUC      : {metrics['auc']:.2f}%")
    print(f"    F1-Score : {metrics['f1']:.2f}%")
    print(f"    Time     : {metrics['training_time']:.2f}s")

    # Save checkpoint
    if args.save_model:
        ckpt_path = os.path.join(
            RESULTS_DIR,
            f"lpne2ggcn_{graph['name'].lower()}_{optimizer_name}.pt",
        )
        torch.save(
            {
                "graphsage_state" : model._graphsage.state_dict(),
                "classifier_state": model._classifier.state_dict(),
                "args"            : vars(args),
                "metrics"         : metrics,
            },
            ckpt_path,
        )
        print(f"  Checkpoint saved → {ckpt_path}")

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Collect datasets
    if args.dataset == "all":
        datasets = get_all_datasets()
    else:
        datasets = [load_dataset(args.dataset)]

    # Collect optimizers
    if args.optimizer == "both":
        optimizers = ["adam", "adadelta"]
    else:
        optimizers = [args.optimizer]

    all_results = {}

    for graph in datasets:
        all_results[graph["name"]] = {}
        for opt in optimizers:
            metrics = run_single(graph, args, opt)
            all_results[graph["name"]][opt] = metrics

    # Save summary
    summary_path = os.path.join(RESULTS_DIR, "run_results.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[main] Results saved → {summary_path}")

    # Optional plots
    if args.plot:
        from utils.visualize import (
            plot_accuracy_comparison,
            plot_auc_comparison,
            plot_timing_chart,
            plot_layer_ablation,
        )
        plot_accuracy_comparison()
        plot_auc_comparison()
        plot_timing_chart()
        plot_layer_ablation()


if __name__ == "__main__":
    main()
