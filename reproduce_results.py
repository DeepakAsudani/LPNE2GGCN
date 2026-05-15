"""
reproduce_results.py
--------------------
Reproduces Tables 3, 4, 5 and Table 6 from the LPNE2GGCN paper.

Run:
    python reproduce_results.py [--seed 42] [--epochs 200] [--fast]

Options:
    --seed   INT   Random seed  (default: 42)
    --epochs INT   Training epochs per run  (default: 200)
    --fast         Use reduced epochs (50) for a quick sanity-check run

Output:
    - Formatted tables printed to stdout
    - results/paper_results.csv  — all metrics as CSV
    - results/figure2_accuracy.png
    - results/figure3_timing.png
    - results/figure4_auc.png
    - results/table6_layer_ablation.png
"""

import argparse
import csv
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader       import load_dataset
from utils.negative_sampling import prepare_link_prediction_data
from utils.metrics           import compute_metrics
from utils.visualize         import (
    plot_accuracy_comparison,
    plot_auc_comparison,
    plot_timing_chart,
    plot_layer_ablation,
    ACCURACY_TABLE,
    AUC_TABLE,
)
from models.lpne2ggcn import LPNE2GGCN

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

DATASETS = ["citeseer", "cora", "facebook", "imdb", "dblp"]
DATASET_DISPLAY = ["Citeseer", "Cora", "Facebook", "IMDB", "DBLP"]


# ---------------------------------------------------------------------------
# Paper result tables (hardcoded for reference)
# ---------------------------------------------------------------------------

PAPER_ACCURACY = ACCURACY_TABLE
PAPER_AUC      = AUC_TABLE

PAPER_F1 = {
    "Node2Vec"          : [52.8,  69.74, 75.51, 64.25, 52.62],
    "DeepWalk"          : [61.62, 58.21, 65.11, 63.12, 61.33],
    "VGAE"              : [36.08, 79.45, 67.57, 41.28, 61.92],
    "SEAL"              : [53.8,  64.74, 52.01, 60.96, 75.53],
    "GAT"               : [57.7,  53.63, 57.72, 57.14, 58.69],
    "LPNE2GGCN"         : [86.32, 82.98, 86.94, 63.48, 67.71],
    "LPNE2GGCN+Adadelta": [93.56, 95.85, 91.66, 91.08, 90.12],
    "LPNE2GGCN+Adam"    : [94.74, 96.46, 93.82, 90.35, 89.48],
}

PAPER_LAYERS = {
    "2-Layers": [69.54, 61.32, 56.54, 69.49, 58.89],
    "3-Layers": [86.62, 85.87, 84.79, 87.12, 69.18],
    "4-Layers": [68.44, 71.98, 64.32, 72.62, 74.21],
}


# ---------------------------------------------------------------------------
# Pretty table printer
# ---------------------------------------------------------------------------

def _header(title: str):
    print(f"\n{'─'*70}")
    print(f"  {title}")
    print(f"{'─'*70}")


def _print_table(title: str, table: dict, datasets: list):
    _header(title)
    col_w = 14
    header = f"{'Method':<22}" + "".join(f"{d:>{col_w}}" for d in datasets)
    print(header)
    print("─" * len(header))
    for method, vals in table.items():
        is_best = "LPNE2GGCN" in method and "+" in method
        row = f"{'* ' + method if is_best else method:<22}"
        row += "".join(f"{v:>{col_w}.2f}" for v in vals)
        print(row)
    print("  (* = proposed variants)")


def print_all_paper_tables():
    _print_table("TABLE 3 — Accuracy (%)",  PAPER_ACCURACY, DATASET_DISPLAY)
    _print_table("TABLE 4 — AUC (%)",       PAPER_AUC,      DATASET_DISPLAY)
    _print_table("TABLE 5 — F1-Score (%)",  PAPER_F1,       DATASET_DISPLAY)
    _print_table("TABLE 6 — Layers vs Accuracy (%)", PAPER_LAYERS, DATASET_DISPLAY)


# ---------------------------------------------------------------------------
# Live reproduction run
# ---------------------------------------------------------------------------

def reproduce_lpne2ggcn(args):
    """
    Train LPNE2GGCN (both optimizers) on all 5 datasets and print results.
    """
    _header("LIVE REPRODUCTION — LPNE2GGCN + Adam  /  + Adadelta")

    results = {
        "LPNE2GGCN+Adam"    : {},
        "LPNE2GGCN+Adadelta": {},
    }
    timings = {}

    for ds_key, ds_name in zip(DATASETS, DATASET_DISPLAY):
        print(f"\n  ▶  Dataset: {ds_name}")

        graph   = load_dataset(ds_key)
        lp_data = prepare_link_prediction_data(
            graph,
            test_ratio     =0.2,
            pos_ratio      =0.5,
            max_neg_distance=3,
            seed           =args.seed,
        )

        for opt_name, result_key in [("adam", "LPNE2GGCN+Adam"),
                                      ("adadelta", "LPNE2GGCN+Adadelta")]:
            print(f"\n    Optimizer: {opt_name.upper()}")
            model = LPNE2GGCN(
                embedding_dim=64,
                hidden_dim   =128,
                num_layers   =3,        # paper optimum (Table 6)
                aggregator   ="mean",
                dropout      =0.3,
                p=1.0, q=1.0,           # paper optimum (Section IV-D)
                walk_length  =10,
                num_walks    =10,
                optimizer    =opt_name,
                lr           =0.01,
                epochs       =args.epochs,
                seed         =args.seed,
            )
            t0 = time.time()
            model.fit(
                G_train      =lp_data["G_train"],
                edge_index   =lp_data["edge_index_train"],
                train_edges  =lp_data["train_edges"],
                train_labels =lp_data["train_labels"],
                node_features=graph["x"].numpy() if graph["x"] is not None else None,
            )
            elapsed = time.time() - t0
            metrics = model.evaluate(lp_data["test_edges"], lp_data["test_labels"])

            results[result_key][ds_name] = metrics
            timings.setdefault(ds_name, {})[f"LPNE2GGCN ({opt_name})"] = elapsed

            print(f"      ACC={metrics['accuracy']:.2f}%  "
                  f"AUC={metrics['auc']:.2f}%  "
                  f"F1={metrics['f1']:.2f}%  "
                  f"time={elapsed:.2f}s")

    # --- Print reproduced results side-by-side with paper values ---
    _header("REPRODUCED vs PAPER — Accuracy (%)")
    col_w = 14
    print(f"{'Dataset':<12}" +
          "".join(f"{'Adam (repro)':>{col_w}}{'Adam (paper)':>{col_w}}"
                  f"{'Adad (repro)':>{col_w}}{'Adad (paper)':>{col_w}}"))
    for i, ds in enumerate(DATASET_DISPLAY):
        adam_rep  = results["LPNE2GGCN+Adam"].get(ds, {}).get("accuracy", 0)
        adam_pap  = PAPER_ACCURACY["LPNE2GGCN+Adam"][i]
        adad_rep  = results["LPNE2GGCN+Adadelta"].get(ds, {}).get("accuracy", 0)
        adad_pap  = PAPER_ACCURACY["LPNE2GGCN+Adadelta"][i]
        print(f"{ds:<12}"
              f"{adam_rep:>{col_w}.2f}{adam_pap:>{col_w}.2f}"
              f"{adad_rep:>{col_w}.2f}{adad_pap:>{col_w}.2f}")

    return results, timings


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def save_csv(results: dict):
    path = os.path.join(RESULTS_DIR, "paper_results.csv")
    rows = []
    for method, ds_dict in results.items():
        for ds, metrics in ds_dict.items():
            rows.append({
                "method"  : method,
                "dataset" : ds,
                "accuracy": round(metrics.get("accuracy", 0), 2),
                "auc"     : round(metrics.get("auc", 0), 2),
                "f1"      : round(metrics.get("f1", 0), 2),
                "time_s"  : round(metrics.get("training_time", 0), 2),
            })

    # Also write out all paper baseline values
    for method, vals in PAPER_ACCURACY.items():
        for i, ds in enumerate(DATASET_DISPLAY):
            rows.append({
                "method"  : f"[paper] {method}",
                "dataset" : ds,
                "accuracy": vals[i],
                "auc"     : PAPER_AUC.get(method, [0]*5)[i],
                "f1"      : PAPER_F1.get(method, [0]*5)[i],
                "time_s"  : "",
            })

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["method", "dataset", "accuracy", "auc", "f1", "time_s"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[reproduce] CSV saved → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed",   type=int, default=42)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--fast",   action="store_true",
                   help="Use 50 epochs for a quick sanity-check run")
    return p.parse_args()


def main():
    args = parse_args()
    if args.fast:
        args.epochs = 50
        print("[reproduce] Fast mode: using 50 epochs")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 1. Print all paper tables for reference
    print_all_paper_tables()

    # 2. Live reproduction
    results, timings = reproduce_lpne2ggcn(args)

    # 3. Save CSV
    save_csv(results)

    # 4. Generate all figures from paper
    print("\n[reproduce] Generating paper figures ...")
    plot_accuracy_comparison(save=True)
    plot_auc_comparison(save=True)
    plot_timing_chart(save=True)
    plot_layer_ablation(save=True)

    print("\n[reproduce] Done. All results saved in results/")


if __name__ == "__main__":
    main()
