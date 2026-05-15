"""
visualize.py
------------
Reproduces Figures 2, 3 and 4 from the paper:
  - Figure 2 : Accuracy comparison bar chart (all methods × all datasets)
  - Figure 3 : Computational Time Chart
  - Figure 4 : AUC comparison bar chart
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

DATASETS = ["Citeseer", "Cora", "Facebook", "IMDB", "DBLP"]

METHODS = [
    "Node2Vec",
    "DeepWalk",
    "VGAE",
    "SEAL",
    "GAT",
    "LPNE2GGCN",
    "LPNE2GGCN+Adadelta",
    "LPNE2GGCN+Adam",
]

# Paper Table 3 — Accuracy
ACCURACY_TABLE = {
    "Node2Vec"          : [74.52, 87.67, 86.94, 83.31, 78.16],
    "DeepWalk"          : [60.49, 75.43, 85.12, 76.22, 60.31],
    "VGAE"              : [84.34, 89.67, 83.31, 66.82, 87.65],
    "SEAL"              : [67.23, 72.87, 58.83, 71.04, 51.43],
    "GAT"               : [81.89, 68.33, 64.13, 56.19, 58.59],
    "LPNE2GGCN"         : [91.32, 90.14, 89.02, 88.67, 82.51],
    "LPNE2GGCN+Adadelta": [93.56, 95.85, 91.66, 91.08, 90.12],
    "LPNE2GGCN+Adam"    : [94.74, 96.46, 93.82, 90.35, 89.48],
}

# Paper Table 4 — AUC
AUC_TABLE = {
    "Node2Vec"          : [79.43, 70.58, 80.64, 67.50, 53.34],
    "DeepWalk"          : [65.45, 66.12, 56.61, 50.28, 31.17],
    "VGAE"              : [80.12, 79.56, 32.44, 89.04, 81.45],
    "SEAL"              : [69.90, 67.53, 49.30, 49.69, 53.49],
    "GAT"               : [78.42, 51.08, 53.20, 55.51, 68.21],
    "LPNE2GGCN"         : [88.79, 87.51, 87.52, 88.82, 78.65],
    "LPNE2GGCN+Adadelta": [94.56, 93.73, 96.66, 92.08, 91.12],
    "LPNE2GGCN+Adam"    : [95.36, 97.46, 92.28, 89.91, 93.65],
}

# Paper Figure 3 — Computational time (seconds)
TIME_TABLE = {
    "Facebook": {"Node2Vec": 4.32, "DeepWalk": 3.82, "SEAL": 4.31,
                 "GAT": 4.16, "VGAE": 3.67, "LPNE2GGCN": 3.08},
    "Cora"    : {"Node2Vec": 3.44, "DeepWalk": 3.21, "SEAL": 3.87,
                 "GAT": 4.02, "VGAE": 3.17, "LPNE2GGCN": 2.28},
}

COLORS = sns.color_palette("tab10", n_colors=len(METHODS))


def _style():
    sns.set_theme(style="whitegrid", font_scale=1.1)


def plot_accuracy_comparison(save: bool = True) -> None:
    """Figure 2 — Accuracy comparison bar chart."""
    _style()
    n_datasets = len(DATASETS)
    n_methods  = len(METHODS)
    x = np.arange(n_datasets)
    width = 0.10

    fig, ax = plt.subplots(figsize=(16, 6))

    for i, method in enumerate(METHODS):
        offset = (i - n_methods / 2 + 0.5) * width
        vals = ACCURACY_TABLE[method]
        bars = ax.bar(x + offset, vals, width, label=method,
                      color=COLORS[i], alpha=0.85, edgecolor="white")
        # Annotate top of each bar
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{v:.1f}",
                ha="center", va="bottom",
                fontsize=5.5, rotation=90,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS, fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Accuracy Comparison Scores — Figure 2", fontsize=14)
    ax.set_ylim(0, 115)
    ax.legend(loc="lower right", fontsize=8, ncol=2)

    plt.tight_layout()
    if save:
        path = os.path.join(RESULTS_DIR, "figure2_accuracy.png")
        plt.savefig(path, dpi=150)
        print(f"[visualize] Saved → {path}")
    plt.show()


def plot_auc_comparison(save: bool = True) -> None:
    """Figure 4 — AUC comparison bar chart."""
    _style()
    n_datasets = len(DATASETS)
    n_methods  = len(METHODS)
    x = np.arange(n_datasets)
    width = 0.10

    fig, ax = plt.subplots(figsize=(16, 6))

    for i, method in enumerate(METHODS):
        offset = (i - n_methods / 2 + 0.5) * width
        vals = AUC_TABLE[method]
        ax.bar(x + offset, vals, width, label=method,
               color=COLORS[i], alpha=0.85, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS, fontsize=12)
    ax.set_ylabel("AUC (%)", fontsize=12)
    ax.set_title("AUC Comparison Scores — Figure 4", fontsize=14)
    ax.set_ylim(0, 110)
    ax.legend(loc="lower right", fontsize=8, ncol=2)

    plt.tight_layout()
    if save:
        path = os.path.join(RESULTS_DIR, "figure4_auc.png")
        plt.savefig(path, dpi=150)
        print(f"[visualize] Saved → {path}")
    plt.show()


def plot_timing_chart(save: bool = True) -> None:
    """Figure 3 — Computational Time Chart."""
    _style()
    models = list(TIME_TABLE["Facebook"].keys())
    fb_times   = [TIME_TABLE["Facebook"][m] for m in models]
    cora_times = [TIME_TABLE["Cora"][m]     for m in models]

    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(x, fb_times,   "o-", color="steelblue",  label="Facebook", linewidth=2)
    ax.plot(x, cora_times, "s-", color="darkorange",  label="Cora",     linewidth=2)

    # Exponential trend for Facebook (as shown in the paper figure)
    z = np.polyfit(x, fb_times, 2)
    p = np.poly1d(z)
    xs = np.linspace(0, len(models) - 1, 200)
    ax.plot(xs, p(xs), "--", color="steelblue", alpha=0.5, label="Exp. (Facebook)")

    # Annotate points
    for xi, (fb, co) in enumerate(zip(fb_times, cora_times)):
        ax.annotate(f"{fb}", (xi, fb),   textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9)
        ax.annotate(f"{co}", (xi, co),   textcoords="offset points",
                    xytext=(0, -14), ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel("Time (seconds)", fontsize=12)
    ax.set_title("Computational Time Chart — Figure 3", fontsize=14)
    ax.set_ylim(0, 5.5)
    ax.legend(fontsize=10)

    plt.tight_layout()
    if save:
        path = os.path.join(RESULTS_DIR, "figure3_timing.png")
        plt.savefig(path, dpi=150)
        print(f"[visualize] Saved → {path}")
    plt.show()


def plot_layer_ablation(save: bool = True) -> None:
    """Table 6 — Effect of number of convolutional layers."""
    _style()
    layer_data = {
        "2-Layers": [69.54, 61.32, 56.54, 69.49, 58.89],
        "3-Layers": [86.62, 85.87, 84.79, 87.12, 69.18],
        "4-Layers": [68.44, 71.98, 64.32, 72.62, 74.21],
    }
    x = np.arange(len(DATASETS))
    width = 0.25
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (label, vals) in enumerate(layer_data.items()):
        offset = (i - 1) * width
        ax.bar(x + offset, vals, width, label=label,
               color=colors[i], alpha=0.85, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS, fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Effect of Number of Convolutional Layers (Table 6)", fontsize=13)
    ax.legend(fontsize=11)
    plt.tight_layout()

    if save:
        path = os.path.join(RESULTS_DIR, "table6_layer_ablation.png")
        plt.savefig(path, dpi=150)
        print(f"[visualize] Saved → {path}")
    plt.show()
