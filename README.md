# LPNE2GGCN: Link Prediction via Node Embedding with Generalized Graph Convolutional Networks

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-orange)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Paper:** *Predicting Probabilistic Links Using Node Embedding Using Graph Neural Network Approach*  
> Riju Bhattacharya, Naresh Kumar Nagwani, Deepak Suresh Asudani, Rishav Dubey  
> *IEEE Access, DOI: 10.1109/ACCESS.2024.0429000*

---

## Overview

LPNE2GGCN is a graph neural network framework for **probabilistic link prediction** that combines:

1. **Node2Vec** — biased second-order random walks to learn structural node embeddings
2. **GraphSAGE** — inductive, scalable graph convolution (Mean / LSTM / Pooling aggregators)
3. **Adam / Adadelta optimizers** — for enhanced convergence

The model is evaluated on five real-world graph datasets: **Citeseer, Cora, Facebook, IMDB, DBLP**.

---

## Repository Structure

```
LPNE2GGCN/
├── README.md                   ← You are here
├── requirements.txt            ← Python dependencies
├── main.py                     ← Single entry-point: train & evaluate all datasets
├── reproduce_results.py        ← Reproduce all Table 3/4/5 results from the paper
│
├── models/
│   ├── __init__.py
│   ├── node2vec.py             ← Node2Vec random-walk embedding
│   ├── graphsage.py            ← GraphSAGE (Mean/LSTM/Pooling aggregators)
│   └── lpne2ggcn.py            ← Full LPNE2GGCN pipeline
│
├── utils/
│   ├── __init__.py
│   ├── data_loader.py          ← Dataset download & preprocessing
│   ├── negative_sampling.py    ← Positive/negative edge pair construction (Algorithm 1)
│   ├── metrics.py              ← AUC, Accuracy, F1-Score
│   └── visualize.py            ← Plot accuracy / AUC / timing charts
│
├── data/                       ← Auto-populated on first run
│   └── (Citeseer, Cora, Facebook, IMDB, DBLP cached here)
│
├── results/                    ← CSV logs & saved model checkpoints
│
└── notebooks/
    └── LPNE2GGCN_Demo.ipynb    ← End-to-end Jupyter walkthrough
```

---

## Quick Start

### 1 — Clone & Install

```bash
git clone https://github.com/<your-username>/LPNE2GGCN.git
cd LPNE2GGCN

# (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2 — Reproduce All Paper Results (Tables 3, 4, 5)

```bash
python reproduce_results.py
```

This downloads all datasets automatically, trains every model variant, and prints a
formatted comparison table identical to Tables 3–5 in the paper.  
Results are also saved to `results/paper_results.csv`.

### 3 — Train a Single Dataset / Variant

```bash
# LPNE2GGCN + Adam on Cora
python main.py --dataset cora --optimizer adam --epochs 200

# LPNE2GGCN + Adadelta on Facebook
python main.py --dataset facebook --optimizer adadelta --epochs 200

# Baseline: Node2Vec only
python main.py --dataset citeseer --model node2vec

# All datasets, all variants
python main.py --dataset all --optimizer both
```

#### Full CLI reference

| Argument | Default | Choices | Description |
|---|---|---|---|
| `--dataset` | `cora` | `cora citeseer facebook imdb dblp all` | Dataset to use |
| `--model` | `lpne2ggcn` | `lpne2ggcn node2vec deepwalk seal gat vgae` | Model variant |
| `--optimizer` | `adam` | `adam adadelta both` | Optimizer |
| `--epochs` | `200` | int | Training epochs |
| `--hidden_dim` | `128` | int | Hidden layer size |
| `--embedding_dim` | `64` | int | Node2Vec output dimension |
| `--walk_length` | `10` | int | Random walk length (walk-l) |
| `--num_walks` | `10` | int | Walks per node (n-walks) |
| `--p` | `1.0` | float | Node2Vec return parameter |
| `--q` | `1.0` | float | Node2Vec in-out parameter |
| `--dropout` | `0.3` | float | Dropout rate |
| `--lr` | `0.01` | float | Learning rate |
| `--layers` | `3` | int | Number of GraphSAGE layers (best=3, see Table 6) |
| `--aggregator` | `mean` | `mean lstm pool` | GraphSAGE aggregator type |
| `--seed` | `42` | int | Random seed for reproducibility |
| `--save_model` | False | flag | Save trained model to `results/` |
| `--plot` | False | flag | Generate accuracy/AUC/timing charts |

---

## Expected Results

### Table 3 — Accuracy (%)

| Method | Citeseer | Cora | Facebook | IMDB | DBLP |
|---|---|---|---|---|---|
| Node2Vec | 74.52 | 87.67 | 86.94 | 83.31 | 78.16 |
| DeepWalk | 60.49 | 75.43 | 85.12 | 76.22 | 60.31 |
| VGAE | 84.34 | 89.67 | 83.31 | 66.82 | 87.65 |
| SEAL | 67.23 | 72.87 | 58.83 | 71.04 | 51.43 |
| GAT | 81.89 | 68.33 | 64.13 | 56.19 | 58.59 |
| **LPNE2GGCN** | 91.32 | 90.14 | 89.02 | 88.67 | 82.51 |
| **LPNE2GGCN + Adadelta** | 93.56 | 95.85 | 91.66 | **91.08** | **90.12** |
| **LPNE2GGCN + Adam** | **94.74** | **96.46** | **93.82** | 90.35 | 89.48 |

### Table 4 — AUC (%)

| Method | Citeseer | Cora | Facebook | IMDB | DBLP |
|---|---|---|---|---|---|
| Node2Vec | 79.43 | 70.58 | 80.64 | 67.5 | 53.34 |
| DeepWalk | 65.45 | 66.12 | 56.61 | 50.28 | 31.17 |
| VGAE | 80.12 | 79.56 | 32.44 | 89.04 | 81.45 |
| SEAL | 69.9 | 67.53 | 49.3 | 49.69 | 53.49 |
| GAT | 78.42 | 51.08 | 53.2 | 55.51 | 68.21 |
| **LPNE2GGCN** | 88.79 | 87.51 | 87.52 | 88.82 | 78.65 |
| **LPNE2GGCN + Adadelta** | 94.56 | 93.73 | **96.66** | 92.08 | 91.12 |
| **LPNE2GGCN + Adam** | **95.36** | **97.46** | 92.28 | 89.91 | **93.65** |

### Table 5 — F1-Score (%)

| Method | Citeseer | Cora | Facebook | IMDB | DBLP |
|---|---|---|---|---|---|
| Node2Vec | 52.8 | 69.74 | 75.51 | 64.25 | 52.62 |
| DeepWalk | 61.62 | 58.21 | 65.11 | 63.12 | 61.33 |
| VGAE | 36.08 | 79.45 | 67.57 | 41.28 | 61.92 |
| SEAL | 53.8 | 64.74 | 52.01 | 60.96 | 75.53 |
| GAT | 57.7 | 53.63 | 57.72 | 57.14 | 58.69 |
| **LPNE2GGCN** | 86.32 | 82.98 | 86.94 | 63.48 | 67.71 |
| **LPNE2GGCN + Adadelta** | 93.56 | 95.85 | 91.66 | **91.08** | **90.12** |
| **LPNE2GGCN + Adam** | **94.74** | **96.46** | **93.82** | 90.35 | 89.48 |

### Table 6 — Effect of Convolutional Layers (Accuracy %)

| Dataset | 2-Layers | 3-Layers | 4-Layers |
|---|---|---|---|
| Citeseer | 69.54 | **86.62** | 68.44 |
| Cora | 61.32 | **85.87** | 71.98 |
| Facebook | 56.54 | **84.79** | 64.32 |
| IMDB | 69.49 | **87.12** | 72.62 |
| DBLP | 58.89 | 69.18 | **74.21** |

---

## Datasets

Datasets are downloaded automatically via `torch_geometric` on first run.

| Dataset | Nodes | Edges | Communities | Attributes | Type |
|---|---|---|---|---|---|
| Citeseer | 3,312 | 4,715 | 6 | 3,703 | Citation |
| Cora | 2,708 | 5,429 | 7 | 1,433 | Citation |
| Facebook | 4,031 | 88,234 | 193 | 1,283 | Social |
| IMDB | 4,780 | 98,010 | 3 | 1,232 | Movie |
| DBLP | 371,080 | 1,049,866 | 13,477 | — | Bibliography |

---

## Hardware & Reproducibility

All experiments in the paper were run on:
- Intel Core i7 @ 3.4 GHz, 32 GB RAM
- No GPU required (CPU-only runs fine for all datasets except DBLP)

To exactly reproduce results, fix the seed:
```bash
python reproduce_results.py --seed 42
```

---

## Citation

If you use this code, please cite the original paper:

```bibtex
@article{bhattacharya2024lpne2ggcn,
  title   = {Predicting Probabilistic Links Using Node Embedding Using
             Graph Neural Network Approach},
  author  = {Bhattacharya, Riju and Nagwani, Naresh Kumar and
             Asudani, Deepak Suresh and Dubey, Rishav},
  journal = {IEEE Access},
  year    = {2024},
  doi     = {10.1109/ACCESS.2024.0429000}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
