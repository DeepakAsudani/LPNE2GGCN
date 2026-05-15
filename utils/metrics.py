"""
metrics.py
----------
Evaluation metrics for link prediction (Section IV-B of the paper):
  - Accuracy  (Eq. 9)
  - F1-Score  (Eq. 10)
  - AUC       (Eq. 11)
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)


def compute_metrics(
    labels: list | np.ndarray,
    scores: list | np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    Compute Accuracy, F1-Score, and AUC for link prediction.

    Parameters
    ----------
    labels    : ground-truth binary labels  (0 = no link, 1 = link)
    scores    : predicted probabilities or similarity scores in [0, 1]
    threshold : decision boundary for Accuracy and F1

    Returns
    -------
    dict with keys 'accuracy', 'f1', 'auc'
    """
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)

    preds = (scores >= threshold).astype(int)

    acc = accuracy_score(labels, preds) * 100.0
    f1  = f1_score(labels, preds, zero_division=0) * 100.0

    # AUC as defined in Eq. 11:
    #   AUC = (n1 + 0.5*n2) / n
    # which equals sklearn's roc_auc_score when ties are broken uniformly.
    try:
        auc = roc_auc_score(labels, scores) * 100.0
    except ValueError:
        # All-same-class edge case
        auc = 50.0

    return {"accuracy": acc, "f1": f1, "auc": auc}


def auc_from_counts(n1: int, n2: int, n: int) -> float:
    """
    Direct implementation of Eq. 11.

    Parameters
    ----------
    n1 : number of comparisons where missing link scored higher
    n2 : number of ties
    n  : total independent comparisons

    Returns
    -------
    AUC in [0, 1]
    """
    return (n1 + 0.5 * n2) / n
