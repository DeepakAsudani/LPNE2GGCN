"""
negative_sampling.py
--------------------
Implements the positive/negative edge-pair construction described in
Algorithm 1 of the LPNE2GGCN paper.

Key design decisions (per the paper):
  - Positive edges  : a random subset of existing edges is withheld
                      (label = 1). Connectivity is preserved: an edge
                      is only removed if both its endpoints remain
                      connected to the rest of the graph.
  - Negative edges  : disconnected node pairs within shortest-path
                      distance ≤ 3, identified via NetworkX.
                      (label = 0)
  - Train/test split: 80 / 20 by default.
"""

import random
import warnings
import numpy as np
import networkx as nx
import torch
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Core helper: build NetworkX graph
# ---------------------------------------------------------------------------

def _edge_index_to_nx(edge_index: torch.Tensor, num_nodes: int) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    edges = edge_index.t().tolist()
    G.add_edges_from(edges)
    return G


# ---------------------------------------------------------------------------
# Positive edge sampling (Algorithm 1, lines 4–7)
# ---------------------------------------------------------------------------

def _sample_positive_edges(
    G: nx.Graph,
    ratio: float = 0.5,
    seed: int = 42,
) -> tuple[list, nx.Graph]:
    """
    Randomly remove up to `ratio` of edges from G (label = 1).
    An edge (u, v) is only removed if doing so keeps both u and v
    connected to the rest of the graph (degree ≥ 1 for both endpoints).

    Returns
    -------
    positive_edges : list of (u, v) pairs that were removed
    G_train        : remaining graph used for embedding / training
    """
    rng = random.Random(seed)
    all_edges = list(G.edges())
    rng.shuffle(all_edges)

    target = int(len(all_edges) * ratio)
    positive_edges = []
    G_train = G.copy()

    for u, v in all_edges:
        if len(positive_edges) >= target:
            break
        # Only remove if both nodes stay connected
        if G_train.degree(u) > 1 and G_train.degree(v) > 1:
            G_train.remove_edge(u, v)
            positive_edges.append((u, v))

    return positive_edges, G_train


# ---------------------------------------------------------------------------
# Negative edge sampling (Algorithm 1 — disconnected pairs ≤ distance 3)
# ---------------------------------------------------------------------------

def _sample_negative_edges(
    G_train: nx.Graph,
    num_samples: int,
    max_distance: int = 3,
    seed: int = 42,
) -> list:
    """
    Sample `num_samples` disconnected node pairs (label = 0) that are
    within shortest-path distance ≤ `max_distance` (paper uses 3).

    For large graphs, we approximate by sampling random non-edge pairs
    and checking connectivity via BFS-limited path length.
    """
    rng = random.Random(seed)
    nodes = list(G_train.nodes())
    existing_edges = set(G_train.edges())
    # Canonicalise: store as frozensets for O(1) lookup
    existing_edges_set = {frozenset(e) for e in existing_edges}

    negatives = []
    attempts = 0
    max_attempts = num_samples * 20  # generous ceiling

    while len(negatives) < num_samples and attempts < max_attempts:
        attempts += 1
        u = rng.choice(nodes)
        v = rng.choice(nodes)
        if u == v:
            continue
        if frozenset({u, v}) in existing_edges_set:
            continue
        # Check distance ≤ max_distance using cutoff BFS
        try:
            d = nx.shortest_path_length(G_train, u, v, cutoff=max_distance)
            if d is not None and d <= max_distance:
                negatives.append((u, v))
                existing_edges_set.add(frozenset({u, v}))  # avoid duplicates
        except nx.NetworkXNoPath:
            pass
        except Exception:
            pass

    # If we could not collect enough close negatives, pad with random pairs
    while len(negatives) < num_samples:
        u, v = rng.sample(nodes, 2)
        if frozenset({u, v}) not in existing_edges_set:
            negatives.append((u, v))
            existing_edges_set.add(frozenset({u, v}))

    return negatives[:num_samples]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def prepare_link_prediction_data(
    graph: dict,
    test_ratio: float = 0.2,
    pos_ratio: float = 0.5,
    max_neg_distance: int = 3,
    seed: int = 42,
) -> dict:
    """
    Full Algorithm 1 pipeline.

    Parameters
    ----------
    graph          : output of data_loader.load_dataset()
    test_ratio     : fraction of labelled pairs held out for testing
    pos_ratio      : fraction of edges withheld as positive samples
    max_neg_distance: maximum shortest-path distance for negative pairs
    seed           : RNG seed

    Returns
    -------
    dict with keys:
      'G_train'         : nx.Graph  (training graph for embedding)
      'train_edges'     : list[(u,v)]
      'train_labels'    : list[int]   (0 or 1)
      'test_edges'      : list[(u,v)]
      'test_labels'     : list[int]
      'edge_index_train': torch.LongTensor  [2, E_train]  (for GNN)
    """
    edge_index = graph["edge_index"]
    num_nodes  = graph["num_nodes"]

    print(f"  [neg_sampling] Building NetworkX graph ...")
    G = _edge_index_to_nx(edge_index, num_nodes)

    # --- Positive edges (removed from training graph) ---
    print(f"  [neg_sampling] Sampling positive edges (ratio={pos_ratio}) ...")
    pos_edges, G_train = _sample_positive_edges(G, ratio=pos_ratio, seed=seed)

    # --- Negative edges (disconnected pairs ≤ dist 3) ---
    n_neg = len(pos_edges)
    print(f"  [neg_sampling] Sampling {n_neg} negative edges (dist ≤ {max_neg_distance}) ...")
    # For large graphs, limit BFS to avoid O(N²) traversal
    if num_nodes > 20_000:
        warnings.warn(
            "Large graph detected — using random non-edge sampling "
            "(skipping distance check for performance).",
            RuntimeWarning,
        )
        neg_edges = _fast_random_negatives(G_train, n_neg, seed=seed)
    else:
        neg_edges = _sample_negative_edges(G_train, n_neg, max_neg_distance, seed)

    # --- Combine and split ---
    all_edges  = pos_edges  + neg_edges
    all_labels = [1] * len(pos_edges) + [0] * len(neg_edges)

    train_edges, test_edges, train_labels, test_labels = train_test_split(
        all_edges, all_labels,
        test_size=test_ratio,
        random_state=seed,
        stratify=all_labels,
    )

    # Rebuild edge_index for training graph (undirected)
    train_src, train_dst = zip(*G_train.edges()) if G_train.number_of_edges() > 0 \
        else ([], [])
    ei = torch.tensor(
        [list(train_src) + list(train_dst),
         list(train_dst) + list(train_src)],
        dtype=torch.long,
    )

    print(
        f"  [neg_sampling] Done. "
        f"train={len(train_edges)} pairs | test={len(test_edges)} pairs"
    )

    return {
        "G_train"         : G_train,
        "train_edges"     : train_edges,
        "train_labels"    : train_labels,
        "test_edges"      : test_edges,
        "test_labels"     : test_labels,
        "edge_index_train": ei,
    }


def _fast_random_negatives(G: nx.Graph, n: int, seed: int = 42) -> list:
    """Fast non-edge sampling for large graphs (no distance check)."""
    rng = random.Random(seed)
    nodes = list(G.nodes())
    edge_set = {frozenset(e) for e in G.edges()}
    negatives = []
    max_attempts = n * 30
    attempts = 0
    while len(negatives) < n and attempts < max_attempts:
        attempts += 1
        u, v = rng.sample(nodes, 2)
        key = frozenset({u, v})
        if key not in edge_set:
            negatives.append((u, v))
            edge_set.add(key)
    # Last resort: sequential fill
    while len(negatives) < n:
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                if len(negatives) >= n:
                    break
                key = frozenset({nodes[i], nodes[j]})
                if key not in edge_set:
                    negatives.append((nodes[i], nodes[j]))
                    edge_set.add(key)
    return negatives[:n]
