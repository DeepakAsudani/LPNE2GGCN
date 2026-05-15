"""
node2vec.py
-----------
Node2Vec implementation following Section III-B of the paper.

Key equations implemented:
  Eq. 2  — transition probability P(C_i = x | C_{i-1} = y)
  Eq. 3  — biased transition weight α_pq(t, x) using parameters p, q
  Eq. 8  — decoder: Decode(u_i, u_j) = P,R(v_i | v_j)

Algorithm:
  1. Generate biased second-order random walks from every node (Alg 1, lines 3-7)
  2. Feed walks to skip-gram with negative sampling (SGNS)
  3. Extract hidden-layer weights as node embeddings
"""

import random
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict


# ---------------------------------------------------------------------------
# Alias sampling — O(1) node sampling per step (used for efficiency)
# ---------------------------------------------------------------------------

class AliasSampler:
    """
    Implements the alias method for O(1) sampling from a discrete distribution.
    Used by Node2Vec for efficient biased random walks.
    """

    def __init__(self, probs: np.ndarray):
        n = len(probs)
        probs = np.asarray(probs, dtype=np.float64)
        probs = probs / probs.sum()            # normalise

        self.alias = np.zeros(n, dtype=np.int64)
        self.prob  = np.zeros(n, dtype=np.float64)

        small, large = [], []
        p_scaled = probs * n

        for i, p in enumerate(p_scaled):
            (small if p < 1.0 else large).append(i)

        while small and large:
            s = small.pop()
            l = large.pop()
            self.prob[s]  = p_scaled[s]
            self.alias[s] = l
            p_scaled[l]  += p_scaled[s] - 1.0
            (small if p_scaled[l] < 1.0 else large).append(l)

        for i in small + large:
            self.prob[i] = 1.0

    def sample(self, rng: np.random.Generator) -> int:
        n  = len(self.prob)
        i  = int(rng.integers(0, n))
        return i if rng.random() < self.prob[i] else int(self.alias[i])


# ---------------------------------------------------------------------------
# Biased random walk generator (Eq. 2 & 3)
# ---------------------------------------------------------------------------

def _compute_transition_probs(
    G: nx.Graph,
    p: float,
    q: float,
) -> dict:
    """
    Pre-compute normalised transition probabilities for every directed edge.
    Implements Eq. 3:

        α_pq(t, x) = 1/p  if d_tx = 0   (return to t)
                     1    if d_tx = 1   (in neighbourhood of t)
                     1/q  if d_tx = 2   (outside neighbourhood)

    Returns
    -------
    alias_nodes : {node: AliasSampler}           (for first step)
    alias_edges : {(src, dst): AliasSampler}     (for subsequent steps)
    """
    alias_nodes = {}
    alias_edges = {}

    # Per-node: uniform over neighbours (first step)
    for node in G.nodes():
        nbrs = list(G.neighbors(node))
        if not nbrs:
            alias_nodes[node] = AliasSampler(np.array([1.0]))
        else:
            alias_nodes[node] = AliasSampler(
                np.array([G[node][nbr].get("weight", 1.0) for nbr in nbrs])
            )

    # Per-directed-edge: second-order biased probabilities (Eq. 3)
    for src in G.nodes():
        for dst in G.neighbors(src):
            nbrs_dst = list(G.neighbors(dst))
            if not nbrs_dst:
                alias_edges[(src, dst)] = AliasSampler(np.array([1.0]))
                continue

            weights = []
            for x in nbrs_dst:
                w = G[dst][x].get("weight", 1.0)
                if x == src:          # d_tx = 0 → return
                    alpha = 1.0 / p
                elif G.has_edge(src, x):  # d_tx = 1 → in neighbourhood
                    alpha = 1.0
                else:                 # d_tx = 2 → explore further
                    alpha = 1.0 / q
                weights.append(w * alpha)

            alias_edges[(src, dst)] = AliasSampler(np.array(weights))

    return alias_nodes, alias_edges


def _generate_walks(
    G: nx.Graph,
    alias_nodes: dict,
    alias_edges: dict,
    walk_length: int,
    num_walks: int,
    seed: int = 42,
) -> list[list[int]]:
    """
    Generate biased second-order random walks (Algorithm 1, lines 3-7).
    Each walk is a list of node IDs.
    """
    rng   = np.random.default_rng(seed)
    nodes = list(G.nodes())
    walks = []

    for _ in range(num_walks):
        rng.shuffle(nodes)
        for start in nodes:
            walk = [start]
            while len(walk) < walk_length:
                cur  = walk[-1]
                nbrs = list(G.neighbors(cur))
                if not nbrs:
                    break
                if len(walk) == 1:
                    # First step: uniform over neighbours
                    idx = alias_nodes[cur].sample(rng)
                    walk.append(nbrs[min(idx, len(nbrs) - 1)])
                else:
                    prev = walk[-2]
                    key  = (prev, cur)
                    idx  = alias_edges[key].sample(rng)
                    walk.append(nbrs[min(idx, len(nbrs) - 1)])
            walks.append(walk)

    return walks


# ---------------------------------------------------------------------------
# Skip-gram with Negative Sampling (SGNS)
# ---------------------------------------------------------------------------

class SkipGramModel(nn.Module):
    """
    Skip-gram model trained on random walk sequences.
    Embeddings are the rows of self.embeddings.weight.
    """

    def __init__(self, num_nodes: int, embedding_dim: int):
        super().__init__()
        self.embeddings = nn.Embedding(num_nodes, embedding_dim, sparse=True)
        self.context    = nn.Embedding(num_nodes, embedding_dim, sparse=True)
        nn.init.xavier_uniform_(self.embeddings.weight.data)
        nn.init.xavier_uniform_(self.context.weight.data)

    def forward(
        self,
        center: torch.Tensor,
        pos_ctx: torch.Tensor,
        neg_ctx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Binary cross-entropy loss for one batch.
        """
        emb_c   = self.embeddings(center).unsqueeze(1)    # [B, 1, D]
        emb_pos = self.context(pos_ctx).unsqueeze(2)      # [B, D, 1]
        emb_neg = self.context(neg_ctx)                   # [B, K, D]

        pos_score = torch.bmm(emb_c, emb_pos).squeeze()   # [B]
        neg_score = torch.bmm(emb_c, emb_neg.transpose(1, 2)).squeeze(1)  # [B, K]

        loss = (
            -torch.log(torch.sigmoid(pos_score) + 1e-8).mean()
            - torch.log(torch.sigmoid(-neg_score) + 1e-8).mean()
        )
        return loss


def _build_vocabulary(walks: list[list[int]]) -> tuple[dict, list]:
    """Build word→id mapping and unigram distribution for negative sampling."""
    freq = defaultdict(int)
    for walk in walks:
        for node in walk:
            freq[node] += 1

    vocab   = sorted(freq.keys())
    node2id = {n: i for i, n in enumerate(vocab)}
    counts  = np.array([freq[n] for n in vocab], dtype=np.float64)
    # Raise to 3/4 power (standard for SGNS negative sampling)
    counts  = counts ** 0.75
    counts /= counts.sum()
    return node2id, vocab, counts


def _train_skipgram(
    walks: list[list[int]],
    num_nodes: int,
    embedding_dim: int,
    window: int = 5,
    num_neg: int = 5,
    epochs: int = 5,
    lr: float = 0.025,
    batch_size: int = 512,
    seed: int = 42,
) -> np.ndarray:
    """Train skip-gram and return embedding matrix [num_nodes, embedding_dim]."""
    torch.manual_seed(seed)
    node2id, vocab, neg_probs = _build_vocabulary(walks)
    id2orig = {i: n for n, i in node2id.items()}   # mapped_id → original node id

    # Build (center, context) pairs
    pairs = []
    for walk in walks:
        for i, center in enumerate(walk):
            lo = max(0, i - window)
            hi = min(len(walk), i + window + 1)
            for j in range(lo, hi):
                if i != j:
                    pairs.append((node2id[center], node2id[walk[j]]))

    if not pairs:
        return np.zeros((num_nodes, embedding_dim), dtype=np.float32)

    pairs = np.array(pairs, dtype=np.int64)
    rng   = np.random.default_rng(seed)
    model = SkipGramModel(len(vocab), embedding_dim)
    opt   = optim.SparseAdam(list(model.parameters()), lr=lr)

    model.train()
    for epoch in range(epochs):
        rng.shuffle(pairs)
        total_loss = 0.0
        n_batches  = 0
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start: start + batch_size]
            if len(batch) == 0:
                continue
            centers = torch.from_numpy(batch[:, 0])
            contexts = torch.from_numpy(batch[:, 1])

            # Negative samples
            neg_ids = rng.choice(
                len(vocab),
                size=(len(batch), num_neg),
                p=neg_probs,
                replace=True,
            )
            neg_ctx = torch.from_numpy(neg_ids.astype(np.int64))

            opt.zero_grad()
            loss = model(centers, contexts, neg_ctx)
            loss.backward()
            opt.step()

            total_loss += loss.item()
            n_batches  += 1

    # Extract embeddings: shape [len(vocab), embedding_dim]
    emb_matrix = model.embeddings.weight.data.numpy()

    # Map back to original node ids
    full_matrix = np.zeros((num_nodes, embedding_dim), dtype=np.float32)
    for mapped_id, orig_id in id2orig.items():
        if orig_id < num_nodes:
            full_matrix[orig_id] = emb_matrix[mapped_id]

    return full_matrix


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class Node2VecEmbedder:
    """
    Full Node2Vec pipeline as described in Section III-B.

    Parameters
    ----------
    p             : return parameter  (controls tendency to revisit)
    q             : in-out parameter  (controls DFS vs BFS balance)
    walk_length   : number of nodes in each random walk (walk-l)
    num_walks     : number of walks starting from each node (n-walks)
    embedding_dim : dimensionality of output embeddings (d)
    window        : skip-gram context window size
    num_neg       : number of negative samples per positive pair
    epochs        : number of training epochs for skip-gram
    seed          : RNG seed
    """

    def __init__(
        self,
        p: float = 1.0,
        q: float = 1.0,
        walk_length: int = 10,
        num_walks: int = 10,
        embedding_dim: int = 64,
        window: int = 5,
        num_neg: int = 5,
        epochs: int = 5,
        seed: int = 42,
    ):
        self.p             = p
        self.q             = q
        self.walk_length   = walk_length
        self.num_walks     = num_walks
        self.embedding_dim = embedding_dim
        self.window        = window
        self.num_neg       = num_neg
        self.epochs        = epochs
        self.seed          = seed
        self.embeddings_   = None  # set after fit()

    def fit(self, G: nx.Graph) -> "Node2VecEmbedder":
        """
        Fit node embeddings on graph G.

        Parameters
        ----------
        G : nx.Graph — the training graph

        Returns
        -------
        self (for chaining)
        """
        num_nodes = G.number_of_nodes()
        if num_nodes == 0:
            raise ValueError("Graph has no nodes.")

        print(f"    [Node2Vec] Pre-computing transition probabilities "
              f"(p={self.p}, q={self.q}) ...")
        alias_nodes, alias_edges = _compute_transition_probs(G, self.p, self.q)

        print(f"    [Node2Vec] Generating walks "
              f"(length={self.walk_length}, walks/node={self.num_walks}) ...")
        walks = _generate_walks(
            G, alias_nodes, alias_edges,
            self.walk_length, self.num_walks, self.seed,
        )

        print(f"    [Node2Vec] Training skip-gram (dim={self.embedding_dim}) ...")
        self.embeddings_ = _train_skipgram(
            walks, num_nodes, self.embedding_dim,
            window=self.window,
            num_neg=self.num_neg,
            epochs=self.epochs,
            seed=self.seed,
        )
        return self

    def get_embeddings(self) -> np.ndarray:
        """Return embedding matrix [num_nodes, embedding_dim]."""
        if self.embeddings_ is None:
            raise RuntimeError("Call fit() before get_embeddings().")
        return self.embeddings_

    def get_link_features(
        self,
        edges: list,
        operator: str = "hadamard",
    ) -> np.ndarray:
        """
        Combine node embeddings for a list of edge pairs.

        Parameters
        ----------
        edges    : list of (u, v) tuples
        operator : 'hadamard' | 'average' | 'l1' | 'l2'

        Returns
        -------
        np.ndarray [len(edges), embedding_dim]
        """
        emb = self.embeddings_
        ops = {
            "hadamard": lambda u, v: emb[u] * emb[v],
            "average" : lambda u, v: (emb[u] + emb[v]) / 2,
            "l1"      : lambda u, v: np.abs(emb[u] - emb[v]),
            "l2"      : lambda u, v: (emb[u] - emb[v]) ** 2,
        }
        fn = ops.get(operator, ops["hadamard"])
        return np.stack([fn(u, v) for u, v in edges], axis=0)
