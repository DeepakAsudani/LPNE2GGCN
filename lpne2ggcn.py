"""
lpne2ggcn.py
------------
LPNE2GGCN: Link Prediction via Node Embedding and
           Extended Generalized Graph Convolutional Networks

Full pipeline (Algorithm 1 of the paper):
  Phase 1  —  Pre-process graph G; generate biased random walks
  Phase 2  —  Feed walks to skip-gram (SGNS) → node embeddings G'
  Phase 3  —  Feed G' to GraphSAGE → refined embeddings
  Phase 4  —  Classify node-pair embeddings → link probability
  Phase 5  —  Optimise with Adam or Adadelta
"""

import time
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from .node2vec import Node2VecEmbedder
from .graphsage import GraphSAGE
from ..utils.metrics import compute_metrics


# ---------------------------------------------------------------------------
# Link classifier head (Algorithm 1, line 13)
# ---------------------------------------------------------------------------

class LinkClassifier(nn.Module):
    """
    Three-layer MLP for link classification (paper: 'hidden neural network
    consisting of three layers', Algorithm 1 line 13).

    Input  : concatenated / hadamard node-pair embedding
    Output : sigmoid probability of a link existing
    """

    def __init__(self, in_dim: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim // 2, in_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim // 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Main LPNE2GGCN class
# ---------------------------------------------------------------------------

class LPNE2GGCN:
    """
    Full LPNE2GGCN pipeline.

    Parameters
    ----------
    embedding_dim : Node2Vec output dimension (d)
    hidden_dim    : GraphSAGE hidden layer size
    num_layers    : number of GraphSAGE layers  (paper optimum = 3)
    aggregator    : GraphSAGE aggregator type  ('mean' | 'lstm' | 'pool')
    dropout       : dropout rate (paper default = 0.3)
    p             : Node2Vec return parameter   (paper optimal = 1)
    q             : Node2Vec in-out parameter   (paper optimal = 1)
    walk_length   : random walk length  (paper: 10)
    num_walks     : walks per node      (paper: 10)
    optimizer     : 'adam' | 'adadelta'
    lr            : learning rate       (paper: default Adam/Adadelta)
    epochs        : GNN training epochs
    seed          : random seed
    """

    def __init__(
        self,
        embedding_dim: int   = 64,
        hidden_dim   : int   = 128,
        num_layers   : int   = 3,
        aggregator   : str   = "mean",
        dropout      : float = 0.3,
        p            : float = 1.0,
        q            : float = 1.0,
        walk_length  : int   = 10,
        num_walks    : int   = 10,
        optimizer    : str   = "adam",
        lr           : float = 0.01,
        epochs       : int   = 200,
        seed         : int   = 42,
    ):
        self.embedding_dim = embedding_dim
        self.hidden_dim    = hidden_dim
        self.num_layers    = num_layers
        self.aggregator    = aggregator
        self.dropout       = dropout
        self.p             = p
        self.q             = q
        self.walk_length   = walk_length
        self.num_walks     = num_walks
        self.optimizer_name = optimizer.lower()
        self.lr            = lr
        self.epochs        = epochs
        self.seed          = seed

        self._node2vec   = None
        self._graphsage  = None
        self._classifier = None
        self._scaler     = None
        self.training_time_ = 0.0

    # ------------------------------------------------------------------
    # Phase 1 & 2 — Node2Vec embeddings
    # ------------------------------------------------------------------

    def _fit_node2vec(self, G_train: nx.Graph) -> np.ndarray:
        self._node2vec = Node2VecEmbedder(
            p=self.p, q=self.q,
            walk_length=self.walk_length,
            num_walks=self.num_walks,
            embedding_dim=self.embedding_dim,
            seed=self.seed,
        )
        self._node2vec.fit(G_train)
        return self._node2vec.get_embeddings()   # [N, embedding_dim]

    # ------------------------------------------------------------------
    # Phase 3 — GraphSAGE refinement
    # ------------------------------------------------------------------

    def _fit_graphsage(
        self,
        node2vec_emb : np.ndarray,
        edge_index   : torch.Tensor,
        train_edges  : list,
        train_labels : list,
    ) -> torch.Tensor:
        torch.manual_seed(self.seed)
        num_nodes = node2vec_emb.shape[0]

        # Augment with node features if available (stored externally)
        x = torch.tensor(node2vec_emb, dtype=torch.float)

        # Link embedding dimension: hadamard → same as node emb dim
        link_dim = self.embedding_dim

        # GraphSAGE encoder
        self._graphsage = GraphSAGE(
            in_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            out_dim=self.embedding_dim,
            num_layers=self.num_layers,
            aggregator=self.aggregator,
            dropout=self.dropout,
        )

        # Link classifier
        self._classifier = LinkClassifier(in_dim=link_dim, dropout=self.dropout)

        # Choose optimizer (Algorithm 1, line 14)
        params = (list(self._graphsage.parameters()) +
                  list(self._classifier.parameters()))
        if self.optimizer_name == "adam":
            opt = torch.optim.Adam(params, lr=self.lr, weight_decay=1e-5)
        elif self.optimizer_name == "adadelta":
            opt = torch.optim.Adadelta(params, lr=1.0, rho=0.95, eps=1e-6)
        else:
            raise ValueError(f"Unknown optimizer '{self.optimizer_name}'")

        train_edges_t  = train_edges
        train_labels_t = torch.tensor(train_labels, dtype=torch.float)

        self._graphsage.train()
        self._classifier.train()

        for epoch in range(1, self.epochs + 1):
            opt.zero_grad()

            h = self._graphsage(x, edge_index)          # [N, emb_dim]
            link_emb = self._graphsage.get_link_embedding(h, train_edges_t)
            preds = self._classifier(link_emb)           # [E_train]

            loss = F.binary_cross_entropy(preds, train_labels_t)
            loss.backward()
            opt.step()

            if epoch % 50 == 0 or epoch == 1:
                print(f"      epoch {epoch:3d}/{self.epochs} | loss={loss.item():.4f}")

        # Return final node embeddings (inference mode)
        self._graphsage.eval()
        self._classifier.eval()
        with torch.no_grad():
            h_final = self._graphsage(x, edge_index)
        return h_final

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        G_train      : nx.Graph,
        edge_index   : torch.Tensor,
        train_edges  : list,
        train_labels : list,
        node_features: np.ndarray | None = None,
    ) -> "LPNE2GGCN":
        """
        Train the full LPNE2GGCN pipeline.

        Parameters
        ----------
        G_train       : nx.Graph  — training graph (edges withheld)
        edge_index    : [2, E]    — training edge index for GNN
        train_edges   : list of (u,v) — labelled training pairs
        train_labels  : list of int   — 0/1 labels
        node_features : [N, F]        — optional raw node features to
                                        concatenate with Node2Vec embs

        Returns
        -------
        self
        """
        t0 = time.time()

        print("  [LPNE2GGCN] Phase 1&2: Node2Vec embedding ...")
        n2v_emb = self._fit_node2vec(G_train)     # [N, embedding_dim]

        # Optionally concatenate raw node features (Eq. 8 + GCN)
        if node_features is not None:
            n = min(n2v_emb.shape[0], node_features.shape[0])
            combined = np.concatenate(
                [n2v_emb[:n], node_features[:n]], axis=1
            )
            # Update embedding dim for GraphSAGE input
            self._effective_in_dim = combined.shape[1]
            n2v_emb_in = combined
        else:
            self._effective_in_dim = self.embedding_dim
            n2v_emb_in = n2v_emb

        # Rebuild GraphSAGE with possibly wider input
        self._graphsage = GraphSAGE(
            in_dim    = self._effective_in_dim,
            hidden_dim= self.hidden_dim,
            out_dim   = self.embedding_dim,
            num_layers= self.num_layers,
            aggregator= self.aggregator,
            dropout   = self.dropout,
        )
        self._classifier = LinkClassifier(
            in_dim=self.embedding_dim, dropout=self.dropout
        )

        # Re-run with combined features
        print("  [LPNE2GGCN] Phase 3: GraphSAGE refinement ...")
        self._node2vec_emb = n2v_emb_in    # store for predict()
        self._edge_index   = edge_index

        torch.manual_seed(self.seed)
        x = torch.tensor(n2v_emb_in, dtype=torch.float)
        train_labels_t = torch.tensor(train_labels, dtype=torch.float)

        params = (list(self._graphsage.parameters()) +
                  list(self._classifier.parameters()))
        if self.optimizer_name == "adam":
            opt = torch.optim.Adam(params, lr=self.lr, weight_decay=1e-5)
        else:
            opt = torch.optim.Adadelta(params, lr=1.0, rho=0.95, eps=1e-6)

        self._graphsage.train()
        self._classifier.train()

        for epoch in range(1, self.epochs + 1):
            opt.zero_grad()
            h        = self._graphsage(x, edge_index)
            link_emb = self._graphsage.get_link_embedding(h, train_edges)
            preds    = self._classifier(link_emb)
            loss     = F.binary_cross_entropy(preds, train_labels_t)
            loss.backward()
            opt.step()
            if epoch % 50 == 0 or epoch == 1:
                print(f"      epoch {epoch:3d}/{self.epochs} | loss={loss.item():.4f}")

        self.training_time_ = time.time() - t0
        print(f"  [LPNE2GGCN] Training complete in {self.training_time_:.2f}s")
        return self

    def predict_proba(self, edges: list) -> np.ndarray:
        """
        Predict link probabilities for a list of (u, v) pairs.

        Returns
        -------
        np.ndarray [len(edges)]  — probabilities in [0, 1]
        """
        if self._graphsage is None:
            raise RuntimeError("Call fit() before predict_proba().")

        self._graphsage.eval()
        self._classifier.eval()
        x = torch.tensor(self._node2vec_emb, dtype=torch.float)

        with torch.no_grad():
            h        = self._graphsage(x, self._edge_index)
            link_emb = self._graphsage.get_link_embedding(h, edges)
            probs    = self._classifier(link_emb)

        return probs.numpy()

    def evaluate(self, test_edges: list, test_labels: list) -> dict:
        """
        Evaluate on test edge pairs.

        Returns
        -------
        dict with keys 'accuracy', 'f1', 'auc'
        """
        probs = self.predict_proba(test_edges)
        return compute_metrics(test_labels, probs)
