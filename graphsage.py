"""
graphsage.py
------------
Generalized Graph Convolutional Network (GraphSAGE) implementation
following Section III-C and III-D of the LPNE2GGCN paper.

Implements three aggregator types (Section III-D):
  1. Mean Aggregator   — Eq. 6
  2. LSTM Aggregator   — Section III-D bullet 2
  3. Pooling Aggregator — Eq. 7

Forward propagation follows Eqs. 4 & 5:
  h^k_{N(v)} = AGGREGATE_k({h^{k-1}_u : ∀u ∈ N_v})
  h^k_v      = σ(W_k · COMBINE(h^{k-1}_v, h^{k-1}_{N(v)}))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv
from torch_geometric.utils import add_self_loops


# ---------------------------------------------------------------------------
# Individual aggregator layers
# ---------------------------------------------------------------------------

class MeanAggregatorLayer(nn.Module):
    """
    Mean aggregator (Eq. 6):
        h^(k)_v = σ(W · MEAN({h^{k-1}_v} ∪ {h^{k-1}_u, ∀u ∈ N_v}))
    Wraps torch_geometric SAGEConv in 'mean' mode.
    """

    def __init__(self, in_dim: int, out_dim: int, normalize: bool = True):
        super().__init__()
        self.conv = SAGEConv(in_dim, out_dim, aggr="mean", normalize=normalize)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.conv(x, edge_index)


class LSTMAggregatorLayer(nn.Module):
    """
    LSTM aggregator (Section III-D, bullet 2).
    Neighbours are randomly permuted to handle the lack of inherent order.
    Wraps torch_geometric SAGEConv with LSTM aggregation.
    """

    def __init__(self, in_dim: int, out_dim: int, normalize: bool = True):
        super().__init__()
        self.conv = SAGEConv(in_dim, out_dim, aggr="lstm", normalize=normalize)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.conv(x, edge_index)


class PoolingAggregatorLayer(nn.Module):
    """
    Max-pooling aggregator (Eq. 7):
        Aggregate^(pool)_k = max(σ(W_pool · h^k_{u_i}), ∀u_i ∈ N_v)
    """

    def __init__(self, in_dim: int, out_dim: int, normalize: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.conv   = SAGEConv(in_dim, out_dim, aggr="max", normalize=normalize)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.conv(x, edge_index)


# ---------------------------------------------------------------------------
# Multi-layer GraphSAGE encoder
# ---------------------------------------------------------------------------

class GraphSAGE(nn.Module):
    """
    GraphSAGE encoder with K layers.

    Architecture (Algorithm 1, lines 10-12 of the paper):
      Input  →  [Layer 1] → ReLU → Dropout
             →  [Layer 2] → ReLU → Dropout
             →  …
             →  [Layer K] → output node embeddings

    The paper uses K=3 (Table 6 shows 3-layer is optimal).

    Parameters
    ----------
    in_dim      : input feature dimension
    hidden_dim  : hidden layer size
    out_dim     : output embedding dimension
    num_layers  : number of GraphSAGE layers (K)
    aggregator  : 'mean' | 'lstm' | 'pool'
    dropout     : dropout probability
    """

    _AGGREGATORS = {
        "mean": MeanAggregatorLayer,
        "lstm": LSTMAggregatorLayer,
        "pool": PoolingAggregatorLayer,
    }

    def __init__(
        self,
        in_dim     : int,
        hidden_dim : int = 128,
        out_dim    : int = 64,
        num_layers : int = 3,
        aggregator : str = "mean",
        dropout    : float = 0.3,
    ):
        super().__init__()
        if aggregator not in self._AGGREGATORS:
            raise ValueError(
                f"aggregator must be one of {list(self._AGGREGATORS)}, "
                f"got '{aggregator}'"
            )
        AggLayer = self._AGGREGATORS[aggregator]

        self.layers  = nn.ModuleList()
        self.dropout = dropout

        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        for i in range(num_layers):
            self.layers.append(AggLayer(dims[i], dims[i + 1]))

        self.num_layers = num_layers

    def forward(
        self,
        x          : torch.Tensor,
        edge_index : torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass — Eqs. 4 & 5.

        Parameters
        ----------
        x          : [N, in_dim]  node feature matrix
        edge_index : [2, E]       edge list (COO format)

        Returns
        -------
        h : [N, out_dim]  node embedding matrix
        """
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h, edge_index)
            if i < self.num_layers - 1:   # no activation on last layer
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    def get_link_embedding(
        self,
        h      : torch.Tensor,
        edges  : list,
        operator: str = "hadamard",
    ) -> torch.Tensor:
        """
        Construct link representations by combining node embeddings.

        Parameters
        ----------
        h        : [N, D]  node embeddings from forward()
        edges    : list of (u, v) tuples
        operator : 'hadamard' | 'average' | 'l1' | 'l2' | 'concat'

        Returns
        -------
        link_emb : [len(edges), D] or [len(edges), 2D] for concat
        """
        us = torch.tensor([e[0] for e in edges], dtype=torch.long)
        vs = torch.tensor([e[1] for e in edges], dtype=torch.long)

        hu = h[us]   # [E, D]
        hv = h[vs]   # [E, D]

        if operator == "hadamard":
            return hu * hv
        elif operator == "average":
            return (hu + hv) / 2
        elif operator == "l1":
            return torch.abs(hu - hv)
        elif operator == "l2":
            return (hu - hv) ** 2
        elif operator == "concat":
            return torch.cat([hu, hv], dim=-1)
        else:
            raise ValueError(f"Unknown operator '{operator}'")


# ---------------------------------------------------------------------------
# Graph Attention Network (GAT) — baseline
# ---------------------------------------------------------------------------

class GATBaseline(nn.Module):
    """
    Graph Attention Network (Veličković et al., 2018) — baseline model.
    Used for the GAT row in Tables 3–5.
    """

    def __init__(
        self,
        in_dim    : int,
        hidden_dim: int = 64,
        out_dim   : int = 32,
        heads     : int = 4,
        dropout   : float = 0.3,
    ):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_dim * heads, out_dim, heads=1,
                             concat=False, dropout=dropout)
        self.dropout = dropout

    def forward(
        self,
        x         : torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x
