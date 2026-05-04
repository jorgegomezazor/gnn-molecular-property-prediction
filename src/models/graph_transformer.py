"""
src/models/graph_transformer.py

General, Powerful, Scalable (GPS) Graph Transformer
Reference: Rampášek et al., "Recipe for a General, Powerful, Scalable Graph Transformer",
           NeurIPS 2022.

Architecture:
  - Optional Laplacian Positional Encoding (PE)
  - Each GPS layer = local MPNN  +  global Multi-Head Self-Attention
  - Layer-norm + residual connections throughout
  - Mean readout + MLP head
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.utils import get_laplacian, to_dense_adj, subgraph, to_dense_batch
import numpy as np


# ──────────────────────────────────────────────────────────
# Positional Encoding
# ──────────────────────────────────────────────────────────

class LaplacianPE(nn.Module):
    """
    Laplacian Eigenvector Positional Encoding.

    Computes the k smallest non-trivial eigenvectors of the normalized
    graph Laplacian and projects them to pe_dim.
    """

    def __init__(self, k: int, pe_dim: int):
        super().__init__()
        self.k = k
        self.proj = nn.Linear(k, pe_dim)

    def forward(self, data):
        """Augment data.x with Laplacian PE. Returns updated data."""
        device = data.x.device
        batch = data.batch if (hasattr(data, 'batch') and data.batch is not None) \
                else torch.zeros(data.num_nodes, dtype=torch.long, device=device)
        B = int(batch.max().item()) + 1

        pe_parts = []
        for g in range(B):
            node_mask = (batch == g)
            node_idx  = node_mask.nonzero(as_tuple=True)[0]
            n = node_idx.shape[0]

            # Extract edges for this molecule and relabel to 0..n-1
            sub_ei, _ = subgraph(node_idx, data.edge_index,
                                  relabel_nodes=True, num_nodes=data.num_nodes)

            ei_lap, ew_lap = get_laplacian(sub_ei, normalization="sym", num_nodes=n)
            L = to_dense_adj(ei_lap, edge_attr=ew_lap,
                             max_num_nodes=n).squeeze(0).cpu().numpy().astype(np.float64)

            eigvecs = np.linalg.eigh(L)[1]          # sorted ascending
            k = min(self.k, n - 1)
            vecs = eigvecs[:, 1:k + 1]              # skip trivial eigvec
            if vecs.shape[1] < self.k:
                vecs = np.concatenate(
                    [vecs, np.zeros((n, self.k - vecs.shape[1]))], axis=1
                )
            pe_parts.append(torch.from_numpy(vecs).float())

        pe = torch.cat(pe_parts, dim=0).to(device)  # (N_total, k)
        pe = self.proj(pe)                            # (N_total, pe_dim)
        data.x = torch.cat([data.x.float(), pe], dim=-1)
        return data


# ──────────────────────────────────────────────────────────
# GPS Layer
# ──────────────────────────────────────────────────────────

class GPSLayer(nn.Module):
    """
    One GPS layer: local GIN  +  global Transformer attention.
    """

    def __init__(self, dim: int, num_heads: int,
                 dropout: float = 0.1, attn_dropout: float = 0.1):
        super().__init__()

        # Local: GIN
        mlp = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim)
        )
        self.local_conv = GINConv(mlp, train_eps=True)

        # Global: Multi-Head Self-Attention
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads,
            dropout=attn_dropout, batch_first=True
        )

        # Feed-forward
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch):
        # ── Local branch ──
        h_local = self.local_conv(x, edge_index)          # (N_total, dim)

        # ── Global branch ──
        # to_dense_batch pads variable-length graphs into (B, max_len, dim)
        x_dense, node_mask = to_dense_batch(x, batch)     # (B, max_len, dim), (B, max_len)
        key_pad = ~node_mask                               # True = ignore (padding positions)

        attn_out, _ = self.attn(x_dense, x_dense, x_dense,
                                key_padding_mask=key_pad)  # (B, max_len, dim)

        h_global = attn_out[node_mask]                    # (N_total, dim)

        # ── Combine ──
        h = self.norm1(x + self.dropout(h_local) + self.dropout(h_global))
        h = self.norm2(h + self.dropout(self.ff(h)))
        return h


# ──────────────────────────────────────────────────────────
# Full GPS Model
# ──────────────────────────────────────────────────────────

class GraphTransformer(nn.Module):
    """
    GPS Graph Transformer for molecular property regression.

    Args:
        node_feat_dim : Input node feature dimension.
        hidden_dim    : Internal representation dimension.
        num_layers    : Number of GPS layers.
        num_heads     : Transformer attention heads.
        dropout       : Dropout rate.
        attn_dropout  : Attention dropout rate.
        pe_type       : 'laplacian' or 'none'.
        pe_dim        : Laplacian PE output dimension (if used).
    """

    def __init__(
        self,
        node_feat_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        pe_type: str = "laplacian",
        pe_dim: int = 8,
        **kwargs,
    ):
        super().__init__()
        self.pe_type = pe_type

        # Positional encoding
        if pe_type == "laplacian":
            self.pe_encoder = LaplacianPE(k=pe_dim, pe_dim=pe_dim)
            in_dim = node_feat_dim + pe_dim
        else:
            self.pe_encoder = None
            in_dim = node_feat_dim

        self.input_proj = nn.Linear(in_dim, hidden_dim)

        # GPS layers
        self.layers = nn.ModuleList([
            GPSLayer(hidden_dim, num_heads, dropout, attn_dropout)
            for _ in range(num_layers)
        ])

        self.pool = global_mean_pool

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data):
        if self.pe_encoder is not None:
            data = self.pe_encoder(data)

        x = self.input_proj(data.x.float())
        edge_index = data.edge_index
        batch = data.batch

        for layer in self.layers:
            x = layer(x, edge_index, batch)

        graph_repr = self.pool(x, batch)
        return self.head(graph_repr).squeeze(-1)
