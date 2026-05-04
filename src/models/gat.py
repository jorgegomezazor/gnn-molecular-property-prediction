"""
src/models/gat.py

Graph Attention Network (GATv2)
Reference: Veličković et al., "Graph Attention Networks", ICLR 2018.
           Brody et al., "How Attentive are Graph Attention Networks?", ICLR 2022 (GATv2).

Architecture:
  - GATv2Conv layers with multi-head attention
  - Optional BatchNorm after each layer
  - Mean/max readout
  - MLP regression head

Note: We use GATv2 which fixes the static attention issue in the original GAT.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool


class GAT(nn.Module):
    """
    Graph Attention Network (GATv2) for molecular property regression.

    Args:
        node_feat_dim   : Input node feature dimension.
        hidden_dim      : Hidden dimension per head.
        num_layers      : Number of GATv2 layers.
        num_heads       : Number of attention heads.
        concat_heads    : If True, concatenate heads; otherwise average.
        dropout         : Dropout on node features.
        attention_dropout: Dropout on attention coefficients.
        readout         : Graph-level pooling: 'mean'.
        batch_norm      : Use BatchNorm after each layer.
    """

    def __init__(
        self,
        node_feat_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 4,
        num_heads: int = 4,
        concat_heads: bool = True,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        readout: str = "mean",
        batch_norm: bool = True,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout    = dropout

        # Dimension after each layer (concat or average)
        out_per_layer = hidden_dim * num_heads if concat_heads else hidden_dim

        # Input projection
        self.node_embed = nn.Linear(node_feat_dim, hidden_dim * num_heads
                                    if concat_heads else hidden_dim)

        # GAT layers
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        for i in range(num_layers):
            in_dim = (hidden_dim * num_heads if concat_heads else hidden_dim)
            self.convs.append(
                GATv2Conv(
                    in_channels=in_dim,
                    out_channels=hidden_dim,
                    heads=num_heads,
                    concat=concat_heads,
                    dropout=attention_dropout,
                )
            )
            self.bns.append(
                nn.BatchNorm1d(out_per_layer) if batch_norm else nn.Identity()
            )

        self.pool = global_mean_pool

        # Head
        self.head = nn.Sequential(
            nn.Linear(out_per_layer, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data):
        x, edge_index, batch = (
            data.x.float(),
            data.edge_index,
            data.batch,
        )

        h = F.relu(self.node_embed(x))

        for conv, bn in zip(self.convs, self.bns):
            h = conv(h, edge_index)
            h = bn(h)
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        graph_repr = self.pool(h, batch)
        return self.head(graph_repr).squeeze(-1)
