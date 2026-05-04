"""
src/models/mpnn.py

Neural Message Passing Network (MPNN)
Reference: Gilmer et al., "Neural Message Passing for Quantum Chemistry", ICML 2017.

Architecture:
  - Edge network: maps edge features → message weight matrices
  - Message passing steps with GRU-based node update
  - Set2Set readout for graph-level representation
  - MLP head for regression
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv, Set2Set, global_mean_pool


class MPNN(nn.Module):
    """
    Message Passing Neural Network for molecular property regression.

    Args:
        node_feat_dim  : Dimension of input node features.
        edge_feat_dim  : Dimension of input edge features.
        hidden_dim     : Hidden representation size.
        num_layers     : Number of message-passing steps (T in the paper).
        readout        : Aggregation method — 'set2set' or 'mean'.
        set2set_steps  : Number of Set2Set processing steps.
        dropout        : Dropout probability.
    """

    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        readout: str = "set2set",
        set2set_steps: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.readout_type = readout
        self.dropout = dropout

        # Node embedding: project raw features → hidden_dim
        self.node_embed = nn.Linear(node_feat_dim, hidden_dim)

        # Edge network: maps edge_feat → (hidden_dim × hidden_dim) matrix
        # used as the weight matrix for NNConv
        edge_net = nn.Sequential(
            nn.Linear(edge_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * hidden_dim),
        )
        self.conv = NNConv(hidden_dim, hidden_dim, edge_net, aggr="add")

        # GRU cell for node state update (as in the original paper)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=False)

        # Readout
        if readout == "set2set":
            self.readout_layer = Set2Set(hidden_dim, processing_steps=set2set_steps)
            readout_out_dim = 2 * hidden_dim
        else:
            self.readout_layer = global_mean_pool
            readout_out_dim = hidden_dim

        # Regression head
        self.head = nn.Sequential(
            nn.Linear(readout_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x.float(),
            data.edge_index,
            data.edge_attr.float(),
            data.batch,
        )

        # Initial node embedding
        h = F.relu(self.node_embed(x))          # (N, hidden_dim)

        # GRU hidden state: shape (1, N, hidden_dim)
        h_gru = h.unsqueeze(0)

        # Message passing steps
        for _ in range(self.num_layers):
            m = F.relu(self.conv(h, edge_index, edge_attr))  # messages
            h, h_gru = self.gru(m.unsqueeze(0), h_gru)
            h = h.squeeze(0)

        # Readout
        if self.readout_type == "set2set":
            out = self.readout_layer(h, batch)
        else:
            out = self.readout_layer(h, batch)

        return self.head(out).squeeze(-1)
