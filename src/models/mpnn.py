"""MPNN (Gilmer et al., ICML 2017): edge-conditioned message passing with GRU
node updates and Set2Set readout."""

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv, Set2Set, global_mean_pool


class MPNN(nn.Module):

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

        self.node_embed = nn.Linear(node_feat_dim, hidden_dim)

        # Edge network maps edge features to a hidden_dim x hidden_dim weight
        # matrix used by NNConv.
        edge_net = nn.Sequential(
            nn.Linear(edge_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * hidden_dim),
        )
        self.conv = NNConv(hidden_dim, hidden_dim, edge_net, aggr="add")

        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=False)

        if readout == "set2set":
            self.readout_layer = Set2Set(hidden_dim, processing_steps=set2set_steps)
            readout_out_dim = 2 * hidden_dim
        else:
            self.readout_layer = global_mean_pool
            readout_out_dim = hidden_dim

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

        h = F.relu(self.node_embed(x))
        h_gru = h.unsqueeze(0)

        for _ in range(self.num_layers):
            m = F.relu(self.conv(h, edge_index, edge_attr))
            h, h_gru = self.gru(m.unsqueeze(0), h_gru)
            h = h.squeeze(0)

        out = self.readout_layer(h, batch)
        return self.head(out).squeeze(-1)
