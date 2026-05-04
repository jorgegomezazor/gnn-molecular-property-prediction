"""GIN (Xu et al., ICLR 2019) with Jumping-Knowledge concatenation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool


def build_mlp(in_dim: int, hidden_dim: int, num_layers: int,
              batch_norm: bool = True) -> nn.Module:
    layers = []
    for i in range(num_layers):
        d_in = in_dim if i == 0 else hidden_dim
        layers.append(nn.Linear(d_in, hidden_dim))
        if batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class GIN(nn.Module):

    def __init__(
        self,
        node_feat_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 5,
        eps: float = 0.0,
        train_eps: bool = True,
        readout: str = "sum",
        mlp_layers: int = 2,
        dropout: float = 0.0,
        batch_norm: bool = True,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.readout_type = readout

        self.node_embed = nn.Linear(node_feat_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            mlp = build_mlp(hidden_dim, hidden_dim, mlp_layers, batch_norm)
            self.convs.append(GINConv(mlp, eps=eps, train_eps=train_eps))
            self.bns.append(nn.BatchNorm1d(hidden_dim) if batch_norm else nn.Identity())

        self.pool = global_add_pool if readout == "sum" else global_mean_pool

        # Head consumes the concatenation of per-layer graph reps (JK).
        head_in = hidden_dim * (num_layers + 1)
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden_dim),
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
        layer_outs = [self.pool(h, batch)]

        for conv, bn in zip(self.convs, self.bns):
            h = conv(h, edge_index)
            h = bn(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            layer_outs.append(self.pool(h, batch))

        graph_repr = torch.cat(layer_outs, dim=-1)
        return self.head(graph_repr).squeeze(-1)
