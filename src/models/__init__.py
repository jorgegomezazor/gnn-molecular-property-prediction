from .mpnn import MPNN
from .gin import GIN
from .gat import GAT
from .graph_transformer import GraphTransformer


MODEL_REGISTRY = {
    "mpnn": MPNN,
    "gin": GIN,
    "gat": GAT,
    "graph_transformer": GraphTransformer,
}


def build_model(cfg: dict, node_feat_dim: int, edge_feat_dim: int):
    name = cfg["name"]
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}"
        )
    cls = MODEL_REGISTRY[name]

    kwargs = dict(
        node_feat_dim=node_feat_dim,
        edge_feat_dim=edge_feat_dim,
        hidden_dim=cfg.get("hidden_dim", 64),
        num_layers=cfg.get("num_layers", 3),
        dropout=cfg.get("dropout", 0.0),
    )

    if name == "mpnn":
        kwargs.update(
            readout=cfg.get("readout", "set2set"),
            set2set_steps=cfg.get("set2set_steps", 3),
        )
    elif name == "gin":
        kwargs.update(
            eps=cfg.get("eps", 0.0),
            train_eps=cfg.get("train_eps", True),
            readout=cfg.get("readout", "sum"),
            mlp_layers=cfg.get("mlp_layers", 2),
            batch_norm=cfg.get("batch_norm", True),
        )
    elif name == "gat":
        kwargs.update(
            num_heads=cfg.get("num_heads", 4),
            concat_heads=cfg.get("concat_heads", True),
            attention_dropout=cfg.get("attention_dropout", 0.1),
            batch_norm=cfg.get("batch_norm", True),
        )
    elif name == "graph_transformer":
        kwargs.update(
            num_heads=cfg.get("num_heads", 4),
            attn_dropout=cfg.get("attn_dropout", 0.1),
            pe_type=cfg.get("pe_type", "laplacian"),
            pe_dim=cfg.get("pe_dim", 8),
        )

    # GIN/GAT/Transformer ignore edge attributes here
    if name in ("gin", "gat", "graph_transformer"):
        kwargs.pop("edge_feat_dim", None)

    return cls(**kwargs)
