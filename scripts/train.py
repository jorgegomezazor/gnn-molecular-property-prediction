"""Train a single model on QM9.

Usage:
    python scripts/train.py --config configs/gin.yaml [--seed 42] [--device cuda]
"""

import os
import sys
import argparse
import random

import yaml
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import QM9Dataset
from src.models import build_model
from src.training.trainer import Trainer
from src.training.evaluate import evaluate
from src.utils.logger import ExperimentLogger
from src.utils.visualization import plot_learning_curves, plot_scatter


def deep_merge(base: dict, override: dict) -> dict:
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: str) -> dict:
    """Load YAML config, expanding any 'defaults: [...]' entries first."""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if "defaults" in cfg:
        merged = {}
        for default in cfg.pop("defaults"):
            default_path = os.path.join("configs", f"{default}.yaml")
            with open(default_path, encoding="utf-8") as f:
                base = yaml.safe_load(f)
            deep_merge(merged, base)
        deep_merge(merged, cfg)
        return merged
    return cfg


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Train a GNN on QM9.")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config (e.g. configs/gin.yaml)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override random seed from config.")
    parser.add_argument("--device", type=str, default=None,
                        help="Override device: 'cuda' or 'cpu'.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["training"]["seed"] = args.seed
        cfg["dataset"]["seed"] = args.seed
    if args.device is not None:
        cfg["device"] = args.device

    seed = cfg["training"].get("seed", 42)
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)

    model_name = cfg["model"]["name"]
    run_name = f"{model_name}_seed{seed}"

    print(f"\n{'='*60}")
    print(f"  Model  : {model_name.upper()}")
    print(f"  Device : {device}")
    print(f"  Seed   : {seed}")
    print(f"  Run    : {run_name}")
    print(f"{'='*60}\n")

    dataset = QM9Dataset(cfg["dataset"])
    train_loader, val_loader, test_loader = dataset.get_loaders(
        batch_size=cfg["training"]["batch_size"],
        num_workers=0,
        pin_memory=(device == "cuda"),
    )

    model = build_model(
        cfg["model"],
        node_feat_dim=dataset.num_node_features,
        edge_feat_dim=dataset.num_edge_features,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Trainable parameters: {n_params:,}")

    trainer = Trainer(model, dataset, cfg, device=device, run_name=run_name)
    history = trainer.fit(train_loader, val_loader)
    trainer.load_best()

    log_dir = cfg["logging"]["log_dir"]
    unit = cfg["dataset"].get("target_unit", "")
    results = evaluate(model, test_loader, dataset, device, run_name, log_dir)

    logger = ExperimentLogger(log_dir, run_name, cfg)
    for epoch, (tl, vm, vr) in enumerate(
        zip(history["train_loss"], history["val_mae"], history["val_rmse"]), 1
    ):
        logger.log_epoch(epoch, {"train_loss": tl, "val_mae": vm, "val_rmse": vr})
    logger.save_summary(results)

    fig_dir = os.path.join(log_dir, run_name, "figures")
    plot_learning_curves(history, run_name, fig_dir, unit=unit)

    # Predictions on the test set, for the scatter plot
    model.eval()
    all_pred, all_target = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred = dataset.denormalize(model(batch))
            all_pred.append(pred.cpu())
            all_target.append(batch.y.squeeze(-1).cpu())
    plot_scatter(torch.cat(all_pred), torch.cat(all_target),
                 run_name, fig_dir, unit=unit)

    print(f"\n[Done] Results and figures saved to {os.path.join(log_dir, run_name)}/")


if __name__ == "__main__":
    main()
