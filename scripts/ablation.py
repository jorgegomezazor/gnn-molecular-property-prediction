"""Ablation sweeps over depth and/or hidden_dim for a chosen model.

Usage:
    python scripts/ablation.py --model gin --sweep depth
    python scripts/ablation.py --model gin --sweep hidden_dim
    python scripts/ablation.py --model gin --sweep both
"""

import os
import sys
import argparse
import copy

import torch
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.train import load_config, set_seed
from src.data.dataset import QM9Dataset
from src.models import build_model
from src.training.trainer import Trainer
from src.training.evaluate import evaluate
from src.utils.visualization import plot_ablation_heatmap


DEPTH_VALUES = [1, 2, 3, 4, 5, 6]
HIDDEN_DIM_VALUES = [32, 64, 128, 256]


def run_single(cfg: dict, device: str, run_tag: str) -> dict:
    seed = cfg["training"].get("seed", 42)
    set_seed(seed)

    dataset = QM9Dataset(cfg["dataset"])
    train_loader, val_loader, test_loader = dataset.get_loaders(
        batch_size=cfg["training"]["batch_size"], num_workers=0
    )

    model = build_model(cfg["model"],
                        node_feat_dim=dataset.num_node_features,
                        edge_feat_dim=dataset.num_edge_features)

    trainer = Trainer(model, dataset, cfg, device=device, run_name=run_tag)
    trainer.fit(train_loader, val_loader)
    trainer.load_best()

    return evaluate(model, test_loader, dataset, device, run_tag,
                    cfg["logging"]["log_dir"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gin",
                        choices=["mpnn", "gin", "gat", "graph_transformer"])
    parser.add_argument("--sweep", type=str, default="depth",
                        choices=["depth", "hidden_dim", "both"])
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    base_cfg = load_config(f"configs/{args.model}.yaml")
    base_cfg["training"]["seed"] = args.seed
    # Shorter runs for the ablation
    base_cfg["training"]["epochs"] = 100
    base_cfg["training"]["early_stopping_patience"] = 15

    records = []

    if args.sweep in ("depth", "both"):
        dim_vals = HIDDEN_DIM_VALUES if args.sweep == "both" else [base_cfg["model"]["hidden_dim"]]
        for d in DEPTH_VALUES:
            for h in dim_vals:
                cfg = copy.deepcopy(base_cfg)
                cfg["model"]["num_layers"] = d
                cfg["model"]["hidden_dim"] = h
                tag = f"ablation_{args.model}_L{d}_H{h}"
                print(f"\n[Ablation] num_layers={d}, hidden_dim={h} -> {tag}")
                res = run_single(cfg, args.device, tag)
                records.append({"num_layers": d, "hidden_dim": h,
                                "mae": res["mae"], "rmse": res["rmse"]})

    elif args.sweep == "hidden_dim":
        for h in HIDDEN_DIM_VALUES:
            cfg = copy.deepcopy(base_cfg)
            cfg["model"]["hidden_dim"] = h
            tag = f"ablation_{args.model}_H{h}"
            print(f"\n[Ablation] hidden_dim={h} -> {tag}")
            res = run_single(cfg, args.device, tag)
            records.append({"num_layers": cfg["model"]["num_layers"],
                            "hidden_dim": h,
                            "mae": res["mae"], "rmse": res["rmse"]})

    os.makedirs("results", exist_ok=True)
    df = pd.DataFrame(records)
    csv_path = f"results/ablation_{args.model}_{args.sweep}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[Ablation] Results saved to {csv_path}")

    if args.sweep == "both":
        plot_ablation_heatmap(
            df, x_col="num_layers", y_col="hidden_dim", val_col="mae",
            title=f"{args.model.upper()} Ablation - Test MAE",
            save_dir="results/figures"
        )
    elif args.sweep == "depth":
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(df["num_layers"], df["mae"], marker="o")
        ax.set_xlabel("num_layers")
        ax.set_ylabel("Test MAE")
        ax.set_title(f"{args.model.upper()} - Depth Ablation")
        os.makedirs("results/figures", exist_ok=True)
        fig.savefig(f"results/figures/ablation_{args.model}_depth.pdf",
                    bbox_inches="tight")
        plt.close()
    elif args.sweep == "hidden_dim":
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(df["hidden_dim"], df["mae"], marker="s", color="#F44336")
        ax.set_xlabel("hidden_dim")
        ax.set_ylabel("Test MAE")
        ax.set_title(f"{args.model.upper()} - Width Ablation")
        os.makedirs("results/figures", exist_ok=True)
        fig.savefig(f"results/figures/ablation_{args.model}_hidden_dim.pdf",
                    bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    main()
