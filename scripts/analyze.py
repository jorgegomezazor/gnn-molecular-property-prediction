"""
scripts/analyze.py

Post-training analysis:
  1. Aggregates test results from all run directories into a summary table.
  2. Generates the model-comparison bar chart (for the report).
  3. Generates t-SNE of graph-level embeddings (requires trained checkpoints).

Usage:
    python scripts/analyze.py --results_dir results/ --seed 42
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.train import load_config, set_seed
from src.data.dataset import QM9Dataset
from src.models import build_model
from src.utils.visualization import plot_model_comparison, plot_tsne_embeddings


MODELS = ["mpnn", "gin", "gat", "graph_transformer"]


def load_summary(results_dir: str, model: str, seed: int) -> dict | None:
    run_name = f"{model}_seed{seed}"
    path = os.path.join(results_dir, run_name, "summary.json")
    if not os.path.exists(path):
        print(f"[Analyze] Warning: no summary found for {run_name} — skipping.")
        return None
    with open(path) as f:
        return json.load(f)


def extract_embeddings(model, loader, device: str) -> tuple:
    """Run forward pass and return (embeddings, targets) as numpy arrays."""
    model.eval()
    all_emb, all_tgt = [], []

    # Hook to capture graph-level representation before the final head
    # We monkey-patch the model's head to also return the pre-head repr.
    embeddings_buffer = {}

    def hook_fn(module, inp, out):
        embeddings_buffer["emb"] = inp[0].detach().cpu()

    # Attach hook to first layer of head (Linear)
    hook = model.head[0].register_forward_hook(hook_fn)

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            _ = model(batch)
            all_emb.append(embeddings_buffer["emb"])
            all_tgt.append(batch.y.squeeze(-1).cpu())

    hook.remove()
    return (
        torch.cat(all_emb).numpy(),
        torch.cat(all_tgt).numpy(),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results/")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tsne", action="store_true",
                        help="Generate t-SNE plots (slow).")
    args = parser.parse_args()

    set_seed(args.seed)
    fig_dir = os.path.join(args.results_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # ── 1. Aggregate results ──
    rows    = []
    results = {}
    for model_name in MODELS:
        summary = load_summary(args.results_dir, model_name, args.seed)
        if summary is None:
            continue
        m = summary["test_metrics"]
        rows.append({
            "model": model_name,
            "MAE":   m["mae"],
            "RMSE":  m["rmse"],
        })
        results[model_name] = m

    df = pd.DataFrame(rows)
    df_sorted = df.sort_values("MAE")
    print("\n" + "="*50)
    print("  MODEL COMPARISON — Test MAE")
    print("="*50)
    print(df_sorted.to_string(index=False))
    print("="*50 + "\n")

    csv_path = os.path.join(args.results_dir, "comparison_table.csv")
    df_sorted.to_csv(csv_path, index=False)
    print(f"[Analyze] Comparison table saved to {csv_path}")

    # ── 2. Comparison bar chart ──
    if results:
        # Attempt to get target unit from any config
        unit = ""
        try:
            cfg = load_config(f"configs/{list(results.keys())[0]}.yaml")
            unit = cfg["dataset"].get("target_unit", "")
        except Exception:
            pass
        plot_model_comparison(results, save_dir=fig_dir, unit=unit, metric="mae")
        plot_model_comparison(results, save_dir=fig_dir, unit=unit, metric="rmse")

    # ── 3. t-SNE embeddings (optional, slow) ──
    if args.tsne:
        print("[Analyze] Generating t-SNE embeddings …")
        for model_name in MODELS:
            run_name = f"{model_name}_seed{args.seed}"
            ckpt = os.path.join(args.results_dir, run_name, "best_model.pt")
            if not os.path.exists(ckpt):
                print(f"[Analyze] No checkpoint for {run_name} — skipping t-SNE.")
                continue

            cfg     = load_config(f"configs/{model_name}.yaml")
            dataset = QM9Dataset(cfg["dataset"])
            _, _, test_loader = dataset.get_loaders(batch_size=64, num_workers=0)

            model = build_model(cfg["model"],
                                node_feat_dim=dataset.num_node_features,
                                edge_feat_dim=dataset.num_edge_features)
            model.load_state_dict(torch.load(ckpt, map_location=args.device))
            model = model.to(args.device)

            emb, tgt = extract_embeddings(model, test_loader, args.device)
            plot_tsne_embeddings(
                emb, tgt,
                label_name=cfg["dataset"].get("target_name", "target"),
                run_name=run_name,
                save_dir=os.path.join(args.results_dir, run_name, "figures"),
            )

    print(f"\n[Analyze] All figures saved to {fig_dir}")


if __name__ == "__main__":
    main()
