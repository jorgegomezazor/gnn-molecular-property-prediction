"""
src/utils/visualization.py

Plotting utilities for the report:
  1. Learning curves (train loss + val MAE over epochs)
  2. Prediction scatter plot (pred vs. true)
  3. Model comparison bar chart (MAE across models)
  4. Attention weight visualization (for GAT)
  5. t-SNE of molecular embeddings
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List, Optional


# ── Matplotlib style ───────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
})
COLORS = ["#2196F3", "#F44336", "#4CAF50", "#FF9800"]


# ── 1. Learning Curves ─────────────────────────────────────

def plot_learning_curves(history: dict, run_name: str,
                         save_dir: str, unit: str = ""):
    """
    Plot train loss and validation MAE on the same figure.

    Args:
        history  : dict with 'train_loss', 'val_mae', 'val_rmse' lists.
        run_name : Title suffix.
        save_dir : Directory to save the figure.
        unit     : Target unit label (e.g. 'D' for Debye).
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(epochs, history["train_loss"], color=COLORS[0])
    ax1.set_title("Training Loss (MAE, normalized)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")

    ax2.plot(epochs, history["val_mae"], color=COLORS[1])
    ax2.set_title(f"Validation MAE [{unit}]")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel(f"MAE [{unit}]")

    fig.suptitle(f"Learning Curves — {run_name}", fontweight="bold")
    fig.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{run_name}_learning_curves.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Viz] Saved learning curves → {path}")


# ── 2. Prediction Scatter ──────────────────────────────────

def plot_scatter(preds: torch.Tensor, targets: torch.Tensor,
                 run_name: str, save_dir: str, unit: str = ""):
    """Scatter plot of predicted vs. true values."""
    p = preds.numpy()
    t = targets.numpy()

    lim = [min(p.min(), t.min()) * 0.95, max(p.max(), t.max()) * 1.05]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(t, p, alpha=0.3, s=8, color=COLORS[0], rasterized=True)
    ax.plot(lim, lim, "k--", linewidth=1, label="y = x")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(f"True [{unit}]")
    ax.set_ylabel(f"Predicted [{unit}]")
    ax.set_title(f"Predictions vs. Ground Truth — {run_name}")
    ax.legend()
    fig.tight_layout()

    path = os.path.join(save_dir, f"{run_name}_scatter.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Viz] Saved scatter plot → {path}")


# ── 3. Model Comparison Bar Chart ─────────────────────────

def plot_model_comparison(results: Dict[str, dict], save_dir: str,
                          unit: str = "", metric: str = "mae"):
    """
    Bar chart comparing MAE (or RMSE) across models.

    Args:
        results : {model_name: {"mae": float, "rmse": float}}
    """
    names  = list(results.keys())
    values = [results[n][metric] for n in names]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(names, values, color=COLORS[:len(names)], edgecolor="white",
                  width=0.5)
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=10)
    ax.set_ylabel(f"{metric.upper()} [{unit}]")
    ax.set_title(f"Test {metric.upper()} Comparison", fontweight="bold")
    fig.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"model_comparison_{metric}.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Viz] Saved model comparison → {path}")


# ── 4. Ablation Heatmap ────────────────────────────────────

def plot_ablation_heatmap(ablation_df: pd.DataFrame, x_col: str,
                          y_col: str, val_col: str,
                          title: str, save_dir: str):
    """
    Heatmap for ablation results (e.g., depth × hidden_dim → MAE).
    """
    pivot = ablation_df.pivot(index=y_col, columns=x_col, values=val_col)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(pivot, annot=True, fmt=".4f", cmap="YlOrRd_r",
                linewidths=0.5, ax=ax)
    ax.set_title(title, fontweight="bold")
    fig.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"ablation_{title.replace(' ', '_')}.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Viz] Saved ablation heatmap → {path}")


# ── 5. t-SNE of molecular embeddings ──────────────────────

def plot_tsne_embeddings(embeddings: np.ndarray, labels: np.ndarray,
                         label_name: str, run_name: str, save_dir: str):
    """
    t-SNE of graph-level embeddings coloured by a molecular property.

    Args:
        embeddings : (N, d) numpy array of graph representations.
        labels     : (N,) numpy array of property values (for colouring).
        label_name : Name of the property (axis label).
    """
    from sklearn.manifold import TSNE

    print("[Viz] Running t-SNE (this may take a moment) …")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    z = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(z[:, 0], z[:, 1], c=labels, cmap="viridis",
                    alpha=0.5, s=8, rasterized=True)
    plt.colorbar(sc, ax=ax, label=label_name)
    ax.set_title(f"t-SNE of Learned Embeddings — {run_name}", fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{run_name}_tsne.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Viz] Saved t-SNE plot → {path}")
