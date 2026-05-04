"""
src/utils/metrics.py

Standard regression metrics for molecular property prediction.
All functions operate on torch.Tensor or numpy arrays.
"""

import torch
import numpy as np


def compute_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean Absolute Error."""
    return (pred - target).abs().mean().item()


def compute_rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Root Mean Squared Error."""
    return ((pred - target) ** 2).mean().sqrt().item()


def compute_r2(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Coefficient of determination R²."""
    ss_res = ((target - pred) ** 2).sum()
    ss_tot = ((target - target.mean()) ** 2).sum()
    return 1.0 - (ss_res / (ss_tot + 1e-8)).item()


def summarize_metrics(pred: torch.Tensor, target: torch.Tensor,
                      unit: str = "") -> dict:
    """Return a dict with MAE, RMSE, R² and optional unit."""
    return {
        "mae":  compute_mae(pred, target),
        "rmse": compute_rmse(pred, target),
        "r2":   compute_r2(pred, target),
        "unit": unit,
    }
