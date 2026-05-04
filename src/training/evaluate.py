"""Test-set evaluation: writes a per-run CSV and prints a summary."""

import os
import torch
import pandas as pd

from src.utils.metrics import compute_mae, compute_rmse


def evaluate(model, test_loader, dataset, device: str, run_name: str,
             log_dir: str) -> dict:
    model.eval()
    all_pred, all_target = [], []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred = model(batch)
            pred_real = dataset.denormalize(pred)
            target_real = batch.y.squeeze(-1)
            all_pred.append(pred_real.cpu())
            all_target.append(target_real.cpu())

    preds = torch.cat(all_pred)
    targets = torch.cat(all_target)

    mae = compute_mae(preds, targets)
    rmse = compute_rmse(preds, targets)

    results = {
        "run": run_name,
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
    }

    results_path = os.path.join(log_dir, run_name, "test_results.csv")
    pd.DataFrame([results]).to_csv(results_path, index=False)

    unit = dataset.cfg.get("target_unit", "")
    print(f"\n{'='*50}")
    print(f"  TEST RESULTS - {run_name}")
    print(f"  MAE  : {mae:.4f} {unit}")
    print(f"  RMSE : {rmse:.4f} {unit}")
    print(f"{'='*50}\n")

    return results
