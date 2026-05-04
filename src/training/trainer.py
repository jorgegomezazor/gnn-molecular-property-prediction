"""
src/training/trainer.py

Training loop with:
  - Adam optimizer + cosine / step LR scheduler
  - Early stopping on validation MAE
  - TensorBoard logging
  - Checkpoint saving (best val model)
"""

import os
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import Optional

from src.utils.metrics import compute_mae, compute_rmse


class Trainer:
    """
    Generic trainer for GNN molecular property regression.

    Args:
        model       : PyTorch model.
        dataset     : QM9Dataset instance (for de-normalization).
        cfg         : Full config dict (training sub-dict + logging sub-dict).
        device      : 'cuda' or 'cpu'.
        run_name    : Name used for checkpoint / log directories.
    """

    def __init__(self, model, dataset, cfg: dict,
                 device: str = "cpu", run_name: str = "run"):
        self.model   = model.to(device)
        self.dataset = dataset
        self.device  = device
        self.run_name = run_name

        train_cfg = cfg["training"]
        log_cfg   = cfg["logging"]

        self.epochs    = train_cfg["epochs"]
        self.lr        = train_cfg["learning_rate"]
        self.wd        = train_cfg.get("weight_decay", 1e-5)
        self.patience  = train_cfg.get("early_stopping_patience", 30)
        self.grad_clip = train_cfg.get("gradient_clip", 5.0)
        self.log_every = log_cfg.get("log_every_n_epochs", 10)

        self.log_dir = os.path.join(log_cfg["log_dir"], run_name)
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)

        self.optimizer = Adam(model.parameters(), lr=self.lr, weight_decay=self.wd)

        sched = train_cfg.get("scheduler", "cosine")
        if sched == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer, T_max=train_cfg.get("scheduler_t_max", self.epochs)
            )
        elif sched == "step":
            self.scheduler = StepLR(self.optimizer, step_size=50, gamma=0.5)
        else:
            self.scheduler = None

        self.best_val_mae = float("inf")
        self.best_epoch   = 0
        self.no_improve   = 0
        self.ckpt_path    = os.path.join(self.log_dir, "best_model.pt")

    # ------------------------------------------------------------------ #

    def _step(self, batch, train: bool = True):
        """Forward + loss for one batch. Returns (loss, pred, target) tensors."""
        batch = batch.to(self.device)
        target = self.dataset.normalize(batch.y.squeeze(-1))

        pred = self.model(batch)
        loss = nn.functional.l1_loss(pred, target)

        if train:
            self.optimizer.zero_grad()
            loss.backward()
            if self.grad_clip:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

        # Denormalize for metric computation
        pred_real   = self.dataset.denormalize(pred.detach())
        target_real = batch.y.squeeze(-1)
        return loss.item(), pred_real, target_real

    def _epoch(self, loader, train: bool):
        if train:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        all_pred, all_target = [], []

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for batch in loader:
                loss, pred, target = self._step(batch, train=train)
                total_loss += loss
                all_pred.append(pred.cpu())
                all_target.append(target.cpu())

        preds   = torch.cat(all_pred)
        targets = torch.cat(all_target)
        mae  = compute_mae(preds, targets)
        rmse = compute_rmse(preds, targets)
        return total_loss / len(loader), mae, rmse

    # ------------------------------------------------------------------ #

    def fit(self, train_loader, val_loader):
        """Full training loop. Returns history dict."""
        history = {"train_loss": [], "val_mae": [], "val_rmse": []}
        t0 = time.time()

        for epoch in range(1, self.epochs + 1):
            train_loss, train_mae, _ = self._epoch(train_loader, train=True)
            val_loss,   val_mae, val_rmse = self._epoch(val_loader,   train=False)

            if self.scheduler:
                self.scheduler.step()

            # TensorBoard
            self.writer.add_scalar("Loss/train", train_loss, epoch)
            self.writer.add_scalar("MAE/val",    val_mae,    epoch)
            self.writer.add_scalar("RMSE/val",   val_rmse,   epoch)
            self.writer.add_scalar("LR", self.optimizer.param_groups[0]["lr"], epoch)

            history["train_loss"].append(train_loss)
            history["val_mae"].append(val_mae)
            history["val_rmse"].append(val_rmse)

            # Early stopping / checkpoint
            if val_mae < self.best_val_mae:
                self.best_val_mae = val_mae
                self.best_epoch   = epoch
                self.no_improve   = 0
                torch.save(self.model.state_dict(), self.ckpt_path)
            else:
                self.no_improve += 1

            if epoch % self.log_every == 0:
                elapsed = time.time() - t0
                print(
                    f"[{self.run_name}] Epoch {epoch:3d}/{self.epochs}  "
                    f"train_loss={train_loss:.4f}  "
                    f"val_mae={val_mae:.4f}  "
                    f"val_rmse={val_rmse:.4f}  "
                    f"({elapsed:.0f}s elapsed)"
                )

            if self.no_improve >= self.patience:
                print(f"[{self.run_name}] Early stopping at epoch {epoch}. "
                      f"Best val MAE={self.best_val_mae:.4f} at epoch {self.best_epoch}.")
                break

        self.writer.close()
        return history

    def load_best(self):
        """Load the best checkpoint back into model."""
        self.model.load_state_dict(torch.load(self.ckpt_path, map_location=self.device))
        print(f"[{self.run_name}] Loaded best checkpoint "
              f"(epoch {self.best_epoch}, val MAE={self.best_val_mae:.4f})")
