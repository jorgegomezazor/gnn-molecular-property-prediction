"""Per-run logger: writes history.csv (epoch metrics) and summary.json (final)."""

import os
import json
import csv
from datetime import datetime


class ExperimentLogger:

    def __init__(self, log_dir: str, run_name: str, cfg: dict):
        self.run_dir = os.path.join(log_dir, run_name)
        os.makedirs(self.run_dir, exist_ok=True)

        self.history_path = os.path.join(self.run_dir, "history.csv")
        self.summary_path = os.path.join(self.run_dir, "summary.json")

        self.cfg = cfg
        self.run_name = run_name
        self._history_initialized = False

    def log_epoch(self, epoch: int, metrics: dict):
        row = {"epoch": epoch, **metrics}
        if not self._history_initialized:
            with open(self.history_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                writer.writeheader()
            self._history_initialized = True

        with open(self.history_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writerow(row)

    def save_summary(self, test_metrics: dict):
        summary = {
            "run_name": self.run_name,
            "timestamp": datetime.now().isoformat(),
            "test_metrics": test_metrics,
            "config": self.cfg,
        }
        with open(self.summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[Logger] Summary saved to {self.summary_path}")
