# GNN-3: GNNs for Molecular Property Prediction

**Geometric Deep Learning — Final Project (Academic Year 2025–2026)**

## Research Question

> *Does architectural expressivity (GIN, GAT, Graph Transformer) improve prediction of the dipole moment (μ) on QM9 compared to a standard MPNN baseline, and what structural features does each model learn?*

We compare four architectures on the **QM9** benchmark targeting the **μ (dipole moment)** property (target index 0), analyse learned representations via gradient-based attribution, and run ablations on depth and hidden dimension.

---

## Project Structure

```
gnn3-molecular/
├── configs/              # YAML configs per model
│   ├── base.yaml
│   ├── mpnn.yaml
│   ├── gin.yaml
│   ├── gat.yaml
│   └── graph_transformer.yaml
├── data/                 # Auto-downloaded QM9 dataset
├── src/
│   ├── data/
│   │   ├── dataset.py    # QM9 loading + splits + normalization
│   │   └── featurizer.py # Atom / bond featurization
│   ├── models/
│   │   ├── mpnn.py       # MPNN (Gilmer et al., ICML 2017)
│   │   ├── gin.py        # GIN  (Xu et al., ICLR 2019)
│   │   ├── gat.py        # GAT  (Veličković et al., ICLR 2018)
│   │   └── graph_transformer.py  # GPS (Rampášek et al., NeurIPS 2022)
│   ├── training/
│   │   ├── trainer.py    # Training loop with early stopping
│   │   └── evaluate.py   # MAE / RMSE + per-property metrics
│   └── utils/
│       ├── metrics.py
│       ├── logger.py
│       └── visualization.py  # Attribution + embedding plots
├── scripts/
│   ├── train.py          # Main training entry point
│   ├── run_all.sh        # Train all models sequentially
│   ├── ablation.py       # Depth / width ablation sweep
│   └── analyze.py        # Post-hoc analysis + figures
├── notebooks/
│   └── analysis.ipynb    # Exploratory analysis notebook
├── results/              # Saved checkpoints + CSV logs
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** PyTorch Geometric requires a matching PyTorch version.  
> Visit https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html  
> for platform-specific install commands if `pip install` fails.

### 2. Download QM9 (automatic)

The dataset is downloaded automatically on first run via `torch_geometric.datasets.QM9`.  
It will be placed in `data/QM9/`.

---

## Usage

### Train a single model

```bash
python scripts/train.py --config configs/mpnn.yaml
python scripts/train.py --config configs/gin.yaml
python scripts/train.py --config configs/gat.yaml
python scripts/train.py --config configs/graph_transformer.yaml
```

### Train all models (sequential)

```bash
bash scripts/run_all.sh
```

### Run ablation study

```bash
python scripts/ablation.py --model gin --sweep depth
python scripts/ablation.py --model gin --sweep hidden_dim
```

### Generate figures and analysis

```bash
python scripts/analyze.py --results_dir results/
```

---

## Reproducing Main Results

Set the random seed (already fixed in configs to **42**):

```bash
python scripts/train.py --config configs/mpnn.yaml --seed 42
```

Expected runtime per model on a single GPU (RTX 3060): ~15–25 minutes for 300 epochs.  
On CPU: ~2–3 hours per model.

---

## Target Property

| Index | Symbol | Property             | Unit  |
|-------|--------|----------------------|-------|
| 0     | μ      | Dipole moment        | D     |
| 1     | α      | Polarizability       | a₀³   |
| 2     | ε_HOMO | HOMO energy          | eV    |
| ...   | ...    | ...                  | ...   |

Change the target in `configs/base.yaml` → `target_idx`.

---

## References

- Gilmer et al., *Neural Message Passing for Quantum Chemistry*, ICML 2017  
- Xu et al., *How Powerful are Graph Neural Networks?*, ICLR 2019  
- Veličković et al., *Graph Attention Networks*, ICLR 2018  
- Rampášek et al., *Recipe for a General, Powerful, Scalable Graph Transformer*, NeurIPS 2022  
- Ramakrishnan et al., *Quantum chemistry structures and properties of 134 kilo molecules*, 2014
