# Graph Regression with GNNs on ZINC: Notebook Walkthrough

This mini tutorial distills the `notebooks/graph_reg.ipynb` notebook into a step-by-step guide. It keeps the math compact, points to the key cell outputs, and explains how to reproduce the runs.

## 1) Setup
- Install deps (first cell): `pip install torch-geometric rdkit tensorboard pandas seaborn xgboost`.
- The notebook auto-detects GPU (`cuda` if available). CPU works too, just slower.
- Seeds are fixed for reproducibility.

## 2) Task and Data
- Dataset: ZINC molecular graphs (subset=True → 10k/1k/1k train/val/test).
- Target: molecular logP (octanol–water partition coefficient) as a scalar regression.
- Loaders: PyG `ZINC` + `DataLoader` with batch size 128 and 2 workers.
- Cell output: progress logs show automatic download and processing of the split.

## 3) Quick Visual Checks
- Cell `visualize_grid` plots a 5×5 grid of random molecules with titles `LogP: <value>`. Use it to sanity-check graph structure and target scale.
- Cell `plot_dataset_stats` shows histograms of logP, node counts, edge counts, and average degree. This confirms a narrow logP range and modest graph sizes.

## 4) Model Family (FlexibleGNN)
- Node/edge embeddings: atoms → 21-way embedding; bonds → 4-way embedding, both to `hidden_channels=128`.
- Message passing block repeated `num_layers` times:
- GINE option uses an MLP and residual epsilon:
    $$
    h_v^{(k+1)} = \phi\big((1+\epsilon)h_v^{(k)} + \sum_{u\in\mathcal{N}(v)} \psi(h_u^{(k)}, e_{uv})\big)
    $$
  - GAT/GATv2 option uses edge-aware attention:
    $$
    \alpha_{vu} = \mathrm{softmax}_u\big(a^\top[Wh_v \Vert Wh_u \Vert e_{uv}]\big),\quad
    h_v^{(k+1)} = \sigma\Big(\sum_{u\in\mathcal{N}(v)} \alpha_{vu} W h_u\Big)
    $$
  - Each layer: BatchNorm → ReLU → Dropout(0.5).
- Readout: `global_add_pool` over nodes, then a 2-layer MLP to a single logP prediction.

## 5) Training and Metrics
- Loss: MSE on graph-level targets; metric: MAE on validation/test.
  $$
  \mathrm{MAE} = \frac{1}{N}\sum_i |\hat{y}_i - y_i|
  $$
- Optimizer: AdamW (lr=1e-3, weight decay=1e-5), 20 epochs.
- Logging: TensorBoard scalars (`runs/<exp>/`) for train loss, val loss, val MAE, and LR. The `%tensorboard --logdir runs` cell opens the dashboard.

## 6) Baselines
- Mean baseline: predict the train-set mean logP → Val MAE **1.4785**.
- XGBoost baseline: handcrafted features `[n_atoms, n_bonds, atom_hist(21), bond_hist(4)]` → Val MAE **0.6442**.

## 7) Experiments (cell output table)
| Experiment | Final Train Loss | Best Val MAE |
| --- | --- | --- |
| Mean Baseline | – | 1.4785 |
| XGBoost Baseline | – | 0.6442 |
| GINE_Default (4 layers, 128 hid) | 1.0773 | **0.5180** |
| GINE_Deep (6 layers) | 1.1098 | 0.5729 |
| GAT_Default (4 heads) | 1.3803 | 0.5883 |
| GATv2_Default (4 heads) | 1.3902 | 0.5967 |

Key takeaways from outputs:
- Learned GNNs outperform both baselines; GINE with 4 layers is best on val.
- Deeper GINE (6 layers) did not help here—likely optimization/over-smoothing on this small split.
- Attention variants trail GINE on this task with the current hyperparams.

## 8) Best Model Evaluation
- The `evaluate_best_model` cell reloads the best checkpoint from `runs/<exp>/best_model.pth`.
- Best experiment by val MAE: **GINE_Default (0.5180)**.
- Test-set MAE reported in output: **0.5549**.
- This confirms the GINE default config generalizes slightly worse on test than val but still beats the baselines by a wide margin.

## 9) How to Reproduce
1. Run the notebook top-to-bottom (`jupyter notebook notebooks/graph_reg.ipynb`) so the ZINC subset downloads and caches.
2. Watch the two figures render (molecule grid + dataset histograms) to verify data integrity.
3. Keep `%tensorboard --logdir runs` running to monitor curves as experiments progress.
4. Inspect `runs/` after training for checkpoints and scalar logs; each experiment has its own subfolder.
5. Rerun `experiment_configs` to add/modify configs (e.g., change `num_layers`, `heads`, or `hidden_channels`), then rerun the experiment cell.

## 10) Good Next Steps
- Tune width/depth: try larger `hidden_channels` or different dropout for the GINE default.
- Full dataset: set `subset=False` in `Config` to train on the complete ZINC for potentially lower MAE.
- Better features: incorporate atom/bond continuous features (partial charges, aromaticity flags) or positional encodings.
- Schedulers: cosine decay or OneCycle to stabilize deeper variants.
- Regularization: add gradient clipping to help the deeper GINE stack.
