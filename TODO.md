# pegRNA Distillation — Project TODO

## Status Legend
- [ ] Not started
- [~] In progress
- [x] Done

---

## 1. Baselines

- [~] **Install PRIDICT2 conda environment** — currently solving, finish setup
- [ ] **Run PRIDICT2 on test set** — once env is ready, load trained weights directly in Python and run inference on the same `test_idx` rows used in `train.py`. Record Spearman R (HEK + K562) and inference ms/sample.
- [ ] **Write `baseline_pridict2.py`** — script that loads PRIDICT2 weights, feeds test set features from `data_23k_v1.csv` directly (bypassing CLI), outputs predictions and metrics into `pridict2_baseline_metrics.json`
- [ ] **Add tabular-only XGBoost baseline** — retrain XGBoost using only the 20 tabular features (no embeddings, no thermo). Already have everything needed in `train.py`, just pass zeros or skip the embedding/thermo concat. Save as `xgb_tabular_only_metrics.json`
- [ ] **Add linear probe baseline** — Ridge regression on NT embeddings alone. Add to a new `baselines.py` script alongside the tabular-only run.

---

## 2. MLP Student Model

- [ ] **Write `train_mlp.py`** — small MLP (3 linear layers, dropout, ReLU) that takes the same (1046,) feature vector as XGBoost. Train with:
  - Hard targets: ground truth efficiency scores (MSE loss)
  - Soft targets: PRIDICT2 predictions on training set (requires baseline step above)
  - Combined loss: `alpha * MSE(pred, ground_truth) + (1-alpha) * MSE(pred, pridict2_pred)` with alpha as a tunable hyperparameter
- [ ] **Tune alpha** — try alpha in [0.3, 0.5, 0.7, 1.0] and report val Spearman R for each. Pick best for final test evaluation.
- [ ] **Save MLP model weights** — `mlp_model_HEK.pt` and `mlp_model_K562.pt`
- [ ] **Record MLP inference time** — same `time.perf_counter()` approach as in `train.py`'s `evaluate()`

---

## 3. Ablation Study

Run all combinations below, both HEK and K562, record test Spearman R. Create `ablation.py` that loops through feature subsets and saves results to `ablation_results.json`.

- [ ] Tabular features only (20 features)
- [ ] Thermo features only (2 features)
- [ ] NT embeddings only (1024 features)
- [ ] Tabular + thermo (22 features)
- [ ] Tabular + embeddings (1044 features)
- [ ] Thermo + embeddings (1026 features)
- [ ] All combined — **[x] Done** (current XGBoost result)

Use XGBoost for all ablation runs so the model class is held constant and results are comparable.

---

## 4. Figures

- [ ] **Figure 1 — Pipeline diagram** — SVG schematic showing: sequence → NT v2 → embeddings → concatenate with thermo + tabular → student model → efficiency score. Can generate with the Visualizer or draw manually.
- [ ] **Figure 2 — Accuracy comparison bar chart** — Spearman R on test set for: linear probe, XGBoost (tabular only), XGBoost (all features), MLP student, PRIDICT2. Two subpanels: HEK and K562. Save as `fig2_accuracy_comparison.png`
- [ ] **Figure 3 — Ablation heatmap or grouped bar chart** — all ablation combinations from Section 3. HEK and K562 side by side. Save as `fig3_ablation.png`
- [ ] **Figure 4 — Inference speed benchmark** — log-scale bar chart of ms/sample for: NT v2 forward pass, PRIDICT2, MLP student, XGBoost. Run each 100x on test set and report mean ± std. Save as `fig4_speed.png`
- [ ] **Figure 5 — Scatter plots predicted vs actual** — 2x2 grid: (PRIDICT2 vs XGBoost) × (HEK vs K562). Annotate with Spearman R. Save as `fig5_scatter.png`
- [ ] **Figure 6 — Performance by edit type** — grouped bar chart of Spearman R broken down by Replacement / Insertion / Deletion, for your best model vs PRIDICT2. Save as `fig6_by_edit_type.png`
- [ ] **Write `generate_figures.py`** — single script that loads all results JSONs and saved predictions, generates all figures above using matplotlib/seaborn, saves to `figures/` directory

---

## 5. Analysis

- [ ] **K562 gap analysis** — plot efficiency score distributions for HEK vs K562 (histogram). Show that K562 scores are compressed near zero. Add 1-2 paragraph explanation to paper.
- [ ] **SHAP values on XGBoost** — run `shap.TreeExplainer` on `xgb_model_HEK.json`, plot feature importance. Check whether NT embedding dimensions or tabular features dominate. Save as `fig_shap.png`. Install with `pip install shap`.
- [ ] **Calibration plot** — bin predicted efficiencies into deciles, plot mean predicted vs mean actual. One panel per cell line. Checks whether model is over/underconfident in specific ranges.
- [ ] **Error analysis by edit type** — for each of Replacement/Insertion/Deletion, compute residuals (predicted - actual) and plot distributions. Identify if model systematically over- or under-predicts a specific edit type.

---

## 6. Code Cleanup

- [ ] **Add `--feature_set` argument to `train.py`** — options: `all`, `tabular_only`, `embeddings_only`, `thermo_only`, `tabular_thermo`, `tabular_embeddings`, `thermo_embeddings`. This removes need for a separate `ablation.py`.
- [ ] **Fix XGBoost device warning** — prediction falls back to CPU during evaluation. Pass `device="cuda"` to `model.predict()` or convert X to a `DMatrix` before evaluating.
- [ ] **Add `--seed` reproducibility note** — document that all splits use `seed=42` and results are tied to this. Add a `--run_multiple_seeds` flag that averages over seeds [42, 123, 456] for more robust reported metrics.
- [ ] **`requirements.txt` pin** — update to reflect actual working versions after all PRIDICT2 and MLP work is done.

---

## 7. Paper

- [ ] **Methods section** — describe pipeline, NT v2 model choice, mean pooling, ViennaRNA features, group-aware train/val/test split, XGBoost hyperparameters, MLP architecture and distillation loss
- [ ] **Results section** — accuracy table (all models × both cell lines), ablation table, speed table
- [ ] **Discussion** — address K562 gap, limitations (99bp sequences vs PRIDICT2's 300bp, no fine-tuning of NT v2, no wet lab validation), future work
- [ ] **Target venue** — decide between *Bioinformatics* (Oxford), *Briefings in Bioinformatics*, or MLCB workshop (NeurIPS)

---

## Current Results (reference)

| Model | HEK Spearman R | K562 Spearman R | Inference (ms/sample) |
|---|---|---|---|
| XGBoost (all features) | 0.792 | 0.575 | 0.028 |
| PRIDICT2 | ~0.87 (reported) | ~0.75 (reported) | TBD |
| MLP student | TBD | TBD | TBD |
| Linear probe | TBD | TBD | TBD |
| XGBoost (tabular only) | TBD | TBD | TBD |