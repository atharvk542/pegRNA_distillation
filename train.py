import argparse
import json
import time
import numpy as np
from scipy.stats import spearmanr, pearsonr
from xgboost import XGBRegressor
from data_utils import load_df, build_tabular, group_split, TARGET_HEK, TARGET_K562

def build_features(
    tabular: np.ndarray, embeddings: np.ndarray, thermo: np.ndarray
) -> np.ndarray:
    return np.concatenate([tabular, embeddings, thermo], axis=1).astype(np.float32)

def evaluate(model: XGBRegressor, X: np.ndarray, y: np.ndarray) -> dict:
    start = time.perf_counter()
    preds = model.predict(X)
    elapsed = time.perf_counter() - start
    sr, _ = spearmanr(y, preds)
    pr, _ = pearsonr(y, preds)
    rmse = float(np.sqrt(np.mean((y - preds) ** 2)))
    return {
        "spearman_r": float(sr),
        "pearson_r": float(pr),
        "rmse": rmse,
        "inference_ms_per_sample": (elapsed / len(y)) * 1000,
        "inference_total_s": elapsed,
        "n_samples": len(y),
    }

def train_and_evaluate(
    X_train, y_train, X_val, y_val, X_test, y_test, label: str, output_prefix: str
):
    model = XGBRegressor(
        n_estimators=2000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        tree_method="hist",
        device="cuda",
        n_jobs=-1,
        random_state=42,
        early_stopping_rounds=50,
        eval_metric="rmse",
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=100,
    )
    val_metrics = evaluate(model, X_val, y_val)
    test_metrics = evaluate(model, X_test, y_test)
    print(f"\n[{label}] Val:  {val_metrics}")
    print(f"[{label}] Test: {test_metrics}")
    model_path = f"{output_prefix}_{label}.json"
    model.save_model(model_path)
    print(f"Model saved to {model_path}")
    metrics = {"val": val_metrics, "test": test_metrics}
    metrics_path = f"{output_prefix}_{label}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    return model, metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data_23k_v1.csv")
    parser.add_argument("--embeddings", default="embeddings.npy")
    parser.add_argument("--thermo", default="thermo_features.npy")
    parser.add_argument("--output_prefix", default="xgb_model")
    parser.add_argument("--target", choices=["hek", "k562", "both"], default="both")
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Loading data...")
    df = load_df(args.data)
    tabular = build_tabular(df)
    embeddings = np.load(args.embeddings)
    thermo = np.load(args.thermo)

    assert len(df) == len(embeddings) == len(thermo), (
        f"Row count mismatch: df={len(df)}, embeddings={len(embeddings)}, thermo={len(thermo)}"
    )

    X = build_features(tabular, embeddings, thermo)
    print(f"Feature matrix shape: {X.shape}")

    train_idx, val_idx, test_idx = group_split(
        df, args.val_size, args.test_size, args.seed
    )
    print(
        f"Split sizes — train: {len(train_idx)}, val: {len(val_idx)}, test: {len(test_idx)}"
    )

    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]

    targets = []
    if args.target in ("hek", "both"):
        targets.append((TARGET_HEK, "HEK"))
    if args.target in ("k562", "both"):
        targets.append((TARGET_K562, "K562"))

    all_metrics = {}
    for col, label in targets:
        y = df[col].values.astype(np.float32)
        y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]
        _, metrics = train_and_evaluate(
            X_train, y_train, X_val, y_val, X_test, y_test, label, args.output_prefix
        )
        all_metrics[label] = metrics

    summary_path = f"{args.output_prefix}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nAll metrics saved to {summary_path}")


if __name__ == "__main__":
    main()
