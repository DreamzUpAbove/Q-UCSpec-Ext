"""
Generate core analysis tables for HQFr classification study.

Outputs 5 key tables in CSV + Markdown:
1) Dataset split and class balance
2) Feature statistics by class (train split, raw scale)
3) Model architecture and parameterization
4) Training phase summary
5) Test performance with bootstrap confidence intervals
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from config import BATCH_SIZE, FEATURE_COLUMNS, POST_NN_HIDDEN, RANDOM_STATE, set_seed
from hqfr_model import HQFr
from preprocessing import load_and_preprocess

CLASS_NAMES = {0: "CaF2", 1: "CaF2:Er"}
POS_LABEL = 1.0


def _build_loader(X: np.ndarray, y: np.ndarray, batch_size: int) -> DataLoader:
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float64),
        torch.tensor(y, dtype=torch.float64),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def _collect_probs(model: HQFr, loader: DataLoader, device: str) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probs: list[float] = []
    all_labels: list[float] = []

    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb.to(device))
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
            probs = np.atleast_1d(probs).astype(float)
            all_probs.extend(probs.tolist())
            all_labels.extend(yb.numpy().tolist())

    return np.array(all_probs, dtype=float), np.array(all_labels, dtype=float)


def _binary_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float) -> dict[str, float]:
    preds = (probs > threshold).astype(float)

    tp = float(((preds == 1) & (labels == 1)).sum())
    tn = float(((preds == 0) & (labels == 0)).sum())
    fp = float(((preds == 1) & (labels == 0)).sum())
    fn = float(((preds == 0) & (labels == 1)).sum())

    acc = (tp + tn) / len(labels) if len(labels) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom > 0 else 0.0

    if len(np.unique(labels)) > 1:
        roc_auc = float(roc_auc_score(labels, probs))
        pr_p, pr_r, _ = precision_recall_curve(labels, probs)
        pr_auc = float(auc(pr_r, pr_p))
    else:
        roc_auc = float("nan")
        pr_auc = float("nan")

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": mcc,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }


def _bootstrap_metric_ci(
    labels: np.ndarray,
    probs: np.ndarray,
    threshold: float,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = RANDOM_STATE,
) -> dict[str, tuple[float, float]]:
    rng = np.random.default_rng(seed)
    n = len(labels)
    tracked = ["accuracy", "precision", "recall", "f1", "mcc", "roc_auc", "pr_auc"]
    values: dict[str, list[float]] = {k: [] for k in tracked}

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yb = labels[idx]
        pb = probs[idx]
        m = _binary_metrics(yb, pb, threshold)
        for k in tracked:
            v = m[k]
            if not np.isnan(v):
                values[k].append(float(v))

    lo = 100.0 * (alpha / 2.0)
    hi = 100.0 * (1.0 - alpha / 2.0)
    ci: dict[str, tuple[float, float]] = {}
    for k in tracked:
        arr = np.array(values[k], dtype=float)
        if len(arr) == 0:
            ci[k] = (float("nan"), float("nan"))
        else:
            ci[k] = (float(np.percentile(arr, lo)), float(np.percentile(arr, hi)))
    return ci


def _phase_summary(phase_name: str, phase_hist: dict[str, list[float]]) -> dict[str, Any]:
    val_loss = np.array(phase_hist["val_loss"], dtype=float)
    val_acc = np.array(phase_hist["val_acc"], dtype=float)
    train_loss = np.array(phase_hist["train_loss"], dtype=float)
    train_acc = np.array(phase_hist["train_acc"], dtype=float)
    cscale = np.array(phase_hist["circuit_scale"], dtype=float)

    best_loss_idx = int(np.argmin(val_loss))
    best_acc_idx = int(np.argmax(val_acc))

    return {
        "phase": phase_name,
        "epochs_run": len(train_loss),
        "best_val_loss": float(val_loss[best_loss_idx]),
        "best_val_loss_epoch": best_loss_idx + 1,
        "best_val_acc": float(val_acc[best_acc_idx]),
        "best_val_acc_epoch": best_acc_idx + 1,
        "final_train_loss": float(train_loss[-1]),
        "final_val_loss": float(val_loss[-1]),
        "final_train_acc": float(train_acc[-1]),
        "final_val_acc": float(val_acc[-1]),
        "final_circuit_scale": float(cscale[-1]),
    }


def _format_value(v: Any) -> str:
    if isinstance(v, float):
        if np.isnan(v):
            return "nan"
        if abs(v) >= 1000:
            return f"{v:.2f}"
        return f"{v:.6f}"
    return str(v)


def _to_markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |\n"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |\n"
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(_format_value(row[c]) for c in cols) + " |")
    return header + sep + "\n".join(rows) + "\n"


def _save_table(df: pd.DataFrame, out_dir: Path, stem: str) -> None:
    csv_path = out_dir / f"{stem}.csv"
    md_path = out_dir / f"{stem}.md"
    df.to_csv(csv_path, index=False)
    md_path.write_text(_to_markdown_table(df))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 5 HQFr analysis tables.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints_hqfr/final_model.pth",
        help="Path to trained model checkpoint.",
    )
    parser.add_argument(
        "--history",
        type=str,
        default="checkpoints_hqfr/history.json",
        help="Path to training history JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="hqfr-dennis/analysis_tables",
        help="Directory for output table files.",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Inference device."
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    args = parser.parse_args()

    set_seed(RANDOM_STATE)
    torch.set_default_dtype(torch.float64)

    checkpoint_path = Path(args.checkpoint)
    history_path = Path(args.history)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    history = json.loads(history_path.read_text())
    best_threshold = float(history.get("best_threshold", 0.5))

    X_train, X_val, X_test, y_train, y_val, y_test, transformer, pos_weight = load_and_preprocess(
        verbose=False
    )

    # ---------------- Table 1: Dataset split summary ----------------
    dataset_rows = []
    for split_name, y in (
        ("train", y_train),
        ("val", y_val),
        ("test", y_test),
    ):
        n0 = int(np.sum(y == 0))
        n1 = int(np.sum(y == 1))
        n = len(y)
        dataset_rows.append(
            {
                "split": split_name,
                "n_samples": n,
                f"n_{CLASS_NAMES[0]}": n0,
                f"n_{CLASS_NAMES[1]}": n1,
                f"pct_{CLASS_NAMES[0]}": 100.0 * n0 / n,
                f"pct_{CLASS_NAMES[1]}": 100.0 * n1 / n,
            }
        )
    table1 = pd.DataFrame(dataset_rows)
    _save_table(table1, out_dir, "table1_dataset_split_summary")

    # ---------------- Table 2: Feature stats by class ----------------
    X_train_raw = transformer.inverse_transform(X_train)
    rows2: list[dict[str, Any]] = []
    for feat_idx, feat_name in enumerate(FEATURE_COLUMNS):
        for cls in (0, 1):
            mask = y_train == cls
            vals = X_train_raw[mask, feat_idx]
            rows2.append(
                {
                    "feature": feat_name,
                    "class": CLASS_NAMES[cls],
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "median": float(np.median(vals)),
                    "min": float(np.min(vals)),
                    "max": float(np.max(vals)),
                }
            )
    table2 = pd.DataFrame(rows2)
    _save_table(table2, out_dir, "table2_feature_stats_by_class")

    # ---------------- Table 3: Model architecture ----------------
    model = HQFr(post_nn_hidden=POST_NN_HIDDEN)
    model_info = model.get_model_info()
    arch_rows = [
        ("model_type", model_info["type"]),
        ("n_qubits", model.n_qubits),
        ("n_layers", model.n_layers),
        ("n_repeats", model.quantum_layer.n_repeats),
        ("embedding", "IQP" if model.quantum_layer.use_iqp else "Angle"),
        (
            "observables",
            "X,Y,Z per qubit" if model.quantum_layer.use_multi_pauli else "Z only",
        ),
        ("quantum_output_dim", model.quantum_layer.n_outputs),
        ("pre_params", model_info["pre_params"]),
        ("quantum_params", model_info["quantum_params"]),
        ("post_params", model_info["post_params"]),
        (
            "total_params",
            int(model_info["pre_params"] + model_info["quantum_params"] + model_info["post_params"] + 1),
        ),
        ("initial_circuit_scale", float(model_info["circuit_scale"])),
    ]
    table3 = pd.DataFrame(arch_rows, columns=["parameter", "value"])
    _save_table(table3, out_dir, "table3_model_architecture")

    # ---------------- Table 4: Training phase summary ----------------
    phase1_summary = _phase_summary("phase1", history["phase1"])
    phase2_summary = _phase_summary("phase2", history["phase2"])
    table4 = pd.DataFrame([phase1_summary, phase2_summary])
    table4["best_threshold"] = best_threshold
    th_info = history.get("threshold_selection", {})
    table4["threshold_val_f1"] = float(th_info.get("f1", np.nan))
    table4["threshold_grid_start"] = float(th_info.get("grid_start", np.nan))
    table4["threshold_grid_stop"] = float(th_info.get("grid_stop", np.nan))
    table4["threshold_grid_step"] = float(th_info.get("grid_step", np.nan))
    _save_table(table4, out_dir, "table4_training_phase_summary")

    # ---------------- Table 5: Test performance + CI ----------------
    ckpt = torch.load(checkpoint_path, map_location=args.device)
    model_state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(model_state)
    model = model.to(args.device)

    test_loader = _build_loader(X_test, y_test, args.batch_size)
    test_probs, test_labels = _collect_probs(model, test_loader, args.device)

    point = _binary_metrics(test_labels, test_probs, best_threshold)
    ci = _bootstrap_metric_ci(
        test_labels,
        test_probs,
        threshold=best_threshold,
        n_bootstrap=args.n_bootstrap,
    )
    rows5: list[dict[str, Any]] = [
        {"metric": "threshold", "value": best_threshold, "ci_low": np.nan, "ci_high": np.nan},
        {"metric": "tp", "value": int(point["tp"]), "ci_low": np.nan, "ci_high": np.nan},
        {"metric": "tn", "value": int(point["tn"]), "ci_low": np.nan, "ci_high": np.nan},
        {"metric": "fp", "value": int(point["fp"]), "ci_low": np.nan, "ci_high": np.nan},
        {"metric": "fn", "value": int(point["fn"]), "ci_low": np.nan, "ci_high": np.nan},
    ]
    for metric_name in ("accuracy", "precision", "recall", "f1", "mcc", "roc_auc", "pr_auc"):
        rows5.append(
            {
                "metric": metric_name,
                "value": float(point[metric_name]),
                "ci_low": float(ci[metric_name][0]),
                "ci_high": float(ci[metric_name][1]),
            }
        )
    table5 = pd.DataFrame(rows5)
    _save_table(table5, out_dir, "table5_test_performance_with_ci")

    manifest = {
        "checkpoint": str(checkpoint_path),
        "history": str(history_path),
        "class_mapping": {"0": CLASS_NAMES[0], "1": CLASS_NAMES[1]},
        "pos_weight_train_split": float(pos_weight),
        "tables": [
            "table1_dataset_split_summary.csv",
            "table2_feature_stats_by_class.csv",
            "table3_model_architecture.csv",
            "table4_training_phase_summary.csv",
            "table5_test_performance_with_ci.csv",
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print("Saved analysis tables:")
    for t in manifest["tables"]:
        print(f"  - {out_dir / t}")
    print(f"Manifest: {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
