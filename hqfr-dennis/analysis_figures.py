"""
Generate analysis figures for HQFr training/evaluation.

Figures:
1) Feature separability (train split, per-feature class histograms + mean/std markers)
2) Two-phase training dynamics (loss/accuracy/circuit scale with phase boundary)
3) Threshold analysis on validation set (precision, recall, F1 vs threshold)
4) Discrimination curves on test set (ROC and PR)
5) Error localization (confusion matrix + error rate by energy bins)

Additional classification diagnostic:
6) Probability calibration (reliability curve + score histogram + Brier score)
7) 2D embedding view (UMAP if available, else PCA fallback)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.calibration import calibration_curve
from sklearn.decomposition import PCA
from sklearn.metrics import (
    auc,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, TensorDataset

from config import BATCH_SIZE, FEATURE_COLUMNS, POST_NN_HIDDEN, RANDOM_STATE, set_seed
from hqfr_model import HQFr
from preprocessing import load_and_preprocess

CLASS_NAMES = {0: "CaF2", 1: "CaF2:Er"}
POSITIVE_CLASS_NAME = CLASS_NAMES[1]


def _build_loader(X: np.ndarray, y: np.ndarray, batch_size: int) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float64),
        torch.tensor(y, dtype=torch.float64),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def _collect_probs(
    model: HQFr, loader: DataLoader, device: str, split_name: str
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probs: list[float] = []
    all_labels: list[float] = []

    total_batches = len(loader)
    with torch.no_grad():
        for i, (xb, yb) in enumerate(loader, start=1):
            if i == 1 or (i % 5 == 0) or i == total_batches:
                print(f"[{split_name}] collecting batch {i}/{total_batches}", flush=True)
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
            probs = np.atleast_1d(probs).astype(float)
            all_probs.extend(probs.tolist())
            all_labels.extend(yb.numpy().tolist())

    return np.array(all_probs, dtype=float), np.array(all_labels, dtype=float)


def _metrics_vs_threshold(
    probs: np.ndarray, labels: np.ndarray, thresholds: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    precision_vals = np.zeros_like(thresholds, dtype=float)
    recall_vals = np.zeros_like(thresholds, dtype=float)
    f1_vals = np.zeros_like(thresholds, dtype=float)

    for i, th in enumerate(thresholds):
        preds = (probs > th).astype(float)
        tp = float(((preds == 1) & (labels == 1)).sum())
        fp = float(((preds == 1) & (labels == 0)).sum())
        fn = float(((preds == 0) & (labels == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            (2.0 * precision * recall / (precision + recall))
            if (precision + recall) > 0
            else 0.0
        )
        precision_vals[i] = precision
        recall_vals[i] = recall
        f1_vals[i] = f1

    return precision_vals, recall_vals, f1_vals


def _bootstrap_threshold_bands(
    probs: np.ndarray,
    labels: np.ndarray,
    thresholds: np.ndarray,
    n_bootstrap: int = 300,
    alpha: float = 0.05,
    seed: int = RANDOM_STATE,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = len(labels)
    precision_boot = np.zeros((n_bootstrap, len(thresholds)), dtype=float)
    recall_boot = np.zeros((n_bootstrap, len(thresholds)), dtype=float)
    f1_boot = np.zeros((n_bootstrap, len(thresholds)), dtype=float)

    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        p, r, f1 = _metrics_vs_threshold(probs[idx], labels[idx], thresholds)
        precision_boot[b] = p
        recall_boot[b] = r
        f1_boot[b] = f1

    lo = 100.0 * (alpha / 2.0)
    hi = 100.0 * (1.0 - alpha / 2.0)
    return {
        "precision_lo": np.percentile(precision_boot, lo, axis=0),
        "precision_hi": np.percentile(precision_boot, hi, axis=0),
        "recall_lo": np.percentile(recall_boot, lo, axis=0),
        "recall_hi": np.percentile(recall_boot, hi, axis=0),
        "f1_lo": np.percentile(f1_boot, lo, axis=0),
        "f1_hi": np.percentile(f1_boot, hi, axis=0),
    }


def _bootstrap_roc_pr_bands(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bootstrap: int = 300,
    alpha: float = 0.05,
    seed: int = RANDOM_STATE,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    n = len(labels)
    fpr_grid = np.linspace(0.0, 1.0, 101)
    recall_grid = np.linspace(0.0, 1.0, 101)

    tpr_boot = []
    pr_boot = []
    auc_boot = []
    pr_auc_boot = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yb = labels[idx]
        pb = probs[idx]
        # Skip degenerate resamples with one class only.
        if len(np.unique(yb)) < 2:
            continue

        fpr, tpr, _ = roc_curve(yb, pb)
        interp_tpr = np.interp(fpr_grid, fpr, tpr)
        interp_tpr[0] = 0.0
        interp_tpr[-1] = 1.0
        tpr_boot.append(interp_tpr)
        auc_boot.append(roc_auc_score(yb, pb))

        precision, recall, _ = precision_recall_curve(yb, pb)
        # sklearn returns decreasing recall; reverse to increasing for interpolation
        recall_inc = recall[::-1]
        precision_inc = precision[::-1]
        interp_precision = np.interp(recall_grid, recall_inc, precision_inc)
        pr_boot.append(interp_precision)
        pr_auc_boot.append(auc(recall, precision))

    if len(tpr_boot) == 0 or len(pr_boot) == 0:
        raise RuntimeError("Bootstrap failed: no valid resamples with both classes.")

    tpr_boot_arr = np.array(tpr_boot, dtype=float)
    pr_boot_arr = np.array(pr_boot, dtype=float)

    lo = 100.0 * (alpha / 2.0)
    hi = 100.0 * (1.0 - alpha / 2.0)
    return {
        "fpr_grid": fpr_grid,
        "roc_tpr_lo": np.percentile(tpr_boot_arr, lo, axis=0),
        "roc_tpr_hi": np.percentile(tpr_boot_arr, hi, axis=0),
        "roc_auc_lo": float(np.percentile(auc_boot, lo)),
        "roc_auc_hi": float(np.percentile(auc_boot, hi)),
        "recall_grid": recall_grid,
        "pr_prec_lo": np.percentile(pr_boot_arr, lo, axis=0),
        "pr_prec_hi": np.percentile(pr_boot_arr, hi, axis=0),
        "pr_auc_lo": float(np.percentile(pr_auc_boot, lo)),
        "pr_auc_hi": float(np.percentile(pr_auc_boot, hi)),
    }


def _figure1_feature_separability(
    X_train: np.ndarray, y_train: np.ndarray, out_path: Path
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    colors = {0: "#1f77b4", 1: "#d62728"}
    labels_txt = {0: CLASS_NAMES[0], 1: CLASS_NAMES[1]}

    for i, ax in enumerate(axes):
        x0 = X_train[y_train == 0, i]
        x1 = X_train[y_train == 1, i]

        ax.hist(
            x0,
            bins=35,
            alpha=0.50,
            density=True,
            color=colors[0],
            label=labels_txt[0],
        )
        ax.hist(
            x1,
            bins=35,
            alpha=0.50,
            density=True,
            color=colors[1],
            label=labels_txt[1],
        )

        for arr, c in ((x0, colors[0]), (x1, colors[1])):
            mu = float(np.mean(arr))
            sd = float(np.std(arr))
            ax.axvline(mu, color=c, linewidth=1.8)
            ax.axvline(mu - sd, color=c, linestyle="--", linewidth=1.0, alpha=0.8)
            ax.axvline(mu + sd, color=c, linestyle="--", linewidth=1.0, alpha=0.8)

        ax.set_title(FEATURE_COLUMNS[i], fontsize=10)
        ax.set_xlabel("Standardized value")
        ax.grid(alpha=0.2)
        if i == 0:
            ax.set_ylabel("Density")
            ax.legend(frameon=False, fontsize=9)

    fig.suptitle(f"Feature Separability ({CLASS_NAMES[0]} vs {CLASS_NAMES[1]})", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _figure2_training_dynamics(history: dict, out_path: Path) -> None:
    p1 = history["phase1"]
    p2 = history["phase2"]

    p1_n = len(p1["train_loss"])
    p2_n = len(p2["train_loss"])
    p1_epochs = np.arange(1, p1_n + 1)
    p2_epochs = np.arange(p1_n + 1, p1_n + p2_n + 1)
    boundary = p1_n + 0.5

    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)

    # Loss
    ax = axes[0]
    ax.plot(p1_epochs, p1["train_loss"], label="Train loss (P1)", color="#1f77b4")
    ax.plot(p1_epochs, p1["val_loss"], label="Val loss (P1)", color="#ff7f0e")
    ax.plot(p2_epochs, p2["train_loss"], label="Train loss (P2)", color="#2ca02c")
    ax.plot(p2_epochs, p2["val_loss"], label="Val loss (P2)", color="#d62728")
    ax.axvline(boundary, color="black", linestyle="--", linewidth=1)
    ax.set_title("Loss")
    ax.set_xlabel("")
    ax.set_ylabel("BCE loss")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=8)

    # Accuracy
    ax = axes[1]
    ax.plot(p1_epochs, p1["train_acc"], label="Train acc (P1)", color="#1f77b4")
    ax.plot(p1_epochs, p1["val_acc"], label="Val acc (P1)", color="#ff7f0e")
    ax.plot(p2_epochs, p2["train_acc"], label="Train acc (P2)", color="#2ca02c")
    ax.plot(p2_epochs, p2["val_acc"], label="Val acc (P2)", color="#d62728")
    ax.axvline(boundary, color="black", linestyle="--", linewidth=1)
    ax.set_title("Accuracy")
    ax.set_xlabel("")
    ax.set_ylabel("Accuracy")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=8)

    # Circuit scale
    ax = axes[2]
    ax.plot(p1_epochs, p1["circuit_scale"], label="Circuit scale (P1)", color="#9467bd")
    ax.plot(p2_epochs, p2["circuit_scale"], label="Circuit scale (P2)", color="#8c564b")
    ax.axvline(boundary, color="black", linestyle="--", linewidth=1)
    ax.set_title("Circuit Scale")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Scale value")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=8)

    # Phase annotation block
    txt = (
        f"Phase 1 epochs: {p1_n}\n"
        f"Phase 2 epochs: {p2_n}\n"
        f"Best threshold: {history.get('best_threshold', 0.5):.2f}\n\n"
        "Dashed line = phase transition"
    )
    axes[0].text(
        0.02,
        0.08,
        txt,
        transform=axes[0].transAxes,
        fontsize=9,
        va="bottom",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "lightgray"},
    )

    fig.suptitle("Two-Phase Training Dynamics", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _figure3_threshold_analysis(
    val_probs: np.ndarray, val_labels: np.ndarray, best_threshold: float, out_path: Path
) -> None:
    thresholds = np.linspace(0.30, 0.70, 21)
    precision_vals, recall_vals, f1_vals = _metrics_vs_threshold(
        val_probs, val_labels, thresholds
    )
    bands = _bootstrap_threshold_bands(val_probs, val_labels, thresholds, n_bootstrap=300)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, precision_vals, label="Precision", color="#1f77b4")
    ax.plot(thresholds, recall_vals, label="Recall", color="#ff7f0e")
    ax.plot(thresholds, f1_vals, label="F1", color="#2ca02c", linewidth=2.2)
    ax.fill_between(
        thresholds,
        bands["precision_lo"],
        bands["precision_hi"],
        color="#1f77b4",
        alpha=0.10,
        linewidth=0,
    )
    ax.fill_between(
        thresholds,
        bands["recall_lo"],
        bands["recall_hi"],
        color="#ff7f0e",
        alpha=0.10,
        linewidth=0,
    )
    ax.fill_between(
        thresholds,
        bands["f1_lo"],
        bands["f1_hi"],
        color="#2ca02c",
        alpha=0.12,
        linewidth=0,
    )

    # Mark chosen threshold
    ix = int(np.argmin(np.abs(thresholds - best_threshold)))
    ax.scatter(
        [thresholds[ix]],
        [f1_vals[ix]],
        color="black",
        s=55,
        zorder=3,
        label=f"Chosen threshold = {thresholds[ix]:.2f}",
    )
    ax.axvline(thresholds[ix], color="black", linestyle="--", alpha=0.8)

    ax.set_title(f"Validation Threshold Analysis (Positive = {POSITIVE_CLASS_NAME})")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Metric value")
    ax.set_xlim(0.30, 0.70)
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _figure4_discrimination_curves(
    test_probs: np.ndarray, test_labels: np.ndarray, out_path: Path
) -> None:
    fpr, tpr, _ = roc_curve(test_labels, test_probs)
    roc_auc = roc_auc_score(test_labels, test_probs)
    pr_precision, pr_recall, _ = precision_recall_curve(test_labels, test_probs)
    pr_auc = auc(pr_recall, pr_precision)
    bands = _bootstrap_roc_pr_bands(test_probs, test_labels, n_bootstrap=300)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax = axes[0]
    ax.fill_between(
        bands["fpr_grid"],
        bands["roc_tpr_lo"],
        bands["roc_tpr_hi"],
        color="#1f77b4",
        alpha=0.15,
        linewidth=0,
        label="95% CI (bootstrap)",
    )
    ax.plot(
        fpr,
        tpr,
        color="#1f77b4",
        linewidth=2,
        label=(
            f"AUROC = {roc_auc:.4f} "
            f"[{bands['roc_auc_lo']:.4f}, {bands['roc_auc_hi']:.4f}]"
        ),
    )
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_title("ROC Curve (Test)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)

    ax = axes[1]
    ax.fill_between(
        bands["recall_grid"],
        bands["pr_prec_lo"],
        bands["pr_prec_hi"],
        color="#d62728",
        alpha=0.15,
        linewidth=0,
        label="95% CI (bootstrap)",
    )
    ax.plot(
        pr_recall,
        pr_precision,
        color="#d62728",
        linewidth=2,
        label=(
            f"AUPRC = {pr_auc:.4f} "
            f"[{bands['pr_auc_lo']:.4f}, {bands['pr_auc_hi']:.4f}]"
        ),
    )
    ax.set_title("Precision-Recall Curve (Test)")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)

    fig.suptitle(f"Discrimination Curves ({CLASS_NAMES[0]} vs {CLASS_NAMES[1]})", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _figure5_error_localization(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    energy_raw: np.ndarray,
    out_path: Path,
    bins: int = 12,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    errors = (y_true != y_pred).astype(float)

    bin_edges = np.linspace(float(np.min(energy_raw)), float(np.max(energy_raw)), bins + 1)
    bin_ids = np.digitize(energy_raw, bin_edges[1:-1], right=False)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    err_rates = []
    counts = []
    for b in range(bins):
        mask = bin_ids == b
        cnt = int(mask.sum())
        counts.append(cnt)
        err_rates.append(float(errors[mask].mean()) if cnt > 0 else np.nan)
    err_rates_arr = np.array(err_rates, dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Confusion matrix
    ax = axes[0]
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix (Test)")
    ax.set_xticks([0, 1], labels=[f"Pred {CLASS_NAMES[0]}", f"Pred {CLASS_NAMES[1]}"])
    ax.set_yticks([0, 1], labels=[f"True {CLASS_NAMES[0]}", f"True {CLASS_NAMES[1]}"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Error rate by energy bins
    ax = axes[1]
    ax.plot(centers, err_rates_arr, marker="o", color="#d62728", label="Error rate")
    ax.set_title("Error Rate by Energy Bin (Test)")
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Error rate")
    ax.set_ylim(0.0, min(1.0, np.nanmax(err_rates_arr) + 0.1))
    ax.grid(alpha=0.2)

    ax2 = ax.twinx()
    ax2.bar(
        centers,
        counts,
        width=(bin_edges[1] - bin_edges[0]) * 0.85,
        alpha=0.18,
        color="#1f77b4",
        label="Samples/bin",
    )
    ax2.set_ylabel("Sample count")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, loc="upper right")

    fig.suptitle(f"Error Localization ({CLASS_NAMES[0]} vs {CLASS_NAMES[1]})", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _figure6_probability_calibration(
    test_probs: np.ndarray, test_labels: np.ndarray, out_path: Path
) -> None:
    prob_true, prob_pred = calibration_curve(test_labels, test_probs, n_bins=10, strategy="uniform")
    brier = brier_score_loss(test_labels, test_probs)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax = axes[0]
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Perfect calibration")
    ax.plot(prob_pred, prob_true, marker="o", color="#1f77b4", linewidth=2, label="Model")
    ax.set_title(f"Reliability Curve (Brier = {brier:.4f})")
    ax.set_xlabel(f"Predicted P({POSITIVE_CLASS_NAME})")
    ax.set_ylabel(f"Observed frequency of {POSITIVE_CLASS_NAME}")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)

    ax = axes[1]
    ax.hist(
        test_probs[test_labels == 0],
        bins=20,
        alpha=0.55,
        color="#1f77b4",
        label=CLASS_NAMES[0],
        density=True,
    )
    ax.hist(
        test_probs[test_labels == 1],
        bins=20,
        alpha=0.55,
        color="#d62728",
        label=CLASS_NAMES[1],
        density=True,
    )
    ax.set_title("Predicted Probability Distribution")
    ax.set_xlabel(f"Predicted P({POSITIVE_CLASS_NAME})")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)

    fig.suptitle("Probability Calibration and Score Distribution", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _figure7_embedding_projection(
    X_test: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: Path,
    seed: int = RANDOM_STATE,
) -> str:
    method_name = "UMAP"
    try:
        import umap  # type: ignore[import-untyped]

        reducer = umap.UMAP(
            n_neighbors=20,
            min_dist=0.10,
            n_components=2,
            metric="euclidean",
            random_state=seed,
        )
        embedding = reducer.fit_transform(X_test)
    except Exception:
        # Fallback keeps the script working if umap-learn is unavailable.
        method_name = "PCA (fallback)"
        embedding = PCA(n_components=2).fit_transform(X_test)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Panel A: true classes
    ax = axes[0]
    for cls, color in ((0, "#1f77b4"), (1, "#d62728")):
        mask = y_true == cls
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=24,
            alpha=0.75,
            color=color,
            edgecolors="none",
            label=CLASS_NAMES[cls],
        )
    ax.set_title("True Class Clusters")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)

    # Panel B: classification correctness
    ax = axes[1]
    correct = y_true == y_pred
    ax.scatter(
        embedding[correct, 0],
        embedding[correct, 1],
        s=24,
        alpha=0.70,
        color="#2ca02c",
        edgecolors="none",
        label="Correct",
    )
    ax.scatter(
        embedding[~correct, 0],
        embedding[~correct, 1],
        s=28,
        alpha=0.85,
        color="#ff7f0e",
        edgecolors="black",
        linewidths=0.25,
        label="Misclassified",
    )
    ax.set_title("Prediction Correctness")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)

    fig.suptitle(f"2D Embedding of Test Samples ({method_name})", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return method_name


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate HQFr analysis figures.")
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
        default="hqfr-dennis/analysis_figures",
        help="Directory for output figure PNG files.",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Inference device."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size for probability collection.",
    )
    parser.add_argument(
        "--energy-bins",
        type=int,
        default=12,
        help="Number of bins for error-rate-vs-energy plot.",
    )
    args = parser.parse_args()

    set_seed(RANDOM_STATE)
    torch.set_default_dtype(torch.float64)

    checkpoint_path = Path(args.checkpoint)
    history_path = Path(args.history)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not history_path.exists():
        raise FileNotFoundError(f"History JSON not found: {history_path}")

    history = json.loads(history_path.read_text())
    best_threshold = float(history.get("best_threshold", 0.5))

    X_train, X_val, X_test, y_train, y_val, y_test, transformer, _ = load_and_preprocess(
        verbose=False
    )
    val_loader = _build_loader(X_val, y_val, args.batch_size)
    test_loader = _build_loader(X_test, y_test, args.batch_size)

    model = HQFr(post_nn_hidden=POST_NN_HIDDEN)
    ckpt = torch.load(checkpoint_path, map_location=args.device)
    model_state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(model_state)
    model = model.to(args.device)

    val_probs, val_labels = _collect_probs(model, val_loader, args.device, "val")
    test_probs, test_labels = _collect_probs(model, test_loader, args.device, "test")
    test_preds = (test_probs > best_threshold).astype(float)

    # Recover physical energy values from standardized features.
    energy_col = FEATURE_COLUMNS.index("Energy (eV)")
    X_test_raw = transformer.inverse_transform(X_test)
    energy_test = X_test_raw[:, energy_col]

    # Generate and save figures
    print("Generating Figure 1...", flush=True)
    _figure1_feature_separability(
        X_train, y_train, output_dir / "figure1_feature_separability.png"
    )
    print("Generating Figure 2...", flush=True)
    _figure2_training_dynamics(history, output_dir / "figure2_training_dynamics.png")
    print("Generating Figure 3...", flush=True)
    _figure3_threshold_analysis(
        val_probs, val_labels, best_threshold, output_dir / "figure3_threshold_analysis.png"
    )
    print("Generating Figure 4...", flush=True)
    _figure4_discrimination_curves(
        test_probs, test_labels, output_dir / "figure4_discrimination_curves.png"
    )
    print("Generating Figure 5...", flush=True)
    _figure5_error_localization(
        y_true=test_labels,
        y_pred=test_preds,
        energy_raw=energy_test,
        out_path=output_dir / "figure5_error_localization.png",
        bins=args.energy_bins,
    )
    print("Generating Figure 6...", flush=True)
    _figure6_probability_calibration(
        test_probs=test_probs,
        test_labels=test_labels,
        out_path=output_dir / "figure6_probability_calibration.png",
    )
    print("Generating Figure 7...", flush=True)
    embedding_method = _figure7_embedding_projection(
        X_test=X_test,
        y_true=test_labels,
        y_pred=test_preds,
        out_path=output_dir / "figure7_embedding_projection.png",
    )

    manifest = {
        "checkpoint": str(checkpoint_path),
        "history": str(history_path),
        "best_threshold": best_threshold,
        "embedding_method": embedding_method,
        "figures": [
            "figure1_feature_separability.png",
            "figure2_training_dynamics.png",
            "figure3_threshold_analysis.png",
            "figure4_discrimination_curves.png",
            "figure5_error_localization.png",
            "figure6_probability_calibration.png",
            "figure7_embedding_projection.png",
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print("Saved analysis figures:")
    for fig_name in manifest["figures"]:
        print(f"  - {output_dir / fig_name}")
    print(f"Manifest: {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
