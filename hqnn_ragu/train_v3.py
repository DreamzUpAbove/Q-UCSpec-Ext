"""
Training Pipeline V3 - Co-Training Strategy
"""

import json
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


class TrainHistory(TypedDict):
    train_loss: list[float]
    val_loss: list[float]
    train_acc: list[float]
    val_acc: list[float]
    best_threshold: float


def find_best_threshold(
    model: nn.Module, val_loader: DataLoader[Any], device: str = "cpu"
) -> float:
    """Find optimal classification threshold using F1-score on validation set."""
    model.eval()
    all_probs: list[float] = []
    all_labels: list[float] = []
    with torch.no_grad():
        for batch_x, _batch_y in val_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            probs_np = torch.sigmoid(outputs).squeeze().cpu().numpy()
            if probs_np.ndim == 0:
                probs_list: list[float] = [float(probs_np.item())]
            else:
                probs_list = probs_np.tolist()
            all_probs.extend(probs_list)
            all_labels.extend(_batch_y.numpy().tolist())  # ← this line was missing
    probs_arr: np.ndarray[Any, np.dtype[Any]] = np.array(all_probs)
    labels_arr: np.ndarray[Any, np.dtype[Any]] = np.array(all_labels)

    best_threshold = 0.5
    best_f1 = 0.0

    for threshold in np.arange(0.3, 0.7, 0.02):
        preds = (probs_arr > threshold).astype(float)
        tp = ((preds == 1) & (labels_arr == 1)).sum()
        fp = ((preds == 1) & (labels_arr == 0)).sum()
        fn = ((preds == 0) & (labels_arr == 1)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold


def _train_epoch(
    model: nn.Module,
    train_loader: DataLoader[Any],
    criterion: nn.Module,
    optimizer: Optimizer,
    device: str,
) -> tuple[float, float]:
    """Single training epoch."""
    model.train()
    train_loss, train_correct, train_total = 0.0, 0, 0

    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device).unsqueeze(1).float()

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += loss.item()
        preds = (torch.sigmoid(outputs) > 0.5).float()
        train_correct += (preds == batch_y).sum().item()
        train_total += batch_y.size(0)

    return train_loss / len(train_loader), train_correct / train_total


def _validate(
    model: nn.Module,
    val_loader: DataLoader[Any],
    criterion: nn.Module,
    device: str,
) -> tuple[float, float]:
    """Validation."""
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = (
                batch_x.to(device),
                batch_y.to(device).unsqueeze(1).float(),
            )
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            val_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            val_correct += (preds == batch_y).sum().item()
            val_total += batch_y.size(0)

    return val_loss / len(val_loader), val_correct / val_total


def train_v3(
    model: nn.Module,
    train_loader: DataLoader[Any],
    val_loader: DataLoader[Any],
    pos_weight: float = 1.0,
    epochs: int = 30,
    lr_pre: float = 0.001,
    lr_quantum: float = 0.0005,
    lr_post: float = 0.001,
    device: str = "cpu",
    save_dir: str | Path = "checkpoints_v3",
    patience: int = 10,
) -> TrainHistory:
    """
    V3 Training Strategy - Co-Training from Start.

    Args:
        model: S_HQNN model
        train_loader: Training data loader
        val_loader: Validation data loader
        pos_weight: Class weight for imbalance
        epochs: Number of training epochs
        lr_pre: Learning rate for Pre-NN
        lr_quantum: Learning rate for Quantum layer (lower)
        lr_post: Learning rate for Post-NN
        device: Training device
        save_dir: Directory to save checkpoints
        patience: Early stopping patience
    """

    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True, parents=True)
    model = model.to(device)

    # Loss with class balancing
    pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float64).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    # Layerwise learning rates
    param_groups = [
        {
            "params": model.pre_quantum.parameters(),
            "lr": lr_pre,
            "name": "pre_quantum",
        },
        {
            "params": model.quantum_layer.parameters(),
            "lr": lr_quantum,
            "name": "quantum",
        },
        {
            "params": model.post_quantum.parameters(),
            "lr": lr_post,
            "name": "post_quantum",
        },
    ]

    optimizer = torch.optim.Adam(param_groups)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    history: TrainHistory = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "best_threshold": 0.5,
    }

    best_val_loss = float("inf")
    patience_counter = 0

    print(f"\n{'=' * 60}")
    print("SHQNN V3 TRAINING - Co-Training Strategy")
    print(f"{'=' * 60}")
    print(f"  Epochs: {epochs}")
    print(f"  LR Pre-NN: {lr_pre}")
    print(f"  LR Quantum: {lr_quantum}")
    print(f"  LR Post-NN: {lr_post}")
    print(f"  Pos weight: {pos_weight:.4f}")
    print(f"  Device: {device}")
    print(f"{'=' * 60}\n")

    for epoch in range(epochs):
        train_loss, train_acc = _train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = _validate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Log every 2 epochs or first/last
        if (epoch + 1) % 2 == 0 or epoch == 0 or epoch == epochs - 1:
            print(
                f"Epoch {epoch + 1:3d}/{epochs} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
            )

        # Early stopping and checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                },
                save_path / "best_model.pth",
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⏹ Early stopping at epoch {epoch + 1}")
                break

    # Load the best model to evaluate the threshold correctly
    print("\n  Loading best model to compute optimal threshold...")
    checkpoint = torch.load(save_path / "best_model.pth", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Find optimal threshold
    print("  Finding optimal classification threshold...")
    best_threshold = find_best_threshold(model, val_loader, device)
    history["best_threshold"] = best_threshold
    print(f"  Optimal threshold: {best_threshold:.2f}")

    # Save history
    with open(save_path / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Training completed. Best val loss: {best_val_loss:.4f}")
    print(f"{'=' * 60}\n")

    return history


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader[Any],
    threshold: float = 0.5,
    device: str = "cpu",
) -> dict[str, Any]:
    """Evaluate model on test set with comprehensive metrics."""
    model.eval()
    all_preds: list[float] = []
    all_labels: list[float] = []
    all_probs: list[float] = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            probs_np = torch.sigmoid(outputs).squeeze().cpu().numpy()
            if probs_np.ndim == 0:
                probs_list: list[float] = [float(probs_np.item())]
            else:
                probs_list = probs_np.tolist()
            preds = (np.array(probs_list) > threshold).astype(float).tolist()
            all_probs.extend(probs_list)
            all_preds.extend(preds)
            all_labels.extend(batch_y.numpy().tolist())

    # Use NEW variables for numpy arrays — don't reassign the lists
    preds_arr: np.ndarray[Any, np.dtype[Any]] = np.array(all_preds)
    labels_arr: np.ndarray[Any, np.dtype[Any]] = np.array(all_labels)

    # Metrics — use preds_arr and labels_arr everywhere below
    tp = ((preds_arr == 1) & (labels_arr == 1)).sum()
    tn = ((preds_arr == 0) & (labels_arr == 0)).sum()
    fp = ((preds_arr == 1) & (labels_arr == 0)).sum()
    fn = ((preds_arr == 0) & (labels_arr == 1)).sum()

    accuracy = (tp + tn) / len(labels_arr)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom > 0 else 0

    print(f"\n{'=' * 50}")
    print(f"TEST RESULTS (threshold={threshold:.2f})")
    print(f"{'=' * 50}")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  MCC:       {mcc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print("\n  Confusion Matrix:")
    print(f"    TP: {int(tp):4d}  FP: {int(fp):4d}")
    print(f"    FN: {int(fn):4d}  TN: {int(tn):4d}")
    print(f"{'=' * 50}\n")

    return {
        "accuracy": accuracy,
        "f1": f1,
        "mcc": mcc,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": {
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
        },
    }
