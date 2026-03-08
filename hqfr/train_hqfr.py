"""
Two-Phase Training Pipeline for HQFr (Pretrain-Frozen Strategy)

Training Strategy:
- Phase 1: Classical pre-training (30 epochs default)
  - Quantum layer FROZEN
  - Pre-NN, Post-NN, circuit_scale optimized with Adam

- Phase 2: Quantum fine-tuning (20 epochs default)
  - Classical layers FROZEN
  - Only quantum layer optimized with lower learning rate

Based on arXiv 2025 HQNN Pretrain-Frozen and Transfer Learning research
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc  # type: ignore[import-untyped]

from config import (
    LR_PRE,
    LR_POST,
    MAX_GRAD_NORM,
    MAX_GRAD_NORM_QUANTUM,
    EPOCHS_PHASE1,
    EPOCHS_PHASE2,
    LR_QUANTUM_PHASE2_MULTI_PAULI,
    LR_QUANTUM_PHASE2_SINGLE_PAULI,
)


def find_best_threshold(model, val_loader, device="cpu"):
    """Find optimal classification threshold using F1-score."""
    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            probs = torch.sigmoid(outputs).squeeze().cpu().numpy()

            if probs.ndim == 0:
                probs = [probs.item()]
            else:
                probs = probs.tolist()
            all_probs.extend(probs)
            all_labels.extend(batch_y.numpy().tolist())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    best_threshold = 0.5
    best_f1 = 0.0

    for threshold in np.arange(0.3, 0.7, 0.02):
        preds = (all_probs > threshold).astype(float)
        tp = ((preds == 1) & (all_labels == 1)).sum()
        fp = ((preds == 1) & (all_labels == 0)).sum()
        fn = ((preds == 0) & (all_labels == 1)).sum()

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
    model, train_loader, criterion, optimizer, device, max_grad_norm=MAX_GRAD_NORM
):
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()

        train_loss += loss.item()
        preds = (torch.sigmoid(outputs) > 0.5).float()
        train_correct += (preds == batch_y).sum().item()
        train_total += batch_y.size(0)

    return train_loss / len(train_loader), train_correct / train_total


def _validate(model, val_loader, criterion, device):
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


def train_hqfr(
    model,
    train_loader,
    val_loader,
    pos_weight=1.0,
    epochs_phase1=EPOCHS_PHASE1,
    epochs_phase2=EPOCHS_PHASE2,
    lr_pre=None,
    lr_post=None,
    lr_quantum=None,
    device="cpu",
    save_dir="checkpoints_hqfr",
    patience_phase1=15,
    patience_phase2=10,
    verbose=True,
):
    """
    Two-phase HQFr training.

    Phase 1: Classical pre-training
    Phase 2: Quantum fine-tuning

    Args:
        model: HQFr model
        train_loader: Training data loader
        val_loader: Validation data loader
        pos_weight: Class weight for imbalance
        epochs_phase1: Epochs for classical pre-training
        epochs_phase2: Epochs for quantum fine-tuning
        lr_pre, lr_post: Learning rates for classical layers (Phase 1)
        lr_quantum: Learning rate for quantum layer (Phase 2)
        device: Training device
        save_dir: Directory to save checkpoints
        patience_phase1, patience_phase2: Early stopping patience per phase
        verbose: Print progress

    Returns:
        Training history dict
    """
    # Use config defaults
    lr_pre = lr_pre if lr_pre is not None else LR_PRE
    lr_post = lr_post if lr_post is not None else LR_POST
    lr_quantum = lr_quantum if lr_quantum is not None else None

    if lr_quantum is None:
        if hasattr(model, "quantum_layer") and model.quantum_layer.use_multi_pauli:
            lr_quantum = LR_QUANTUM_PHASE2_MULTI_PAULI
            if verbose:
                print(f"  > Auto-tuned LR for Multi-Pauli: {lr_quantum}")
        else:
            lr_quantum = LR_QUANTUM_PHASE2_SINGLE_PAULI
            if verbose:
                print(f"  > Auto-tuned LR for Single-Pauli: {lr_quantum}")

    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    model = model.to(device)

    # Loss with class balancing
    pos_weight_tensor = torch.tensor([pos_weight]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    history = {
        "phase1": {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "circuit_scale": [],
        },
        "phase2": {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "circuit_scale": [],
        },
        "best_threshold": 0.5,
    }

    # ==================== PHASE 1: Classical Pre-training ====================
    if verbose:
        print(f"\n{'=' * 70}")
        print("HQFr PHASE 1: CLASSICAL PRE-TRAINING")
        print(f"{'=' * 70}")
        print("  Strategy: Train Pre-NN, Post-NN, circuit_scale")
        print("  Quantum layer: FROZEN")
        print(f"  Epochs: {epochs_phase1}")
        print(f"  LR Pre-NN: {lr_pre}")
        print(f"  LR Post-NN: {lr_post}")
        print(f"{'=' * 70}\n")

    model.prepare_phase1()

    optimizer_phase1 = torch.optim.Adam(
        [
            {
                "params": model.pre_quantum.parameters(),
                "lr": lr_pre,
                "name": "pre_quantum",
            },
            {
                "params": model.post_quantum.parameters(),
                "lr": lr_post,
                "name": "post_quantum",
            },
            {"params": [model.circuit_scale], "lr": lr_post, "name": "circuit_scale"},
        ]
    )
    scheduler_phase1 = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_phase1, patience=5, factor=0.5
    )

    best_val_loss_phase1 = float("inf")
    patience_counter = 0

    for epoch in range(epochs_phase1):
        train_loss, train_acc = _train_epoch(
            model,
            train_loader,
            criterion,
            optimizer_phase1,
            device,
            max_grad_norm=MAX_GRAD_NORM,
        )
        val_loss, val_acc = _validate(model, val_loader, criterion, device)

        scheduler_phase1.step(val_loss)

        cg = model.circuit_scale.item()

        history["phase1"]["train_loss"].append(train_loss)
        history["phase1"]["train_acc"].append(train_acc)
        history["phase1"]["val_loss"].append(val_loss)
        history["phase1"]["val_acc"].append(val_acc)
        history["phase1"]["circuit_scale"].append(cg)

        if verbose and (
            (epoch + 1) % 5 == 0 or epoch == 0 or epoch == epochs_phase1 - 1
        ):
            print(
                f"[P1] Epoch {epoch + 1:3d}/{epochs_phase1} | "
                f"Train: {train_loss:.4f}/{train_acc:.4f} | "
                f"Val: {val_loss:.4f}/{val_acc:.4f} | CG: {cg:.4f}"
            )

        # Checkpointing
        if val_loss < best_val_loss_phase1:
            best_val_loss_phase1 = val_loss
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "phase": 1,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                },
                save_dir / "best_model_phase1.pth",
            )
        else:
            patience_counter += 1
            if patience_counter >= patience_phase1:
                if verbose:
                    print(f"\\n⏹ Phase 1 early stopping at epoch {epoch + 1}")
                break

    if verbose:
        print(f"\\n  Phase 1 complete. Best val loss: {best_val_loss_phase1:.4f}")
        print(f"  Circuit gain after Phase 1: {model.circuit_scale.item():.4f}")

    # ==================== PHASE 2: Quantum Fine-tuning ====================
    if verbose:
        print(f"\\n{'=' * 70}")
        print("HQFr PHASE 2: QUANTUM FINE-TUNING")
        print(f"{'=' * 70}")
        print("  Strategy: Train quantum layer only")
        print("  Classical layers: FROZEN")
        print(f"  Epochs: {epochs_phase2}")
        print(f"  LR Quantum: {lr_quantum}")
        print(f"{'=' * 70}\\n")

    model.prepare_phase2()

    optimizer_phase2 = torch.optim.Adam(
        [
            {
                "params": model.quantum_layer.parameters(),
                "lr": lr_quantum,
                "name": "quantum",
            },
        ]
    )
    scheduler_phase2 = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_phase2, patience=10, factor=0.5
    )

    best_val_loss_phase2 = float("inf")
    patience_counter = 0

    for epoch in range(epochs_phase2):
        train_loss, train_acc = _train_epoch(
            model,
            train_loader,
            criterion,
            optimizer_phase2,
            device,
            max_grad_norm=MAX_GRAD_NORM_QUANTUM,
        )
        val_loss, val_acc = _validate(model, val_loader, criterion, device)

        scheduler_phase2.step(val_loss)

        qg = model.circuit_scale.item()

        history["phase2"]["train_loss"].append(train_loss)
        history["phase2"]["train_acc"].append(train_acc)
        history["phase2"]["val_loss"].append(val_loss)
        history["phase2"]["val_acc"].append(val_acc)
        history["phase2"]["circuit_scale"].append(qg)

        if verbose and (
            (epoch + 1) % 5 == 0 or epoch == 0 or epoch == epochs_phase2 - 1
        ):
            print(
                f"[P2] Epoch {epoch + 1:3d}/{epochs_phase2} | "
                f"Train: {train_loss:.4f}/{train_acc:.4f} | "
                f"Val: {val_loss:.4f}/{val_acc:.4f} | CG: {qg:.4f}"
            )

        # Checkpointing
        if val_loss < best_val_loss_phase2:
            best_val_loss_phase2 = val_loss
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "phase": 2,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                },
                save_dir / "best_model_phase2.pth",
            )
        else:
            patience_counter += 1
            if patience_counter >= patience_phase2:
                if verbose:
                    print(f"\n⏹ Phase 2 early stopping at epoch {epoch + 1}")
                break

    # Find optimal threshold
    if verbose:
        print("\n  Finding optimal classification threshold...")
    best_threshold = find_best_threshold(model, val_loader, device)
    history["best_threshold"] = best_threshold
    if verbose:
        print(f"  Optimal threshold: {best_threshold:.2f}")

    # Save final model and history
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "history": history,
        },
        save_dir / "final_model.pth",
    )

    with open(save_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    if verbose:
        print(f"\n{'=' * 70}")
        print("HQFr TRAINING COMPLETE")
        print(f"  Phase 1 best val loss: {best_val_loss_phase1:.4f}")
        print(f"  Phase 2 best val loss: {best_val_loss_phase2:.4f}")
        print(f"  Final circuit_scale: {model.circuit_scale.item():.4f}")
        print(f"{'=' * 70}\n")

    return history


def evaluate(model, test_loader, threshold=0.5, device="cpu", verbose=True):
    """Evaluate model on test set with comprehensive metrics."""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            probs = torch.sigmoid(outputs).squeeze().cpu().numpy()

            if probs.ndim == 0:
                probs = [probs.item()]
            else:
                probs = probs.tolist()

            preds = (np.array(probs) > threshold).astype(float).tolist()
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(batch_y.numpy().tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Metrics
    tp = ((all_preds == 1) & (all_labels == 1)).sum()
    tn = ((all_preds == 0) & (all_labels == 0)).sum()
    fp = ((all_preds == 1) & (all_labels == 0)).sum()
    fn = ((all_preds == 0) & (all_labels == 1)).sum()

    accuracy = (tp + tn) / len(all_labels)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

    # MCC
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom > 0 else 0

    if verbose:
        print(f"\n{'=' * 50}")
        print(f"HQFr TEST RESULTS (threshold={threshold:.2f})")
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

    # Compute ROC-AUC and PR-AUC
    all_probs_arr = np.array(all_probs)
    if len(np.unique(all_labels)) > 1:
        roc_auc = roc_auc_score(all_labels, all_probs_arr)
        pr_precision, pr_recall, _ = precision_recall_curve(all_labels, all_probs_arr)
        pr_auc = auc(pr_recall, pr_precision)
    else:
        roc_auc = 0.0
        pr_auc = 0.0

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
        "probabilities": all_probs,
        "predictions": all_preds.tolist(),
        "labels": all_labels.tolist(),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }
