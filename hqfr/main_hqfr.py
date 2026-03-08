"""
Main Entry Point for HQFr Training (Pretrain-Frozen Strategy)
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import torch
from torch.utils.data import TensorDataset, DataLoader

from config import BATCH_SIZE, LR_PRE, LR_POST, RANDOM_STATE, set_seed, POST_NN_HIDDEN
from preprocessing import load_and_preprocess
from hqfr_model import HQFr
from train_hqfr import train_hqfr, evaluate, EPOCHS_PHASE1, EPOCHS_PHASE2


def main():
    parser = argparse.ArgumentParser(description="HQFr Training (Pretrain-Frozen)")
    parser.add_argument(
        "--epochs-phase1",
        type=int,
        default=EPOCHS_PHASE1,
        help="Phase 1 epochs (classical)",
    )
    parser.add_argument(
        "--epochs-phase2",
        type=int,
        default=EPOCHS_PHASE2,
        help="Phase 2 epochs (quantum)",
    )
    parser.add_argument(
        "--lr-pre", type=float, default=LR_PRE, help="Pre-NN learning rate"
    )
    parser.add_argument(
        "--lr-post", type=float, default=LR_POST, help="Post-NN learning rate"
    )
    parser.add_argument(
        "--lr-quantum",
        type=float,
        default=None,
        help="Quantum layer learning rate (Phase 2). Default: Auto-tuned.",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument(
        "--save-dir", type=str, default="checkpoints_hqfr", help="Checkpoint directory"
    )
    args = parser.parse_args()

    # Reproducibility
    set_seed(RANDOM_STATE)
    torch.set_default_dtype(torch.float64)

    print("=" * 70)
    print("HQFr - HYBRID QUANTUM NN WITH PRETRAIN-FROZEN STRATEGY")
    print("=" * 70)
    print("\nArchitecture (HQFr-style):")
    print("  ✓ Pre-NN: 3→3")
    print("  ✓ Quantum: 3 qubits, 4 layers (AngleEmbed + SEL)")
    print("  ✓ Post-NN: 9→16→1")
    print("\nTwo-Phase Training Strategy:")
    print("  Phase 1: Classical pre-training (quantum FROZEN)")
    print("  Phase 2: Quantum fine-tuning (classical FROZEN)")
    print("\nConfiguration:")
    print(f"  Epochs Phase 1: {args.epochs_phase1}")
    print(f"  Epochs Phase 2: {args.epochs_phase2}")
    print(f"  LR Pre-NN:  {args.lr_pre}")
    print(f"  LR Post-NN: {args.lr_post}")
    print(f"  LR Quantum: {args.lr_quantum}")
    print("=" * 70)

    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test, transformer, pos_weight = (
        load_and_preprocess()
    )

    # Create datasets
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float64),
        torch.tensor(y_train, dtype=torch.float64),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float64),
        torch.tensor(y_val, dtype=torch.float64),
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float64),
        torch.tensor(y_test, dtype=torch.float64),
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # Initialize HQFr model
    model = HQFr(post_nn_hidden=POST_NN_HIDDEN)

    # Train with 2-phase strategy
    history = train_hqfr(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        pos_weight=pos_weight,
        epochs_phase1=args.epochs_phase1,
        epochs_phase2=args.epochs_phase2,
        lr_pre=args.lr_pre,
        lr_post=args.lr_post,
        lr_quantum=args.lr_quantum,
        device=args.device,
        save_dir=args.save_dir,
        verbose=True,
    )

    # Evaluate
    print("\nEvaluating on test set...")
    results = evaluate(
        model=model,
        test_loader=test_loader,
        threshold=history["best_threshold"],
        device=args.device,
        verbose=True,
    )

    print("=" * 70)
    print("HQFr EXPERIMENT COMPLETE")
    print(f"  Final Accuracy: {results['accuracy']:.4f}")
    print(f"  Final MCC: {results['mcc']:.4f}")
    print(f"  Circuit Gain: {model.circuit_scale.item():.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
