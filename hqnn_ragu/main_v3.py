"""
Main Entry Point for S_HQNN V3 Training

Usage:
    python -m main_v3 --epochs 30 --batch_size 64
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

try:
    from .preprocessing import load_and_preprocess
    from .shqnn_model import S_HQNN
    from .train_v3 import evaluate_model, train_v3
except ImportError:
    from preprocessing import load_and_preprocess  # type: ignore[no-redef]
    from shqnn_model import S_HQNN  # type: ignore[no-redef]
    from train_v3 import evaluate_model, train_v3  # type: ignore[no-redef]


def main() -> None:
    parser = argparse.ArgumentParser(description="HQNN V3 Training - Quantum Advantage")

    # Training parameters
    parser.add_argument("--epochs", type=int, default=30, help="Total training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument(
        "--lr_pre", type=float, default=0.001, help="Learning rate for Pre-NN"
    )
    parser.add_argument(
        "--lr_quantum",
        type=float,
        default=0.0005,
        help="Learning rate for Quantum layer",
    )
    parser.add_argument(
        "--lr_post", type=float, default=0.001, help="Learning rate for Post-NN"
    )
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cpu", "cuda"]
    )
    parser.add_argument(
        "--patience", type=int, default=10, help="Early stopping patience"
    )
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)

    # Model parameters
    parser.add_argument(
        "--n_qubits",
        type=int,
        default=3,
        help="Number of qubits (should match features)",
    )
    parser.add_argument(
        "--n_layers", type=int, default=4, help="Number of quantum layers"
    )
    parser.add_argument(
        "--n_repeats", type=int, default=1, help="IQP embedding repeats"
    )
    parser.add_argument(
        "--use_gpu_quantum", action="store_true", help="Use GPU for quantum simulation"
    )
    parser.add_argument(
        "--use_hidden_pre", action="store_true", help="Add hidden layer in Pre-NN"
    )

    args = parser.parse_args()

    # Device selection
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    torch.set_default_dtype(torch.float64)

    # Reproducibility
    try:
        from .config import set_seed
    except ImportError:
        from config import set_seed  # type: ignore[no-redef]

    set_seed()

    # Create save directory
    if args.save_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        args.save_dir = f"checkpoints_v3_{timestamp}"

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("HQNN V3 - IMPROVED QUANTUM ADVANTAGE ARCHITECTURE")
    print("=" * 70)
    print("\nKey Features:")
    print("  ✓ Uses all 3 valid features")
    print(f"  ✓ {args.n_qubits} qubits for 1:1 feature mapping")
    print("  ✓ Minimal Pre-NN (prevents classical dominance)")
    print("  ✓ Unified IQP encoding (proven best)")
    print("  ✓ Co-training from start (no Phase 0 freeze)")
    print("\nConfiguration:")
    print(f"  Qubits: {args.n_qubits}")
    print(f"  Quantum layers: {args.n_layers}")
    print(f"  IQP repeats: {args.n_repeats}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  LR Pre/Quantum/Post: {args.lr_pre}/{args.lr_quantum}/{args.lr_post}")
    print(f"  Device: {device}")
    print(f"  Save dir: {args.save_dir}")
    print()

    # Load data
    print("Loading data...")
    X_train, X_val, X_test, y_train, y_val, y_test, transformer, pos_weight = (
        load_and_preprocess(data_dir=args.data_dir)
    )

    # Data loaders
    train_dataset = TensorDataset(
        torch.DoubleTensor(X_train), torch.LongTensor(y_train)
    )
    val_dataset = TensorDataset(torch.DoubleTensor(X_val), torch.LongTensor(y_val))
    test_dataset = TensorDataset(torch.DoubleTensor(X_test), torch.LongTensor(y_test))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # Initialize model
    print("\nInitializing SHQNN V3...")
    model = S_HQNN(
        input_dim=X_train.shape[1],
        n_qubits=args.n_qubits,
        n_layers=args.n_layers,
        n_repeats=args.n_repeats,
        use_gpu=args.use_gpu_quantum,
        use_hidden_pre=args.use_hidden_pre,
    )

    # Training
    print("\nStarting training...")
    history = train_v3(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        pos_weight=pos_weight,
        epochs=args.epochs,
        lr_pre=args.lr_pre,
        lr_quantum=args.lr_quantum,
        lr_post=args.lr_post,
        device=device,
        save_dir=args.save_dir,
        patience=args.patience,
    )

    # Load best model and evaluate
    print("\nLoading best model for evaluation...")
    checkpoint = torch.load(Path(args.save_dir) / "best_model.pth", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    best_threshold: float = history["best_threshold"]

    metrics = evaluate_model(
        model, test_loader, device=device, threshold=best_threshold
    )

    # Add experiment info
    metrics["experiment"] = {
        "version": "v3",
        "n_qubits": args.n_qubits,
        "n_layers": args.n_layers,
        "n_repeats": args.n_repeats,
        "timestamp": datetime.now().isoformat(),
    }

    # Save metrics
    with open(Path(args.save_dir) / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'=' * 70}")
    print("TRAINING COMPLETE")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1-Score: {metrics['f1']:.4f}")
    print(f"  Results saved to: {args.save_dir}/")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
