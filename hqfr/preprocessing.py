"""
Data Preprocessing for HQFr

Uses 3 valid spectral features for CaF₂/CaF₂:Er classification.
Features are transformed using StandardScaler for zero-mean, unit-variance.

Features:
  1. α (Absorption cm^-1)  - Absorption coefficient
  2. κ (Extinction coeff)  - Extinction coefficient
  3. Energy (eV)           - Photon energy
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]
from pathlib import Path

from config import (
    DATA_DIR,
    CAF2_FILE,
    CAF2_ER_FILE,
    FEATURE_COLUMNS,
    VAL_RATIO,
    TEST_RATIO,
    RANDOM_STATE,
)


def load_and_preprocess(data_dir=None, random_state=None, verbose=True):
    """
    Load and preprocess CaF₂/CaF₂:Er spectral data.

    Args:
        data_dir: Path to data directory (default: uses config.DATA_DIR)
        random_state: Random seed for reproducibility (default: config.RANDOM_STATE)
        verbose: Print preprocessing summary

    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test, transformer, pos_weight)
    """
    if data_dir is None:
        data_dir = DATA_DIR
    else:
        data_dir = Path(data_dir)

    if random_state is None:
        random_state = RANDOM_STATE

    # Load datasets
    caf2_df = pd.read_csv(data_dir / CAF2_FILE)
    caf2_er_df = pd.read_csv(data_dir / CAF2_ER_FILE)

    # Labels: 0 = CaF₂ (host), 1 = CaF₂:Er (doped)
    caf2_df["label"] = 0
    caf2_er_df["label"] = 1

    # Merge datasets
    df = pd.concat([caf2_df, caf2_er_df], ignore_index=True)

    # Extract features and labels
    X = df[FEATURE_COLUMNS].values
    y = df["label"].values

    # Calculate pos_weight for class imbalance handling
    n_class_0 = int(np.sum(y == 0))
    n_class_1 = int(np.sum(y == 1))
    pos_weight = n_class_0 / n_class_1

    # Stratified split: 70% train, 15% val, 15% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_RATIO, stratify=y, random_state=random_state
    )

    # Split remaining into train/val
    val_ratio_adjusted = VAL_RATIO / (1 - TEST_RATIO)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_ratio_adjusted,
        stratify=y_temp,
        random_state=random_state,
    )

    # transformer = PowerTransformer(method='box-cox', standardize=True) # use PowerTransformer for max accuracy
    transformer = StandardScaler()
    X_train = transformer.fit_transform(X_train)
    X_val = transformer.transform(X_val)
    X_test = transformer.transform(X_test)

    if verbose:
        print("=" * 60)
        print("HQFr DATA PREPROCESSING")
        print("=" * 60)
        print("\nDataset loaded:")
        print(f"  Train: {len(X_train)} samples")
        print(f"  Val:   {len(X_val)} samples")
        print(f"  Test:  {len(X_test)} samples")
        print(f"\nFeatures ({len(FEATURE_COLUMNS)}):")
        for i, col in enumerate(FEATURE_COLUMNS):
            print(f"    {i + 1}. {col}")
        print(f"\nClass distribution (train): {np.bincount(y_train)}")
        print(f"Pos weight: {pos_weight:.4f}")
        print("Preprocessing: StandardScaler (zero-mean, unit-variance)")
        print("=" * 60)

    return X_train, X_val, X_test, y_train, y_val, y_test, transformer, pos_weight
