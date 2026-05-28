"""
Data Preprocessing for HQNN
"""

from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from .config import (
        CAF2_ER_FILE,
        CAF2_FILE,
        DATA_DIR,
        FEATURE_COLUMNS,
        RANDOM_STATE,
        TEST_RATIO,
        VAL_RATIO,
    )
except ImportError:
    from config import (  # type: ignore[no-redef]
        CAF2_ER_FILE,
        CAF2_FILE,
        DATA_DIR,
        FEATURE_COLUMNS,
        RANDOM_STATE,
        TEST_RATIO,
        VAL_RATIO,
    )


def load_and_preprocess(
    data_dir: str | Path | None = None,
    random_state: int | None = None,
    verbose: bool = True,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.int64],
    StandardScaler,
    float,
]:
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
    y_arr: NDArray[np.int64] = np.asarray(y, dtype=np.int64)

    # Stratified split: 70% train, 15% val, 15% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y_arr, test_size=TEST_RATIO, stratify=y_arr, random_state=random_state
    )

    # Split remaining into train/val
    val_ratio_adjusted = VAL_RATIO / (1 - TEST_RATIO)
    y_temp_arr: NDArray[np.int64] = np.asarray(y_temp, dtype=np.int64)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp_arr,
        test_size=val_ratio_adjusted,
        stratify=y_temp_arr,
        random_state=random_state,
    )

    # Calculate pos_weight on train split only
    y_train_arr: NDArray[np.int64] = np.asarray(y_train, dtype=np.int64)
    n_class_0 = int((y_train_arr == 0).sum())
    n_class_1 = int((y_train_arr == 1).sum())
    pos_weight = n_class_0 / n_class_1

    # PowerTransformer for Gaussian-like distribution
    # transformer = PowerTransformer(method='box-cox', standardize=True)
    # StandardScaler for minimal transformation (to prevent classical dominance)
    transformer = StandardScaler()
    X_train = transformer.fit_transform(X_train)
    X_val = transformer.transform(X_val)
    X_test = transformer.transform(X_test)

    if verbose:
        print("=" * 60)
        print("HQNN DATA PREPROCESSING")
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
        print("=" * 60)

    # return X_train, X_val, X_test, y_train, y_val, y_test, transformer, pos_weight
    return (
        np.array(X_train, dtype=np.float64),
        np.array(X_val, dtype=np.float64),
        np.array(X_test, dtype=np.float64),
        np.array(y_train, dtype=np.int64),
        np.array(y_val, dtype=np.int64),
        np.array(y_test, dtype=np.int64),
        transformer,
        float(pos_weight),
    )
