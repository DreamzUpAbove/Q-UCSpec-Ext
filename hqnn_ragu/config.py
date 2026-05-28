# Configuration file for HQNN

from pathlib import Path

# Model parameters
N_QUBITS = 3  # Number of Qubits
N_LAYERS = 4  #  Number of StronglyEntanglingLayers
N_REPEATS = 1  # Number of IQP embeddings repeats

ENTANGLEMENT_RANGES = [1, 1, 2, 2]

# Training parameters
EPOCHS = 50
BATCH_SIZE = 64
PATIENCE = 15
USE_GPU_QUANTUM = False  # Set True if lightning.gpu available

# Layerwise learning rates
LR_PRE = 0.001
LR_QUANTUM = 0.0005
LR_POST = 0.001

# Regularization parameters
DROPOUT_RATE = 0.15
MAX_GRAD_NORM = 1.0

# Data configuration
# Features used (3 valid features)
FEATURE_COLUMNS = [
    "κ (Extinction coeff)",  # Extinction coefficient
    "α (Absorption cm^-1)",  # Absorption coefficient
    "Energy (eV)",  # Energy
    # 'ε₂ (Imag dielectric)',     # Imaginary part of dielectric
    # 'n (Refractive index)',     # Refractive index
    # 'OscStrength'               # Oscillator strength
]

# Data files (relative to project root)
DATA_DIR = Path(__file__).parent.parent / "data"
CAF2_FILE = "caf2_qml_full_descriptors.csv"
CAF2_ER_FILE = "caf2_er_qml_full_descriptors.csv"

# Train/Val/Test split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SEED = 42
RANDOM_STATE = 42


def set_seed(seed: int = SEED) -> None:
    """Set all random seeds for reproducibility."""
    import random

    import numpy as np
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
