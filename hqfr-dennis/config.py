# HQFr Hybrid Pretrain frozen model configuration

from pathlib import Path

N_QUBITS = 3
N_LAYERS = 4
N_REPEATS = 3
CIRCUIT_SCALE_INIT = 1.0
USE_IQP = False
USE_MULTI_PAULI = True
INIT_STRATEGY = "uniform"

# Post-processing dimensions
POST_NN_HIDDEN = 16
DROPOUT_RATE = 0.3

# Entanglement ranges (Auto-adjusted in QuantumLayer if None)
ENTANGLEMENT_RANGES = None

# Training Configuration
EPOCHS_PHASE1 = 30
EPOCHS_PHASE2 = 100
BATCH_SIZE = 64
PATIENCE = 15
MAX_GRAD_NORM = 1.0
MAX_GRAD_NORM_QUANTUM = 0.5

# Layerwise learning rates
LR_PRE = 0.001
LR_QUANTUM = 0.001
LR_POST = 0.001

# Specific LRs for Phase 2
LR_QUANTUM_PHASE2_MULTI_PAULI = 0.0003
LR_QUANTUM_PHASE2_SINGLE_PAULI = 0.001

# Layerwise learning rates
LR_PRE = 0.001
LR_QUANTUM = 0.001
LR_POST = 0.001

# Regularization
DROPOUT_RATE = 0.15  # In PostQuantumNN
MAX_GRAD_NORM = 1.0  # Gradient clipping

# ------------------------------------------------------------
# Data Configuration
# ------------------------------------------------------------
RANDOM_STATE = 42

# Top 3 features
FEATURE_COLUMNS = [
    "α (Absorption cm^-1)",  # Absorption coefficient
    "κ (Extinction coeff)",  # Extinction coefficient
    "Energy (eV)",  # Energy
]

# Data files (relative to project root)
DATA_DIR = Path(__file__).parent / "data"
CAF2_FILE = "caf2_qml_full_descriptors.csv"
CAF2_ER_FILE = "caf2_er_qml_full_descriptors.csv"

# Train/Val/Test split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Quantum Simulation
USE_GPU_QUANTUM = False  # Set True if lightning.gpu available
DIFF_METHOD = "backprop"  # "backprop" for simulator

# Reproducibility
SEED = 42


def set_seed(seed=SEED):
    """Set all random seeds for reproducibility."""
    import torch
    import numpy as np
    import random

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
