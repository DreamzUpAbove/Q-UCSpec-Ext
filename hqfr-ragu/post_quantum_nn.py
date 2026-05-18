"""
Post-Quantum Neural Network
"""

import torch.nn as nn

from config import N_QUBITS, DROPOUT_RATE


class PostQuantumNN(nn.Module):
    """
    Compact post-quantum classifier.
    """

    def __init__(self, input_dim=None, hidden_dim=8, dropout=None):
        """
        Initialize post-quantum classifier.

        Args:
            input_dim: Number of quantum outputs (default: config.N_QUBITS)
            hidden_dim: Hidden layer size (default: 8)
            dropout: Dropout rate (default: config.DROPOUT_RATE)
        """
        super().__init__()

        if input_dim is None:
            input_dim = N_QUBITS
        if dropout is None:
            dropout = DROPOUT_RATE

        self.net = nn.Sequential(
            # Layer 1: Interpret quantum measurements
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            # Output: Raw logits for BCEWithLogitsLoss
            nn.Linear(hidden_dim, 1),
        )

        # Force float64 to match quantum layer
        self.net = self.net.double()

        # n_params = input_dim * hidden_dim + hidden_dim + hidden_dim * 1 + 1 + hidden_dim
        print("✓ PostQuantumNN initialized")
        print(f"    Architecture: {input_dim}→{hidden_dim}→1")
        print(f"    Dropout: {dropout}")
        print("    Output: Raw logits (use BCEWithLogitsLoss)")

    def forward(self, x):
        """Forward pass - outputs raw logits."""
        return self.net(x)
