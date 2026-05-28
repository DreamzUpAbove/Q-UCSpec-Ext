"""
Post-Quantum Neural Network
"""

from typing import cast

import torch
import torch.nn as nn


class PostQuantumNN(nn.Module):
    """
    Compact post-quantum classifier
    """

    def __init__(self, input_dim: int = 3, hidden_dim: int = 8) -> None:
        """
        Initialize post-quantum classifier.

        Args:
            input_dim: Number of quantum outputs (= n_qubits, default: 3)
            hidden_dim: Hidden layer size (default: 8)
        """
        super().__init__()

        self.net = nn.Sequential(
            # Layer 1: Interpret quantum measurements
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.15),
            # Output: Raw logits for BCEWithLogitsLoss
            nn.Linear(hidden_dim, 1),
        )

        # Force float64 to match quantum layer
        self.net = self.net.double()

        # n_params = input_dim * hidden_dim + hidden_dim + hidden_dim * 1 + 1 + hidden_dim  # includes batchnorm
        print("PostQuantumNNv3 initialized")
        print(f"Architecture: {input_dim}→{hidden_dim}→1")
        print("Output: Raw logits (use BCEWithLogitsLoss)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass - outputs raw logits."""
        return cast(torch.Tensor, self.net(x))
