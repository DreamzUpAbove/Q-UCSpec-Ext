"""
Pre-Quantum Neural Network

MINIMAL architecture to prevent classical dominance.
Just a linear transform from 6 features to 6 qubits.

"""

import torch.nn as nn

from config import N_QUBITS


class PreQuantumNN(nn.Module):
    """
    Minimal pre-quantum processing - just a linear transform.
    """

    def __init__(self, input_dim=6, output_dim=None, use_hidden=False):
        """
        Initialize minimal Pre-Quantum NN.

        Args:
            input_dim: Number of input features (default: 6)
            output_dim: Number of outputs = qubits (default: N_QUBITS from config)
            use_hidden: If True, add small hidden layer for feature mixing
        """
        super().__init__()

        if output_dim is None:
            output_dim = N_QUBITS

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_hidden = use_hidden

        if use_hidden:
            self.net = nn.Sequential(
                nn.Linear(input_dim, 8),
                nn.ReLU(),
                nn.Linear(8, output_dim),
            )
            n_params = input_dim * 8 + 8 + 8 * output_dim + output_dim
        else:
            self.net = nn.Linear(input_dim, output_dim)
            n_params = input_dim * output_dim + output_dim

        # Force float64 for precision
        self.net = self.net.double()

        print("✓ PreQuantumNN initialized")
        print(f"    Architecture: {input_dim}→{'8→' if use_hidden else ''}{output_dim}")
        print(f"    Parameters: {n_params}")

    def forward(self, x):
        """Forward pass - simple transform, no scaling."""
        return self.net(x)
