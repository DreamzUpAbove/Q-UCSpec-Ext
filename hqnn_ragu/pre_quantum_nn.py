"""
Pre-Quantum Neural Network
"""

from typing import cast

import torch
import torch.nn as nn


class PreQuantumNN(nn.Module):
    """
    Minimal pre-quantum processing - just a linear transform.
    """

    net: nn.Sequential | nn.Linear

    def __init__(
        self, input_dim: int = 3, output_dim: int = 3, use_hidden: bool = False
    ) -> None:
        """
        Initialize minimal Pre-Quantum NN.
        Args:
            input_dim: Number of input features (default: 3)
            output_dim: Number of outputs = qubits (default: 3)
            use_hidden: If True, add small hidden layer for mixing
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_hidden = use_hidden

        if use_hidden:
            # Minimal mixing: input→8→output
            self.net = nn.Sequential(
                nn.Linear(input_dim, 8),
                nn.ReLU(),
                nn.Linear(8, output_dim),
            )
            n_params = input_dim * 8 + 8 + 8 * output_dim + output_dim
        else:
            # Direct mapping
            self.net = nn.Linear(input_dim, output_dim)
            n_params = input_dim * output_dim + output_dim

        # Force float64
        self.net = self.net.double()

        print("PreQuantumNNv3 initialized")
        print(f"Architecture: {input_dim}→{'8→' if use_hidden else ''}{output_dim}")
        print(f"Parameters: {n_params}")
        print("NOTE: No scaling applied - quantum layer will scale to [0,π]")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - simple transform, no scaling.
        The quantum layer will apply scale_to_phase() on these outputs.
        """
        return cast(torch.Tensor, self.net(x))
