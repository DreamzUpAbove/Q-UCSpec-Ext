"""
SHQNN Model V3 - Improved Quantum Advantage Architecture
"""

from typing import cast

import torch
import torch.nn as nn

try:
    from .post_quantum_nn import PostQuantumNN
    from .pre_quantum_nn import PreQuantumNN
    from .quantum_layer import QuantumLayer
except ImportError:
    from post_quantum_nn import PostQuantumNN  # type: ignore[no-redef]
    from pre_quantum_nn import PreQuantumNN  # type: ignore[no-redef]
    from quantum_layer import QuantumLayer  # type: ignore[no-redef]


class S_HQNN(nn.Module):
    """
    Hybrid Quantum Neural Network V3.
    """

    def __init__(
        self,
        input_dim: int = 3,
        n_qubits: int = 3,
        n_layers: int = 4,
        n_repeats: int = 1,
        use_gpu: bool = False,
        use_hidden_pre: bool = False,
    ):
        """
        Initialize S_HQNN V3.

        Args:
            input_dim: Number of input features (default: 3)
            n_qubits: Number of qubits (default: 3, matching 3 features)
            n_layers: Number of quantum layers (default: 4)
            n_repeats: IQP embedding repeats (default: 1)
            use_gpu: Use GPU for quantum simulation
            use_hidden_pre: Add small hidden layer in Pre-NN
        """
        super().__init__()

        self.input_dim = input_dim
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # Quantum processing (3 qubits, unified IQP)
        self.quantum_layer = QuantumLayer(
            n_qubits=n_qubits, n_layers=n_layers, n_repeats=n_repeats, use_gpu=use_gpu
        )

        # Minimal pre-processing
        self.pre_quantum = PreQuantumNN(
            input_dim=input_dim, output_dim=n_qubits, use_hidden=use_hidden_pre
        )

        # Compact post-processing (3→8→1)
        self.post_quantum = PostQuantumNN(input_dim=n_qubits, hidden_dim=8)

        # Count parameters
        pre_params = sum(p.numel() for p in self.pre_quantum.parameters())
        qua_params = sum(p.numel() for p in self.quantum_layer.parameters())
        post_params = sum(p.numel() for p in self.post_quantum.parameters())
        total = pre_params + qua_params + post_params

        print(f"\n{'=' * 50}")
        print("S_HQNN V3 INITIALIZED")
        print(f"{'=' * 50}")
        print(f"  Pre-NN params:  {pre_params}")
        print(f"  Quantum params: {qua_params}")
        print(f"  Post-NN params: {post_params}")
        print(f"  Total params:   {total}")
        print(f"{'=' * 50}\n")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through SHQNN V3.

        Args:
            x: Input tensor (batch_size, 3)

        Returns:
            Logits (batch_size, 1)
        """
        device = x.device

        # Stage 1: Pre-quantum (minimal transform)
        pre_out = self.pre_quantum(x)

        # Stage 2: Quantum processing
        quantum_out = self.quantum_layer(pre_out)

        quantum_out = quantum_out.to(device)

        # Stage 3: Post-quantum classification
        logits = self.post_quantum(quantum_out)

        return cast(torch.Tensor, logits)

    def freeze_quantum(self) -> None:
        """Freeze quantum layer parameters."""
        for param in self.quantum_layer.parameters():
            param.requires_grad = False
        print("✓ Quantum layer frozen")

    def unfreeze_quantum(self) -> None:
        """Unfreeze quantum layer parameters."""
        for param in self.quantum_layer.parameters():
            param.requires_grad = True
        print("✓ Quantum layer unfrozen")

    def get_model_info(self) -> dict[str, int]:
        """Get model configuration info."""
        return {
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
            "pre_params": sum(p.numel() for p in self.pre_quantum.parameters()),
            "quantum_params": sum(p.numel() for p in self.quantum_layer.parameters()),
            "post_params": sum(p.numel() for p in self.post_quantum.parameters()),
            "total_params": sum(p.numel() for p in self.parameters()),
        }
