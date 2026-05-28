"""
Quantum Layer - Unified IQP Architecture
"""

from typing import Any, cast

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
from pennylane.measurements import ExpectationMP

try:
    from .config import (
        ENTANGLEMENT_RANGES,
        N_LAYERS,
        N_QUBITS,
        N_REPEATS,
        USE_GPU_QUANTUM,
    )
except ImportError:
    from config import (
        ENTANGLEMENT_RANGES,
        N_LAYERS,
        N_QUBITS,
        N_REPEATS,
        USE_GPU_QUANTUM,
    )  # type: ignore[no-redef]


def scale_to_phase(
    x: torch.Tensor, min_phase: float = 0.0, max_phase: float = np.pi
) -> torch.Tensor:
    """
    Scale standardized inputs to phase range [min_phase, max_phase].
    Uses sigmoid for clean [0, 1] mapping without extreme values.

    Args:
        x: Input tensor (standardized, mean≈0, std≈1)
        min_phase: Minimum phase value (default: 0)
        max_phase: Maximum phase value (default: π)

    Returns:
        Scaled tensor in [min_phase, max_phase]
    """
    x_mapped = torch.sigmoid(x)  # Maps to (0, 1)
    x_phase = min_phase + x_mapped * (max_phase - min_phase)
    return x_phase


class QuantumLayer(nn.Module):
    """
    Unified IQP quantum layer.
    """

    def __init__(
        self,
        n_qubits: int | None = None,
        n_layers: int | None = None,
        n_repeats: int | None = None,
        ranges: list[int] | None = None,
        use_gpu: bool | None = None,
        shots: int | None = None,
        backend: Any | None = None,
    ) -> None:
        """
        Initialize quantum layer.

        Args:
            n_qubits: Number of qubits (default: config.N_QUBITS)
            n_layers: Number of StronglyEntanglingLayers (default: config.N_LAYERS)
            n_repeats: IQP embedding repeats (default: config.N_REPEATS)
            ranges: Entanglement ranges per layer (default: config.ENTANGLEMENT_RANGES)
            use_gpu: Use GPU simulator if available (default: config.USE_GPU_QUANTUM)
            shots: Number of shots for finite sampling (None=analytical)
            backend: Qiskit backend for noisy simulation (None=default.qubit)
        """
        super().__init__()

        # Use config defaults if not specified
        self.n_qubits = n_qubits if n_qubits is not None else N_QUBITS
        self.n_layers = n_layers if n_layers is not None else N_LAYERS
        self.n_repeats = n_repeats if n_repeats is not None else N_REPEATS
        use_gpu = use_gpu if use_gpu is not None else USE_GPU_QUANTUM

        # Default: gradual increase in entanglement range
        if ranges is None:
            if ENTANGLEMENT_RANGES and len(ENTANGLEMENT_RANGES) == self.n_layers:
                self.ranges = ENTANGLEMENT_RANGES
            else:
                # [1, 1, 2, 2] for 4 layers: starts local, becomes more global
                self.ranges = [
                    1 if i < self.n_layers // 2 else min(2, self.n_qubits - 1)
                    for i in range(self.n_layers)
                ]
        else:
            if len(ranges) != self.n_layers:
                raise ValueError(f"len(ranges) must equal n_layers ({self.n_layers})")
            self.ranges = ranges

        # Device selection
        self.shots = shots
        diff_method = "backprop"

        if backend is not None:
            # Use provided Qiskit backend
            self.dev = qml.device(
                "qiskit.aer", wires=self.n_qubits, backend=backend, shots=shots
            )
            diff_method = "parameter-shift"
            print(f"✓ Quantum device: qiskit.aer (noisy, shots={shots})")
        elif shots is not None:
            # Use default.qubit with finite shots
            self.dev = qml.device("default.qubit", wires=self.n_qubits, shots=shots)
            diff_method = "parameter-shift"
            print(f"✓ Quantum device: default.qubit (shots={shots})")
        elif use_gpu:
            try:
                self.dev = qml.device("lightning.gpu", wires=self.n_qubits)
                print("✓ Quantum device: lightning.gpu (CUDA)")
            except Exception:
                self.dev = qml.device("default.qubit", wires=self.n_qubits)
                print("✓ Quantum device: default.qubit (CPU fallback)")
        else:
            self.dev = qml.device("default.qubit", wires=self.n_qubits)
            print("✓ Quantum device: default.qubit (CPU)")

        # Quantum circuit
        self.qnode = qml.QNode(
            self._circuit, self.dev, interface="torch", diff_method=diff_method
        )

        # Weight shape: (n_layers, n_qubits, 3) - applied ALL AT ONCE
        weight_shapes = {"weights": (self.n_layers, self.n_qubits, 3)}
        # self.qlayer = TorchLayer(self.qnode, weight_shapes)
        self.qlayer: nn.Module = qml.qnn.TorchLayer(self.qnode, weight_shapes)  # type: ignore[reportCallIssue]

        # Identity-like initialization (small values near 0)
        self._init_identity_weights()

        n_params = self.n_layers * self.n_qubits * 3
        print("✓ QuantumLayer initialized")
        print(f"    Qubits: {self.n_qubits}")
        print(f"    IQP repeats: {self.n_repeats}")
        print(f"    StronglyEntanglingLayers: {self.n_layers} (unified)")
        print(f"    Entanglement ranges: {self.ranges}")
        print(f"    Parameters: {n_params}")
        print("    Feature scaling: sigmoid → [0, 2π]")

    def _init_identity_weights(self) -> None:
        """Initialize weights near zero for identity-like behavior."""
        for p in self.qlayer.parameters():
            nn.init.uniform_(p, -0.01, 0.01)

    def _circuit(
        self, inputs: torch.Tensor, weights: torch.Tensor
    ) -> list[ExpectationMP]:
        """
        Unified IQP circuit architecture.

        Structure:
        1. IQP embedding (fixed feature map)
        2. StronglyEntanglingLayers (trainable variational)
        3. PauliZ measurements
        """
        # IQP embedding: creates non-linear x_i * x_j terms
        qml.IQPEmbedding(inputs, wires=range(self.n_qubits), n_repeats=self.n_repeats)

        # All variational layers applied at once with controlled ranges
        qml.StronglyEntanglingLayers(
            weights, wires=range(self.n_qubits), ranges=self.ranges
        )

        # Return PauliZ expectation on all qubits
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with feature scaling.

        Args:
            x: Input from PreQuantumNN (raw values, standardized)

        Returns:
            Quantum expectation values (n_qubits dimensions)
        """
        # Scale inputs to [0, 2π] for phase encoding
        x_scaled = scale_to_phase(x, min_phase=0.0, max_phase=2 * np.pi)

        # Quantum processing
        # return self.qlayer(x_scaled)
        return cast(torch.Tensor, self.qlayer(x_scaled))
