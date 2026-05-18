"""
Enhanced Quantum Layer
1. Identity initialization (Grant et al., Quantum 2019)
2. IQP embedding (Havlíček et al., Nature 2019)
3. Multi-Pauli measurements (broader information capture)
"""

from typing import Optional

import torch
import torch.nn as nn
import pennylane as qml  # type: ignore[import-untyped]
import numpy as np

from config import N_QUBITS, N_LAYERS, N_REPEATS, USE_GPU_QUANTUM


class QuantumLayer(nn.Module):
    """
    - Identity-like initialization (avoids barren plateaus)
    - IQP embedding (second-order feature interactions)
    - Multi-Pauli measurements (captures X, Y, Z directions)
    """

    def __init__(
        self,
        n_qubits: Optional[int] = None,
        n_layers: Optional[int] = None,
        n_repeats: Optional[int] = None,
        entanglement_ranges: Optional[list] = None,
        use_gpu: Optional[bool] = None,
        use_iqp: Optional[bool] = True,
        use_multi_pauli: Optional[bool] = True,
        init_strategy: Optional[str] = "identity",
        shots: Optional[int] = None,
        backend=None,
    ):
        super().__init__()

        # Use config defaults
        self.n_qubits = n_qubits if n_qubits is not None else N_QUBITS
        self.n_layers = n_layers if n_layers is not None else N_LAYERS
        self.n_repeats = n_repeats if n_repeats is not None else N_REPEATS
        use_gpu = use_gpu if use_gpu is not None else USE_GPU_QUANTUM

        self.use_iqp = use_iqp
        self.use_multi_pauli = use_multi_pauli
        self.init_strategy = init_strategy

        # Handle entanglement ranges
        if entanglement_ranges is not None:
            self.ranges = entanglement_ranges
        else:
            from config import ENTANGLEMENT_RANGES

            if ENTANGLEMENT_RANGES and len(ENTANGLEMENT_RANGES) == self.n_layers:
                self.ranges = ENTANGLEMENT_RANGES
            else:
                self.ranges = [
                    1 if i < self.n_layers // 2 else min(2, self.n_qubits - 1)
                    for i in range(self.n_layers)
                ]

        # Verify ranges match layers
        if len(self.ranges) != self.n_layers:
            # Auto-extend if possible
            if len(self.ranges) < self.n_layers:
                self.ranges = self.ranges + [self.ranges[-1]] * (
                    self.n_layers - len(self.ranges)
                )
            else:
                self.ranges = self.ranges[: self.n_layers]

        # Device selection
        self.shots = shots
        diff_method = "backprop"

        if backend is not None:
            self.dev = qml.device(
                "qiskit.aer", wires=self.n_qubits, backend=backend, shots=shots
            )
            diff_method = "parameter-shift"
            print(f"✓ Quantum device: qiskit.aer (noisy, shots={shots})")
        elif shots is not None:
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

        # Output dimension
        self.n_outputs = self.n_qubits * 3 if use_multi_pauli else self.n_qubits

        # Weight shape: (n_layers, n_qubits, 3)
        weight_shapes = {"weights": (self.n_layers, self.n_qubits, 3)}
        self.qlayer = qml.qnn.TorchLayer(self.qnode, weight_shapes)

        # Initialize with research-backed strategy
        self._initialize_weights()

        n_params = self.n_layers * self.n_qubits * 3
        print("✓ EnhancedQuantumLayer initialized")
        print(f"    Qubits: {self.n_qubits}")
        print(f"    Layers: {self.n_layers}")
        print(f"    Embedding: {'IQP (2nd-order)' if use_iqp else 'Angle (1st-order)'}")
        print(f"    Repeats: {self.n_repeats}")
        print(f"    Observables: {'X,Y,Z (3x)' if use_multi_pauli else 'Z only'}")
        print(f"    Initialization: {init_strategy}")
        print(f"    Entanglement: {self.ranges}")
        print(f"    Parameters: {n_params}")
        print(f"    Output dim: {self.n_outputs}")

    def _initialize_weights(self):
        """
        Options:
        - 'identity': Layer-wise identity (Grant et al., Quantum 2019)
        - 'beta': Beta distribution (Kulshrestha et al., 2025)
        - 'gaussian': Gaussian with problem-specific variance
        - 'uniform': Original uniform (fallback)
        """
        if self.init_strategy == "identity":
            # Identity-like initialization to avoid barren plateaus
            with torch.no_grad():
                for layer_idx in range(self.n_layers):
                    if layer_idx == 0:
                        # First layer: exact identity (Rot(0,0,0) = I)
                        self.qlayer.weights.data[layer_idx] = 0.0
                    else:
                        # Other layers: small random around zero
                        # Scale increases with depth for gradual complexity
                        scale = 0.01 * (1 + layer_idx / self.n_layers)
                        self.qlayer.weights.data[layer_idx] = (
                            torch.randn(self.n_qubits, 3, dtype=torch.float64) * scale
                        )

        elif self.init_strategy == "beta":
            # Beta distribution initialization for binary classification
            from torch.distributions import Beta

            beta_dist = Beta(2.0, 5.0)  # Skewed distribution
            with torch.no_grad():
                samples = beta_dist.sample((self.n_layers, self.n_qubits, 3))
                # Map [0,1] to parameter range
                self.qlayer.weights.data = (samples - 0.5) * np.pi

        elif self.init_strategy == "gaussian":
            # Gaussian with variance tuned for quantum circuits
            with torch.no_grad():
                for p in self.qlayer.parameters():
                    nn.init.normal_(p, mean=0.0, std=0.1)

        else:  # 'uniform' or unknown
            # Original uniform initialization
            with torch.no_grad():
                for p in self.qlayer.parameters():
                    nn.init.uniform_(p, -0.01, 0.01)

    def _circuit(self, inputs, weights):
        """
        Structure per layer:
        1. IQP embedding (creates x_i * x_j nonlinear terms) OR Angle embedding
        2. StronglyEntanglingLayers with controlled range
        3. Multi-Pauli measurement (X, Y, Z) OR single Z
        """
        n_qubits = self.n_qubits

        # Data re-uploading: encode data before each variational layer
        for layer_idx in range(self.n_layers):
            # Step 1: Feature embedding
            if self.use_iqp:
                # IQP embedding creates second-order feature interactions
                # H + RZ(x_i) + RZZ(x_i * x_j)
                qml.IQPEmbedding(
                    features=inputs,
                    wires=range(n_qubits),
                    n_repeats=self.n_repeats,
                )
            else:
                # Standard angle embedding (RY rotations)
                qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")

            # Step 2: Variational layer with controlled entanglement
            sel_weights = weights[layer_idx : layer_idx + 1]
            qml.StronglyEntanglingLayers(
                sel_weights, wires=range(n_qubits), ranges=[self.ranges[layer_idx]]
            )

        # Step 3: Measurement
        if self.use_multi_pauli:
            # Measure all three Pauli operators per qubit
            measurements = []
            for i in range(n_qubits):
                measurements.append(qml.expval(qml.PauliZ(i)))
                measurements.append(qml.expval(qml.PauliX(i)))
                measurements.append(qml.expval(qml.PauliY(i)))
            return measurements  # Returns 3*n_qubits values
        else:
            # Original: only Z measurement
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    def forward(self, x):
        """
        Forward pass with feature scaling.

        Args:
            x: Input from PreQuantumNN (raw values, standardized)

        Returns:
            Quantum expectation values (n_qubits or 3*n_qubits dimensions)
        """
        # Scale inputs to [0, π] for phase encoding
        x_scaled = torch.sigmoid(x) * np.pi

        # Quantum processing
        return self.qlayer(x_scaled)
