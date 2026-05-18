"""
HQFr Model - Hybrid QNN with Pretrain-Frozen Strategy

Based on HQFr Pretrain-Frozen research (arXiv 2025):
- Phase 1: Classical pre-training (quantum frozen)
- Phase 2: Quantum training (classical frozen)
"""

import torch
import torch.nn as nn

# LOCAL imports - use data re-uploading quantum layer from this folder
from config import (
    N_QUBITS,
    N_LAYERS,
    N_REPEATS,
    CIRCUIT_SCALE_INIT,
    USE_IQP,
    USE_MULTI_PAULI,
    INIT_STRATEGY,
    POST_NN_HIDDEN,
)
from pre_quantum_nn import PreQuantumNN
from quantum_layer import QuantumLayer
from post_quantum_nn import PostQuantumNN


class HQFr(nn.Module):
    """
    Hybrid Quantum Neural Network with Pretrain-Frozen Strategy (HQFr).

    Two-phase training:
    - Phase 1: Train classical layers (Pre-NN, Post-NN, circuit_scale) while quantum is FROZEN
    - Phase 2: Train quantum layer while classical layers are FROZEN
    """

    def __init__(
        self,
        n_qubits: int | None = None,
        n_layers: int | None = None,
        n_repeats: int | None = None,
        circuit_scale_init: float | None = None,
        use_gpu: bool | None = False,
        use_hidden_pre: bool | None = False,
        shots: int | None = None,
        use_iqp: bool | None = None,
        use_multi_pauli: bool | None = None,
        init_strategy: str | None = None,
        post_nn_hidden: int | None = POST_NN_HIDDEN,
        entanglement_ranges: list | None = None,
        backend=None,
    ):
        super().__init__()

        # Use config defaults
        self.n_qubits = n_qubits if n_qubits is not None else N_QUBITS
        self.n_layers = n_layers if n_layers is not None else N_LAYERS
        n_repeats = n_repeats if n_repeats is not None else N_REPEATS
        circuit_scale_init = (
            circuit_scale_init if circuit_scale_init is not None else CIRCUIT_SCALE_INIT
        )
        use_hidden_pre = use_hidden_pre if use_hidden_pre is not None else False
        post_nn_hidden = (
            post_nn_hidden if post_nn_hidden is not None else POST_NN_HIDDEN
        )

        # Stage 1: Pre-processing (input features = n_qubits)
        self.pre_quantum = PreQuantumNN(
            input_dim=self.n_qubits,  # Input features match qubits
            output_dim=self.n_qubits,
            use_hidden=use_hidden_pre,
        )

        # Stage 2: Quantum processing
        self.quantum_layer = QuantumLayer(
            n_qubits=self.n_qubits,
            n_layers=self.n_layers,
            n_repeats=n_repeats,
            use_gpu=use_gpu,
            use_iqp=use_iqp if use_iqp is not None else USE_IQP,
            use_multi_pauli=(
                use_multi_pauli if use_multi_pauli is not None else USE_MULTI_PAULI
            ),
            init_strategy=init_strategy if init_strategy is not None else INIT_STRATEGY,
            entanglement_ranges=entanglement_ranges,
            shots=shots,
            backend=backend,
        )

        # Stage 3: Post-processing
        # Input dim depends on multi-Pauli (3x) or single (1x)
        self.post_quantum = PostQuantumNN(
            input_dim=self.quantum_layer.n_outputs, hidden_dim=post_nn_hidden
        )

        # Circuit gain
        self.circuit_scale = nn.Parameter(
            torch.tensor(circuit_scale_init, dtype=torch.float64)
        )

        # Track freeze state
        self._quantum_frozen = False
        self._classical_frozen = False

        # Count parameters
        pre_params = sum(p.numel() for p in self.pre_quantum.parameters())
        qua_params = sum(p.numel() for p in self.quantum_layer.parameters())
        post_params = sum(p.numel() for p in self.post_quantum.parameters())
        total = pre_params + qua_params + post_params + 1

        print(f"\n{'=' * 60}")
        print("HQFr INITIALIZED (Pretrain-Frozen Strategy)")
        print(f"{'=' * 60}")
        print(f"  Pre-NN params:     {pre_params:4d}")
        print(f"  Quantum params:    {qua_params:4d}")
        print(f"  Post-NN params:    {post_params:4d}")
        print("  Circuit gain:      1")
        print("  -----------------------------------")
        print(f"  Total params:      {total:4d}")
        print(f"  Initial QG value:  {circuit_scale_init}")
        print("\n  Training Strategy:")
        print("    Phase 1: Classical pre-training (quantum frozen)")
        print("    Phase 2: Quantum training (classical frozen)")
        print(f"{'=' * 60}\n")

    def forward(self, x):
        """Forward pass through HQFr."""
        device = x.device

        # Stage 1: Pre-quantum
        pre_out = self.pre_quantum(x)

        # Stage 2: Quantum processing
        quantum_out = self.quantum_layer(pre_out)

        # Apply circuit gain scaling
        quantum_out = quantum_out.to(device)
        scaled_quantum = self.circuit_scale * quantum_out

        # Stage 3: Post-quantum classification
        logits = self.post_quantum(scaled_quantum)

        return logits

    # ==================== FREEZE/UNFREEZE METHODS ====================

    def freeze_quantum(self):
        """Freeze quantum layer parameters."""
        for param in self.quantum_layer.parameters():
            param.requires_grad = False
        self._quantum_frozen = True
        print("✓ Quantum layer FROZEN")

    def unfreeze_quantum(self):
        """Unfreeze quantum layer parameters."""
        for param in self.quantum_layer.parameters():
            param.requires_grad = True
        self._quantum_frozen = False
        print("✓ Quantum layer UNFROZEN")

    def freeze_classical(self):
        """Freeze all classical layers (Pre-NN, Post-NN, circuit_scale)."""
        for param in self.pre_quantum.parameters():
            param.requires_grad = False
        for param in self.post_quantum.parameters():
            param.requires_grad = False
        self.circuit_scale.requires_grad = False
        self._classical_frozen = True
        print("✓ Classical layers FROZEN (Pre-NN, Post-NN, circuit_scale)")

    def unfreeze_classical(self):
        """Unfreeze classical layers."""
        for param in self.pre_quantum.parameters():
            param.requires_grad = True
        for param in self.post_quantum.parameters():
            param.requires_grad = True
        self.circuit_scale.requires_grad = True
        self._classical_frozen = False
        print("✓ Classical layers UNFROZEN")

    def freeze_circuit_scalen(self):
        """Freeze circuit_scale parameter only."""
        self.circuit_scale.requires_grad = False
        print(f"✓ circuit_scale frozen at {self.circuit_scale.item():.4f}")

    def unfreeze_circuit_scale(self):
        """Unfreeze circuit_scale parameter."""
        self.circuit_scale.requires_grad = True
        print("✓ circuit_scaleunfrozen")

    # ==================== CONFIGURATION FOR PHASES ====================

    def prepare_phase1(self):
        """
        Prepare for Phase 1: Classical Pre-training.
        - Freeze quantum layer
        - Unfreeze classical layers
        """
        print("\n" + "=" * 50)
        print("PHASE 1: Classical Pre-training")
        print("=" * 50)
        self.unfreeze_classical()
        self.freeze_quantum()

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Trainable parameters: {trainable}")
        return trainable

    def prepare_phase2(self):
        """
        Prepare for Phase 2: Quantum Training.
        - Freeze classical layers
        - Unfreeze quantum layer
        """
        print("\n" + "=" * 50)
        print("PHASE 2: Quantum Training")
        print("=" * 50)
        self.freeze_classical()
        self.unfreeze_quantum()

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Trainable parameters: {trainable}")
        return trainable

    def get_phase1_params(self):
        """Get parameters for Phase 1 optimizer (classical only)."""
        params = []
        params.extend(list(self.pre_quantum.parameters()))
        params.extend(list(self.post_quantum.parameters()))
        params.append(self.circuit_scale)
        return [p for p in params if p.requires_grad]

    def get_phase2_params(self):
        """Get parameters for Phase 2 optimizer (quantum only)."""
        return [p for p in self.quantum_layer.parameters() if p.requires_grad]

    def get_model_info(self):
        """Get model configuration info."""
        return {
            "type": "HQFr (Pretrain-Frozen)",
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
            "circuit_scale": self.circuit_scale.item(),
            "quantum_frozen": self._quantum_frozen,
            "classical_frozen": self._classical_frozen,
            "pre_params": sum(p.numel() for p in self.pre_quantum.parameters()),
            "quantum_params": sum(p.numel() for p in self.quantum_layer.parameters()),
            "post_params": sum(p.numel() for p in self.post_quantum.parameters()),
            "trainable_params": sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
        }
