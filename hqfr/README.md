# Q-UCSpec Integrating TDDFT Spectroscopy and Quantum Machine Learning for Photonic Upconversion Materials

**Q-UCSpec** is a QAMP 2025 project that integrates **first-principles Linear Response (Lr) Time-Dependent Density Functional Theory (TDDFT)** simulations with **Quantum Machine Learning (QML)** to explore the optical spectra of **photonic upconversion materials**.  

 Base project - https://github.com/DennisWayo/Q-UCSpec.git

The project focuses on:
This project Aims to extend that research with new hybrid Quantum-Classical combined architecture

---

# Input Features
Three spectral descriptors extracted from LR-TDDFT simulations (defined in config.py):

1. α (Absorption cm^-1) — Absorption coefficient
2. κ (Extinction coeff) — Extinction coefficient
3. Energy (eV) — Photon energy

# Preprocessing
- Data loaded from two CSV files: caf2_qml_full_descriptors.csv (label=0) and caf2_er_qml_full_descriptors.csv (label=1)
- Stratified split: 70% train / 15% val / 15% test
- StandardScaler fitted on train set, applied to val and test
- Class imbalance handled via pos_weight = n_class_0 / n_class_1 in BCEWithLogitsLoss

# Two-Phase Training Strategy
## Phase 1: Classical Pre-Training
- Trainable: Pre-NN, Post-NN, circuit_scale
- Frozen: Quantum Layer
- Optimizer: Adam with per-group learning rates (lr_pre=0.001, lr_post=0.001)
- Scheduler: ReduceLROnPlateau (patience=5, factor=0.5)
- Gradient clipping: max_norm=1.0
- Early stopping: patience=15
- Default epochs: 30
## Phase 2: Quantum Fine-Tuning
- Trainable: Quantum Layer only
- Frozen: Pre-NN, Post-NN, circuit_scale
- Optimizer: Adam (lr=0.0003 for multi-Pauli, lr=0.001 for single-Pauli — auto-selected)
- Scheduler: ReduceLROnPlateau (patience=10, factor=0.5)
- Gradient clipping: max_norm=0.5
- Early stopping: patience=10
- Default epochs: 100

After Phase 2, the optimal classification threshold is found by sweeping 0.30–0.70 (step 0.02) and maximizing F1-score on the validation set.

![Pipeline Architecture](images/architecture_preview.png)

### Repository Structure

```
hqfr
├── config.py
├── preprocessing.py
├── pre_quantum_nn.py
├── quantum_layer.py
├── post_quantum_nn.py
├── hqfr_model.py
├── train_hqfr.py
├── main_hqfr.py
├── README.md
└── demo_notebook.ipynb
└── data/ 
```

Data for this project is taken from https://github.com/DennisWayo/Q-UCSpec/data


### Mentorship
Mentor: [Dennis Wayo](https://github.com/DennisWayo)

QAMP 2025 Project: Q-UCSpec

Mentees: [DavidAlba2627](https://github.com/DavidAlba2627),  [keremyurtseven](https://github.com/keremyurtseven),  [Siriapps](https://github.com/Siriapps), [Alireza Alipour](https://github.com/AlirezaAlipour-ghb),  [GHOST-Q1](https://github.com/GHOST-Q1),  [DreamzUpAbove](https://github.com/DreamzUpAbove),[Reema Alzaid](https://github.com/ReemaAlzaid), [Krishan Sharma](https://github.com/Krishan019).
