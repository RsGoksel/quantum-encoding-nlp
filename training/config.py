"""
Training Configuration
======================
Tum hyperparametreler burada merkezi olarak tanimlanir.
Reproducibility icin her deney ayni config'den okur.
"""

from dataclasses import dataclass


@dataclass
class TrainingConfig:
    # Data
    dataset: str = "imdb"          # "imdb" veya "sst2"
    pca_dim: int = 8               # PCA boyutu (= qubit sayisi)

    # Model
    model_type: str = "quantum"    # "baseline", "attention", "quantum", "quantum_random"
    n_qubits: int = 8
    ansatz_reps: int = 2           # EfficientSU2 tekrar katmani

    # Data subset
    use_subset: bool = False       # True = 2000 sample subset (adil karsilastirma)

    # Training
    epochs: int = 30
    batch_size: int = 64
    lr_classical: float = 0.001    # Klasik parametreler icin
    lr_quantum: float = 0.01       # Quantum parametreler icin (daha yuksek!)
    weight_decay: float = 1e-4
    seed: int = 42

    # Scheduler
    use_scheduler: bool = True
    scheduler_patience: int = 5    # Val loss iyilesmezse lr dusur
    scheduler_factor: float = 0.5

    # Early stopping
    early_stop_patience: int = 10  # Val loss iyilesmezse dur

    # Output
    output_suffix: str = ""    # e.g., "_seed123" for multi-seed experiments

    # Paths
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "results/logs"
