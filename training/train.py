"""
Training Pipeline
=================
Tum modeller (baseline, attention, quantum, encoding variants) icin ortak training loop.

Kullanim:
    PYTHONIOENCODING=utf-8 python training/train.py --model baseline --dataset imdb
    PYTHONIOENCODING=utf-8 python training/train.py --model quantum --dataset imdb
    PYTHONIOENCODING=utf-8 python training/train.py --model quantum_random --dataset imdb

    Phase 6 — Encoding karsilastirmasi:
    PYTHONIOENCODING=utf-8 python training/train.py --model enc_angle --dataset imdb --subset
    PYTHONIOENCODING=utf-8 python training/train.py --model enc_dense --dataset imdb --subset
    PYTHONIOENCODING=utf-8 python training/train.py --model enc_iqp --dataset imdb --subset
    PYTHONIOENCODING=utf-8 python training/train.py --model enc_reupload --dataset imdb --subset

    Phase 6 — Quantum Attention:
    PYTHONIOENCODING=utf-8 python training/train.py --model qattn_angle --dataset imdb --subset
    PYTHONIOENCODING=utf-8 python training/train.py --model qattn_iqp --dataset imdb --subset

Cikti:
    - checkpoints/{model}_{dataset}_best.pt
    - results/logs/{model}_{dataset}_history.json
"""

import os
import sys
import json
import time
import argparse
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

os.environ["PYTHONIOENCODING"] = "utf-8"

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.config import TrainingConfig
from models.classical_baseline import ClassicalBaseline
from models.classical_attention import ClassicalAttention
from models.quantum_attention import QuantumAttentionModel
from models.quantum_attention_fast import QuantumAttentionFast
from models.quantum_encodings import QuantumEncodingModel, QuantumAttentionModel as QAttnModel


def set_seed(seed):
    """Reproducibility icin tum random seed'leri ayarla."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(config):
    """PCA embedding'lerini yukle ve DataLoader olustur."""
    data_path = PROJECT_ROOT / "data" / f"{config.dataset}_embeddings.pt"
    # weights_only=False: kendi olusturdugumuz dosya, numpy array iceriyor (pca_variance)
    data = torch.load(data_path, weights_only=False)

    print(f"  Dataset: {config.dataset}")
    print(f"  Train: {data['train_X'].shape}, Val: {data['val_X'].shape}, Test: {data['test_X'].shape}")
    print(f"  PCA varyans: {data['pca_variance'].sum():.4f} ({data['pca_variance'].sum()*100:.1f}%)")

    # DataLoader'lar
    # Quantum modeller icin kucuk subset kullan
    # Neden: Parameter-shift rule gradient'i 2*n_params ekstra circuit evaluation gerektiriyor
    # 48 param = 96 ek evaluation/sample -> backward pass 83x yavasliyor
    # Cozum: 2000 train, 500 val, 2000 test (quantum ML'de standart yaklasim)
    train_X, train_y = data["train_X"], data["train_y"]
    val_X, val_y = data["val_X"], data["val_y"]
    test_X, test_y = data["test_X"], data["test_y"]

    if _is_quantum_model(config.model_type) or config.use_subset:
        subset_train = min(2000, len(train_X))
        subset_val = min(500, len(val_X))
        subset_test = min(2000, len(test_X))

        gen = torch.Generator().manual_seed(config.seed)
        train_perm = torch.randperm(len(train_X), generator=gen)[:subset_train]
        val_perm = torch.randperm(len(val_X), generator=gen)[:subset_val]
        test_perm = torch.randperm(len(test_X), generator=gen)[:subset_test]

        train_X, train_y = train_X[train_perm], train_y[train_perm]
        val_X, val_y = val_X[val_perm], val_y[val_perm]
        test_X, test_y = test_X[test_perm], test_y[test_perm]

        print(f"  Quantum subset: train={subset_train}, val={subset_val}, test={subset_test}")
        print(f"  (Parameter-shift rule nedeniyle kucuk subset gerekli)")

    train_ds = TensorDataset(train_X, train_y)
    val_ds = TensorDataset(val_X, val_y)
    test_ds = TensorDataset(test_X, test_y)

    # Quantum modellerde kucuk batch (parameter-shift rule memory yuzunden)
    bs = config.batch_size
    if _is_quantum_model(config.model_type):
        bs = min(bs, 16)
        print(f"  Batch size: {bs} (quantum icin kucuk batch)")

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=bs)
    test_loader = DataLoader(test_ds, batch_size=bs)

    return train_loader, val_loader, test_loader


def _is_quantum_model(model_type):
    """Quantum model olup olmadigini kontrol et."""
    quantum_types = (
        "quantum", "quantum_random",
        "enc_angle", "enc_angle_rand",
        "enc_dense", "enc_dense_rand",
        "enc_iqp", "enc_iqp_rand",
        "enc_reupload", "enc_reupload_rand",
        "qattn_angle", "qattn_angle_rand",
        "qattn_iqp", "qattn_iqp_rand",
        "qattn_reupload", "qattn_reupload_rand",
    )
    return model_type in quantum_types


def create_model(config):
    """Model type'a gore model olustur."""
    mt = config.model_type

    if mt == "baseline":
        model = ClassicalBaseline(input_dim=config.pca_dim)
        print(f"  Model: ClassicalBaseline ({model.count_params()} params)")

    elif mt == "attention":
        model = ClassicalAttention(input_dim=config.pca_dim)
        print(f"  Model: ClassicalAttention ({model.count_params()} params)")

    elif mt == "quantum":
        model = QuantumAttentionFast(
            n_qubits=config.n_qubits, reps=config.ansatz_reps, identity_init=True,
        )
        params = model.count_params()
        print(f"  Model: QuantumAttentionFast (identity init)")
        print(f"    Quantum: {params['quantum']}, Classical: {params['classical']}, Total: {params['total']}")

    elif mt == "quantum_random":
        model = QuantumAttentionFast(
            n_qubits=config.n_qubits, reps=config.ansatz_reps, identity_init=False,
        )
        params = model.count_params()
        print(f"  Model: QuantumAttentionFast (random init)")
        print(f"    Quantum: {params['quantum']}, Classical: {params['classical']}, Total: {params['total']}")

    # --- Phase 6: Encoding variants (direct classification) ---
    elif mt.startswith("enc_"):
        encoding_map = {
            "enc_angle": ("angle", True),
            "enc_angle_rand": ("angle", False),
            "enc_dense": ("dense_angle", True),
            "enc_dense_rand": ("dense_angle", False),
            "enc_iqp": ("iqp", True),
            "enc_iqp_rand": ("iqp", False),
            "enc_reupload": ("re-uploading", True),
            "enc_reupload_rand": ("re-uploading", False),
        }
        enc_name, id_init = encoding_map[mt]
        model = QuantumEncodingModel(
            encoding=enc_name, n_qubits=config.n_qubits,
            reps=config.ansatz_reps, identity_init=id_init,
        )
        params = model.count_params()
        init_str = "identity" if id_init else "random"
        print(f"  Model: QuantumEncodingModel (encoding={enc_name}, init={init_str})")
        print(f"    Quantum: {params['quantum']}, Classical: {params['classical']}, Total: {params['total']}")

    # --- Phase 6: Quantum Attention (true attention mechanism) ---
    elif mt.startswith("qattn_"):
        attn_map = {
            "qattn_angle": ("angle", True),
            "qattn_angle_rand": ("angle", False),
            "qattn_iqp": ("iqp", True),
            "qattn_iqp_rand": ("iqp", False),
            "qattn_reupload": ("re-uploading", True),
            "qattn_reupload_rand": ("re-uploading", False),
        }
        enc_name, id_init = attn_map[mt]
        model = QAttnModel(
            encoding=enc_name, n_qubits=config.n_qubits,
            reps=config.ansatz_reps, identity_init=id_init,
        )
        params = model.count_params()
        init_str = "identity" if id_init else "random"
        print(f"  Model: QuantumAttention (encoding={enc_name}, init={init_str})")
        print(f"    Quantum: {params['quantum']}, Classical: {params['classical']}, Total: {params['total']}")

    else:
        raise ValueError(f"Unknown model type: {mt}")

    return model


def create_optimizer(model, config):
    """
    Model icin optimizer olustur.

    Quantum modellerde ayri learning rate kullaniyoruz:
    - Quantum parametreler: lr=0.01 (daha agresif — parameter-shift rule yuzunden
      gradyanlar kuculebilir, yuksek lr bunu telafi eder)
    - Klasik parametreler: lr=0.001 (standart)

    Neden ayri lr?
    - Quantum gradyanlar parameter-shift rule ile hesaplaniyor
    - Bu gradyanlar genellikle klasik backprop gradyanlarindan kucuk
    - Ayni lr kullanirsan quantum parametreler cok yavas ogrenir
    """
    if _is_quantum_model(config.model_type):
        # Quantum ve klasik parametreleri ayir
        quantum_params = list(model.circuit.parameters())
        classical_params = list(model.classifier.parameters())

        optimizer = torch.optim.Adam([
            {"params": quantum_params, "lr": config.lr_quantum},
            {"params": classical_params, "lr": config.lr_classical},
        ], weight_decay=config.weight_decay)
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.lr_classical,
            weight_decay=config.weight_decay,
        )

    return optimizer


def train_epoch(model, loader, criterion, optimizer, device):
    """Tek epoch training."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == batch_y).sum().item()
        total += batch_x.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    """Validation/test evaluation."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            total_loss += loss.item() * batch_x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_x.size(0)

    return total_loss / total, correct / total


def train(config):
    """Full training pipeline."""
    print("=" * 60)
    print(f"Training: {config.model_type} on {config.dataset}")
    print("=" * 60)

    set_seed(config.seed)

    # Quantum modeller CPU'da calisir (Qiskit StatevectorEstimator CPU-only)
    if _is_quantum_model(config.model_type):
        device = torch.device("cpu")
        print(f"  Device: CPU (quantum simulasyon CPU gerektirir)")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  Device: {device}")

    # Data
    train_loader, val_loader, test_loader = load_data(config)

    # Model
    model = create_model(config)
    model.to(device)

    # Optimizer & criterion
    optimizer = create_optimizer(model, config)
    criterion = nn.CrossEntropyLoss()

    # Scheduler: val loss iyilesmezse lr dusur
    scheduler = None
    if config.use_scheduler:
        # Quantum modeller icin tum param gruplarina uygula
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=config.scheduler_patience,
            factor=config.scheduler_factor, verbose=False,
        )

    # Training history
    history = {
        "config": {
            "model_type": config.model_type,
            "dataset": config.dataset,
            "n_qubits": config.n_qubits,
            "ansatz_reps": config.ansatz_reps,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "lr_classical": config.lr_classical,
            "lr_quantum": config.lr_quantum,
            "seed": config.seed,
        },
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "epoch_times": [],
    }

    # Checkpoint dir
    ckpt_dir = PROJECT_ROOT / config.checkpoint_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir = PROJECT_ROOT / config.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    patience_counter = 0
    total_train_time = 0.0

    print(f"\n{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>8} | {'Val Acc':>7} | {'Time':>8}")
    print("-" * 65)

    for epoch in range(1, config.epochs + 1):
        t0 = time.time()

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        epoch_time = time.time() - t0
        total_train_time += epoch_time

        # Scheduler step
        if scheduler:
            scheduler.step(val_loss)

        # History kaydet
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["epoch_times"].append(epoch_time)

        # Print
        print(f"{epoch:5d} | {train_loss:10.4f} | {train_acc:8.4f}  | {val_loss:8.4f} | {val_acc:6.4f} | {epoch_time:7.1f}s")

        # Best model checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            ckpt_path = ckpt_dir / f"{config.model_type}_{config.dataset}{config.output_suffix}_best.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
            }, ckpt_path)
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= config.early_stop_patience:
            print(f"\nEarly stopping at epoch {epoch} (patience={config.early_stop_patience})")
            break

    # Test evaluation
    print("\n" + "=" * 60)
    print("Test Evaluation")
    print("=" * 60)

    # En iyi modeli yukle
    ckpt_path = ckpt_dir / f"{config.model_type}_{config.dataset}{config.output_suffix}_best.pt"
    ckpt = torch.load(ckpt_path, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"  Best epoch: {ckpt['epoch']}")
    print(f"  Val accuracy: {ckpt['val_acc']:.4f}")
    print(f"  Test accuracy: {test_acc:.4f}")
    print(f"  Total training time: {total_train_time:.1f}s ({total_train_time/60:.1f} dk)")
    print(f"  Avg epoch time: {total_train_time/len(history['epoch_times']):.1f}s")

    # History'ye test sonuclarini ekle
    history["test_acc"] = test_acc
    history["test_loss"] = test_loss
    history["best_epoch"] = ckpt["epoch"]
    history["best_val_acc"] = best_val_acc
    history["total_train_time"] = total_train_time

    # History JSON kaydet
    log_path = log_dir / f"{config.model_type}_{config.dataset}{config.output_suffix}_history.json"
    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  History: {log_path}")

    return history


def main():
    parser = argparse.ArgumentParser(description="Quantum Attention Training")
    parser.add_argument("--model", type=str, required=True,
                        choices=[
                            "baseline", "attention", "quantum", "quantum_random",
                            # Phase 6 encoding variants
                            "enc_angle", "enc_angle_rand",
                            "enc_dense", "enc_dense_rand",
                            "enc_iqp", "enc_iqp_rand",
                            "enc_reupload", "enc_reupload_rand",
                            # Phase 6 quantum attention
                            "qattn_angle", "qattn_angle_rand",
                            "qattn_iqp", "qattn_iqp_rand",
                            "qattn_reupload", "qattn_reupload_rand",
                        ],
                        help="Model tipi")
    parser.add_argument("--dataset", type=str, default="imdb",
                        choices=["imdb", "sst2"],
                        help="Dataset")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr_quantum", type=float, default=0.01)
    parser.add_argument("--lr_classical", type=float, default=0.001)
    parser.add_argument("--reps", type=int, default=2, help="EfficientSU2 tekrar sayisi")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subset", action="store_true", help="2000 sample subset kullan")
    parser.add_argument("--suffix", type=str, default="", help="Output filename suffix (e.g., _seed123)")
    args = parser.parse_args()

    config = TrainingConfig(
        model_type=args.model,
        dataset=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr_quantum=args.lr_quantum,
        lr_classical=args.lr_classical,
        ansatz_reps=args.reps,
        seed=args.seed,
        use_subset=args.subset,
        output_suffix=args.suffix,
    )

    train(config)


if __name__ == "__main__":
    main()
