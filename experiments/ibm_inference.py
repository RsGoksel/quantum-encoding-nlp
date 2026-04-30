"""
IBM QPU Inference
=================
Egitilmis quantum modelin parametrelerini IBM'in gercek kuantum bilgisayarinda
calistirarak inference yapar.

BU NE YAPAR:
1. PyTorch-native modeleden egitilmis parametreleri (thetas) yukler
2. Ayni circuit yapisin Qiskit EfficientSU2 ile olusturur
3. IBM QPU'da (ibm_torino, 133 qubit) test sample'lari calistirir
4. Sonuclari simulator sonuclariyla karsilastirir

NEDEN ONEMLI:
- "Gercek kuantum donaniminda da calisiyor" kaniti
- Simulator vs QPU noise etkisi karsilastirmasi
- Bildiri icin gercek donanim sonuclari

IBM QPU SURESI:
- 100 sample x ~2-5s/sample = ~200-500s = ~3-8 dakika
- 10 dakikalik free tier'a sigar

Kullanim:
    PYTHONIOENCODING=utf-8 python experiments/ibm_inference.py
"""

import os
import sys
import json
import time
import math
import torch
import numpy as np
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def load_trained_model():
    """Egitilmis quantum modeli ve test verisini yukle."""
    from models.quantum_attention_fast import QuantumAttentionFast

    # Model olustur (ayni mimari)
    model = QuantumAttentionFast(n_qubits=8, reps=1, identity_init=True)

    # En iyi checkpoint'u yukle
    # Random init daha iyi sonuc verdi, onu kullanalim
    ckpt_path = PROJECT_ROOT / "checkpoints" / "quantum_random_imdb_best.pt"
    if not ckpt_path.exists():
        ckpt_path = PROJECT_ROOT / "checkpoints" / "quantum_imdb_best.pt"

    ckpt = torch.load(ckpt_path, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"Model yuklendi: {ckpt_path.name}")
    print(f"  Val accuracy: {ckpt['val_acc']:.4f}")
    print(f"  Epoch: {ckpt['epoch']}")

    # Egitilmis quantum parametreleri
    thetas = model.circuit.thetas.detach().numpy()
    classifier_weight = model.classifier.weight.detach().numpy()
    classifier_bias = model.classifier.bias.detach().numpy()

    print(f"  Quantum params: {len(thetas)}")
    print(f"  Theta range: [{thetas.min():.4f}, {thetas.max():.4f}]")

    return model, thetas, classifier_weight, classifier_bias


def load_test_data(n_samples=100):
    """Test verisinden subset yukle."""
    data = torch.load(PROJECT_ROOT / "data" / "imdb_embeddings.pt", weights_only=False)

    # Ilk n_samples'i al
    test_X = data["test_X"][:n_samples]
    test_y = data["test_y"][:n_samples]

    # Sigmoid scaling (training'deki gibi)
    test_X_scaled = torch.sigmoid(test_X) * math.pi

    print(f"\nTest verisi: {test_X.shape}")
    print(f"  Label dagilimi: 0={int((test_y==0).sum())}, 1={int((test_y==1).sum())}")

    return test_X, test_X_scaled, test_y


def simulator_inference(model, test_X, test_y):
    """Simulator'de inference yap (referans sonuc)."""
    print("\n" + "=" * 60)
    print("SIMULATOR INFERENCE")
    print("=" * 60)

    model.eval()
    t0 = time.time()
    with torch.no_grad():
        logits = model(test_X)
        preds = logits.argmax(dim=1)
    dt = time.time() - t0

    acc = (preds == test_y).float().mean().item()
    print(f"  Accuracy: {acc:.4f} ({int((preds == test_y).sum())}/{len(test_y)})")
    print(f"  Sure: {dt:.3f}s ({dt/len(test_X)*1000:.1f}ms/sample)")

    return preds.numpy(), acc


def ibm_qpu_inference(thetas, classifier_weight, classifier_bias, test_X_scaled, test_y, n_samples=100):
    """
    IBM QPU'da inference.

    Adimlar:
    1. Qiskit circuit olustur (ayni yapi: angle encoding + EfficientSU2 reps=1)
    2. Egitilmis thetas'i circuit'e bind et
    3. IBM QPU'da calistir
    4. Z beklenen degerlerini al
    5. Klasik classifier ile siniflandir
    """
    print("\n" + "=" * 60)
    print("IBM QPU INFERENCE")
    print("=" * 60)

    from qiskit.circuit import QuantumCircuit, ParameterVector
    from qiskit.circuit.library import EfficientSU2
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2

    # 1. IBM'e baglan
    print("\n  IBM'e baglaniliyor...")
    service = QiskitRuntimeService(channel="ibm_quantum_platform")
    backend = service.least_busy(operational=True, min_num_qubits=8)
    print(f"  Backend: {backend.name} ({backend.num_qubits} qubit)")

    n_qubits = 8

    # 2. Circuit olustur
    input_params = ParameterVector("x", n_qubits)
    ansatz = EfficientSU2(num_qubits=n_qubits, reps=1, entanglement="linear")

    qc = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        qc.ry(input_params[i], i)
    qc.compose(ansatz, inplace=True)

    # 3. Observables
    observables = []
    for i in range(n_qubits):
        pauli_str = "I" * (n_qubits - 1 - i) + "Z" + "I" * i
        observables.append(SparsePauliOp.from_list([(pauli_str, 1.0)]))

    # 4. Transpile
    print("  Circuit transpile ediliyor...")
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    transpiled_circuit = pm.run(qc)
    transpiled_obs = [ob.apply_layout(transpiled_circuit.layout) for ob in observables]
    print(f"  Transpiled depth: {transpiled_circuit.depth()}")

    # 5. Parametreleri hazirla
    # Ansatz parametrelerini egitilmis degerlerle doldur
    ansatz_params = ansatz.parameters

    # Her test sample icin parameter binding
    print(f"\n  {n_samples} sample IBM QPU'da calistirilacak...")
    print(f"  Tahmini QPU suresi: {n_samples * 3:.0f}s (~{n_samples * 3 / 60:.1f} dk)")

    test_data_np = test_X_scaled[:n_samples].numpy()

    # 6. QPU'da calistir — KUCUK BATCH'LER ile (Error 6073 onlemi)
    # Her batch'te 5 sample, her sample icin 8 observable = 40 PUB/batch
    print("  QPU job'lari gonderiliyor (kucuk batch'ler)...")
    t0 = time.time()

    estimator = EstimatorV2(mode=backend)
    batch_size = 5  # 5 sample * 8 obs = 40 PUB per job (guvenli limit)

    all_z = np.zeros((n_samples, n_qubits))

    for batch_start in range(0, n_samples, batch_size):
        batch_end = min(batch_start + batch_size, n_samples)
        batch_n = batch_end - batch_start

        pubs = []
        for i in range(batch_start, batch_end):
            param_values = {}
            for j, p in enumerate(input_params):
                param_values[p] = float(test_data_np[i, j])
            for j, p in enumerate(ansatz_params):
                param_values[p] = float(thetas[j])

            bound = transpiled_circuit.assign_parameters(param_values)
            for obs in transpiled_obs:
                pubs.append((bound, obs))

        job = estimator.run(pubs)
        print(f"  Batch {batch_start//batch_size + 1}/{(n_samples + batch_size - 1)//batch_size}: "
              f"Job {job.job_id()}, {len(pubs)} PUBs...")

        result = job.result()

        for i in range(batch_n):
            for q in range(n_qubits):
                idx = i * n_qubits + q
                all_z[batch_start + i, q] = result[idx].data.evs

    dt = time.time() - t0
    print(f"  QPU toplam sure: {dt:.1f}s ({dt/60:.1f} dk)")

    # 7. Klasik classifier uygula
    z_expectations = all_z
    logits = z_expectations @ classifier_weight.T + classifier_bias
    preds = np.argmax(logits, axis=1)
    test_y_np = test_y[:n_samples].numpy()

    acc = (preds == test_y_np).mean()
    print(f"\n  QPU Accuracy: {acc:.4f} ({int((preds == test_y_np).sum())}/{n_samples})")
    print(f"  QPU Sure: {dt:.1f}s ({dt/n_samples:.1f}s/sample)")

    return preds, acc, dt


def main():
    print("=" * 60)
    print("IBM QPU Inference — Quantum Attention Model")
    print("=" * 60)

    # Model ve veri yukle
    model, thetas, classifier_weight, classifier_bias = load_trained_model()
    n_samples = 10  # Kucuk tut — QPU suresi kiymetli
    test_X, test_X_scaled, test_y = load_test_data(n_samples=n_samples)

    # Simulator inference (referans)
    sim_preds, sim_acc = simulator_inference(model, test_X, test_y)

    # IBM QPU inference
    try:
        qpu_preds, qpu_acc, qpu_time = ibm_qpu_inference(
            thetas, classifier_weight, classifier_bias,
            test_X_scaled, test_y, n_samples=n_samples
        )

        # Karsilastirma
        print("\n" + "=" * 60)
        print("KARSILASTIRMA: Simulator vs IBM QPU")
        print("=" * 60)
        agreement = (sim_preds[:n_samples] == qpu_preds).mean()
        print(f"  Simulator accuracy: {sim_acc:.4f}")
        print(f"  QPU accuracy:       {qpu_acc:.4f}")
        print(f"  Fark:               {abs(sim_acc - qpu_acc):.4f}")
        print(f"  Tahmin uyumu:       {agreement:.4f} ({int(agreement*n_samples)}/{n_samples} ayni)")
        print(f"  QPU toplam sure:    {qpu_time:.1f}s")

        # Sonuclari kaydet
        results = {
            "simulator_accuracy": sim_acc,
            "qpu_accuracy": qpu_acc,
            "difference": abs(sim_acc - qpu_acc),
            "agreement": float(agreement),
            "qpu_time_seconds": qpu_time,
            "n_samples": 50,
            "backend": "ibm_torino",
        }
        results_path = PROJECT_ROOT / "results" / "logs" / "ibm_qpu_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Sonuclar kaydedildi: {results_path}")

    except Exception as e:
        print(f"\n  IBM QPU HATASI: {e}")
        print("  Simulator sonuclari gecerli, QPU sonuclari alinamadi.")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
