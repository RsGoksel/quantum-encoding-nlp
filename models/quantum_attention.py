"""
Quantum Attention Model
=======================
PCA embedding -> Quantum Circuit (EfficientSU2) -> Linear -> Softmax

Bu projenin ANA modeli. Klasik attention yerine kuantum devre kullanir.

Kuantum devrenin yapisi:
1. ENCODING: 8 boyutlu PCA vektoru -> 8 qubit'e angle encoding (Ry gate'ler)
2. ANSATZ: EfficientSU2 (Ry+Rz rotasyonlari + CZ entanglement), identity-initialized
3. MEASUREMENT: Her qubit'in Pauli-Z beklenen degeri -> 8 boyutlu output
4. CLASSIFIER: Linear(8, 2) ile siniflandirma

Identity Initialization (KRITIK):
- Tum trainable rotation acilari = 0 olarak baslatilir
- Ry(0) = I, Rz(0) = I -> rotation'lar identity
- CZ gate'ler parametresiz, her zaman ayni (entanglement yapisi)
- Sonuc: circuit baslangicta "neredeyse identity" gibi davranir
- Gradyanlar sifirdan degil, anlamli bir noktadan baslar
- Bu, barren plateau problemini hafifletir (onceki projelerden ogrenilen ders)

Random Init (kontrol grubu):
- Ayni circuit ama rotation acilari U(0, 2*pi) uniform random
- Beklenti: barren plateau'ya takilir, convergence kotu olur
"""

import torch
import torch.nn as nn
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import EfficientSU2
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def build_quantum_circuit(n_qubits=8, reps=2):
    """
    Quantum attention circuit'ini olustur.

    Args:
        n_qubits: Qubit sayisi (= PCA boyutu)
        reps: EfficientSU2 tekrar katmani sayisi

    Returns:
        qc: Full quantum circuit (encoding + ansatz)
        feature_params: Encoding parametreleri (input)
        ansatz_params: Trainable parametreler (weight)

    Circuit yapisi (8 qubit, 2 reps ornegi):
    |0> — Ry(x0) — Ry(t0) — Rz(t8)  — CZ — Ry(t16) — Rz(t24) — CZ —— Measure Z
    |0> — Ry(x1) — Ry(t1) — Rz(t9)  — CZ — Ry(t17) — Rz(t25) — CZ —— Measure Z
    ...
    |0> — Ry(x7) — Ry(t7) — Rz(t15) — CZ — Ry(t23) — Rz(t31) — CZ —— Measure Z

    x0..x7 = input (PCA features), t0..t47 = trainable
    Toplam trainable: n_qubits * 2 * (reps + 1) = 8 * 2 * 3 = 48
    """
    from qiskit.circuit import ParameterVector

    # Input encoding parametreleri
    input_params = ParameterVector("x", n_qubits)

    # Ansatz (trainable)
    ansatz = EfficientSU2(
        num_qubits=n_qubits,
        reps=reps,
        entanglement="linear",  # Komsu qubit'ler arasi CZ
    )

    # Full circuit: encoding + ansatz
    qc = QuantumCircuit(n_qubits)

    # Angle encoding: Ry(x_i) ile veri yukleme
    for i in range(n_qubits):
        qc.ry(input_params[i], i)

    # Ansatz ekleme
    qc.compose(ansatz, inplace=True)

    return qc, list(input_params), list(ansatz.parameters)


def build_qnn(n_qubits=8, reps=2):
    """
    EstimatorQNN olustur — Qiskit'in quantum neural network sinifi.

    EstimatorQNN ne yapar:
    - Circuit'i input ve weight parametreleriyle calistirir
    - Her qubit icin Pauli-Z beklenen degerini olcer: <Z_i>
    - <Z_i> degeri -1 ile +1 arasinda (|0> icin +1, |1> icin -1)
    - Sonuc: 8 boyutlu vektor (her qubit bir output)

    Gradient nasil hesaplanir:
    - Parameter-shift rule: f'(t) = [f(t + pi/2) - f(t - pi/2)] / 2
    - Her parametre icin 2 ek circuit evaluation gerekir
    - 48 parametre -> 96 ek evaluation / gradient step
    - Bu yuzden quantum training yavas!
    """
    qc, input_params, weight_params = build_quantum_circuit(n_qubits, reps)

    # Observable: her qubit icin Pauli-Z
    observables = []
    for i in range(n_qubits):
        # "IIIZIII" gibi — sadece i. qubit'te Z, digerlerinde I
        pauli_str = "I" * (n_qubits - 1 - i) + "Z" + "I" * i
        observables.append(SparsePauliOp.from_list([(pauli_str, 1.0)]))

    qnn = EstimatorQNN(
        circuit=qc,
        input_params=input_params,
        weight_params=weight_params,
        observables=observables,
        estimator=StatevectorEstimator(),
    )

    return qnn


class QuantumAttentionModel(nn.Module):
    """
    Hybrid quantum-classical model.

    Yapi:
    1. Quantum layer (TorchConnector): PCA(8) -> QNN -> 8-dim output
    2. Linear classifier: 8 -> 2 (positive/negative)

    Parametre dagilimlari:
    - Quantum: 48 (EfficientSU2 rotation acilari)
    - Classical: 18 (Linear 8->2 + bias)
    - Toplam: 66
    """

    def __init__(self, n_qubits=8, reps=2, identity_init=True):
        super().__init__()
        self.n_qubits = n_qubits
        self.identity_init = identity_init

        # QNN olustur
        qnn = build_qnn(n_qubits, reps)

        # Identity init: tum acilari 0'a set et
        # Random init: uniform random [0, 2*pi)
        if identity_init:
            initial_weights = torch.zeros(qnn.num_weights)
        else:
            initial_weights = torch.rand(qnn.num_weights) * 2 * np.pi

        # TorchConnector: Qiskit QNN'i PyTorch Module'e sarar
        # Bu sayede standard PyTorch training loop'u kullanabiliriz
        self.quantum_layer = TorchConnector(qnn, initial_weights=initial_weights)

        # Classical classifier head
        self.classifier = nn.Linear(n_qubits, 2)

    def forward(self, x):
        # x: (batch, 8) — PCA embeddings
        # Quantum layer: (batch, 8) -> (batch, 8)
        q_out = self.quantum_layer(x)
        # Classifier: (batch, 8) -> (batch, 2)
        return self.classifier(q_out)

    def count_params(self):
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        quantum = sum(p.numel() for p in self.quantum_layer.parameters() if p.requires_grad)
        classical = total - quantum
        return {"total": total, "quantum": quantum, "classical": classical}
