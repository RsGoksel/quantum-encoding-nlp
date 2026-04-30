"""
Fast Quantum Attention Model — PyTorch-Native
==============================================
Qiskit EstimatorQNN yerine PyTorch-native statevector simulasyonu.

NEDEN BU DOSYA VAR?
- Qiskit EstimatorQNN parameter-shift rule kullaniyor
- Parameter-shift: her gradient icin 2*N_params ek circuit evaluation
- 48 param -> 96 ek eval -> backward 83x yavasliyor (1.5s/sample!)
- PyTorch autograd: TEK backward pass'te tum gradyanlar -> ~3ms/sample

NASIL CALISIYOR?
- Quantum state = PyTorch complex tensor (2^n boyutlu)
- Gate'ler = matrix carpimi (torch.matmul)
- Rotation gate'ler differentiable (cos/sin torch fonksiyonlari)
- PyTorch autograd chain rule ile gradient hesaplar — parameter-shift'e gerek yok

BU YAKLASIM ONCEKI PROJELERDE KANITLANDI:
- qaoa-maxcut: PyTorch-native QAOA, unitarity PASS
- quantum-state-prep: PyTorch-native VQC, calistiSecond
- 2048-quantum: PyTorch-native VQC, 12 versiyon

IBM QPU ILE UYUMLULUK:
- Training: bu dosyayla (PyTorch-native, hizli)
- IBM QPU inference: ayri Qiskit script ile (sadece test zamani)
- Parametreler transfer edilebilir (ayni circuit yapisi)
"""

import torch
import torch.nn as nn
import numpy as np
import math


class QuantumCircuitLayer(nn.Module):
    """
    PyTorch-native quantum circuit simulasyonu.

    Circuit yapisi (EfficientSU2 ile ayni):
    1. Angle Encoding: Ry(x_i) ile input yukleme
    2. Variational: [Ry(t) - Rz(t) - CZ entanglement] x reps
    3. Measurement: Her qubit icin <Z_i> beklenen degeri

    Tum islemler PyTorch tensor'leri ile — autograd otomatik calisir.
    """

    def __init__(self, n_qubits=8, reps=2, identity_init=True):
        super().__init__()
        self.n_qubits = n_qubits
        self.reps = reps
        self.dim = 2 ** n_qubits  # Hilbert uzay boyutu

        # Trainable parametreler: her rep icin n_qubits Ry + n_qubits Rz
        # + son katmanda n_qubits Ry + n_qubits Rz (EfficientSU2 convention)
        n_params = n_qubits * 2 * (reps + 1)
        if identity_init:
            # Identity init: tum acilar 0 -> Ry(0)=I, Rz(0)=I
            self.thetas = nn.Parameter(torch.zeros(n_params))
        else:
            # Random init: uniform [0, 2*pi)
            self.thetas = nn.Parameter(torch.rand(n_params) * 2 * math.pi)

        # CZ gate matrisini onceden hesapla (parametresiz, sabit)
        # CZ: |00>->|00>, |01>->|01>, |10>->|10>, |11>->-|11>
        self._precompute_cz_gates()

    def _precompute_cz_gates(self):
        """CZ gate'lerini precompute et (linear entanglement)."""
        # Her komsuluk cifti icin CZ gate
        cz_2q = torch.tensor([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, -1],
        ], dtype=torch.complex128)

        self.register_buffer("cz_2q", cz_2q)

    def _ry_gate(self, theta):
        """Ry(theta) = [[cos(t/2), -sin(t/2)], [sin(t/2), cos(t/2)]]"""
        c = torch.cos(theta / 2)
        s = torch.sin(theta / 2)
        return torch.stack([
            torch.stack([c, -s]),
            torch.stack([s, c]),
        ]).to(torch.complex128)

    def _rz_gate(self, theta):
        """Rz(theta) = [[e^(-it/2), 0], [0, e^(it/2)]]"""
        phase = theta / 2
        return torch.stack([
            torch.stack([torch.exp(-1j * phase), torch.tensor(0.0, dtype=torch.complex128)]),
            torch.stack([torch.tensor(0.0, dtype=torch.complex128), torch.exp(1j * phase)]),
        ])

    def _apply_single_qubit_gate(self, state, gate, qubit):
        """
        Tek qubit'e gate uygula.

        Statevector'u (2^n,) -> (2, 2, ..., 2) reshape edip
        ilgili qubit ekseninde matrix carpimi yapiyoruz.

        Bu, kronecker product'tan cok daha hizli!
        """
        n = self.n_qubits
        # State'i n-qubit tensor'e reshape et
        shape = [2] * n
        state = state.reshape(shape)

        # Gate'i ilgili qubit ekseninde uygula
        # torch.einsum ile: gate[i,j] * state[..., j, ...] -> state[..., i, ...]
        # Qubit indeksi = qubit (0'dan baslayarak)
        state = torch.tensordot(gate, state, dims=([1], [qubit]))

        # tensordot sonucu: gate ekseni en basa geliyor
        # Orijinal siraya geri getir
        perm = list(range(1, qubit + 1)) + [0] + list(range(qubit + 1, n))
        state = state.permute(perm)

        return state.reshape(-1)

    def _apply_cz(self, state, q1, q2):
        """
        CZ gate uygula (iki qubit).

        CZ sadece |11> durumuna -1 faz ekler.
        Bu, kolay optimize edilebilir: |11> bilesenlerini bul ve isaret degistir.
        """
        n = self.n_qubits
        shape = [2] * n
        state = state.reshape(shape)

        # |11> durumlarinin isareti degisir
        # q1 ve q2 qubit'lerinin ikisi de 1 olan tum durumlar
        idx = [slice(None)] * n
        idx[q1] = 1
        idx[q2] = 1
        state = state.clone()
        state[tuple(idx)] = -state[tuple(idx)]

        return state.reshape(-1)

    def forward(self, x):
        """
        Batch forward pass.

        Args:
            x: (batch, n_qubits) — PCA features

        Returns:
            (batch, n_qubits) — Pauli-Z beklenen degerleri [-1, 1]

        Islem sirasi:
        1. Input'u [0, pi]'ye scale et (angle encoding icin optimal aralik)
        2. |000...0> baslangic durumu
        3. Angle encoding: Ry(x_i) her qubit'e
        4. Variational layers: [Ry(t) - Rz(t) - CZ_linear] x reps
        5. Son Ry-Rz katmani
        6. Her qubit icin <Z> olcumu
        """
        batch_size = x.shape[0]
        device = x.device

        # Input'u [0, pi]'ye scale et
        # PCA output'lari ~N(0,1). sigmoid: (-inf,inf)->[0,1], sonra *pi: [0,pi]
        # Neden: Ry(0)=|0>, Ry(pi)=|1>. [0,pi] arasi encoding full Bloch sphere kaplar
        x = torch.sigmoid(x) * math.pi

        # Parametre indeksi takibi
        param_idx = 0

        results = []
        for b in range(batch_size):
            # |000...0> baslangic durumu
            state = torch.zeros(self.dim, dtype=torch.complex128, device=device)
            state[0] = 1.0

            # 1. Angle encoding: Ry(x_i) ile input yukleme
            for q in range(self.n_qubits):
                gate = self._ry_gate(x[b, q].to(torch.complex128))
                state = self._apply_single_qubit_gate(state, gate, q)

            # 2. Variational layers
            param_idx = 0
            for rep in range(self.reps + 1):
                # Ry rotasyonlari
                for q in range(self.n_qubits):
                    gate = self._ry_gate(self.thetas[param_idx].to(torch.complex128))
                    state = self._apply_single_qubit_gate(state, gate, q)
                    param_idx += 1

                # Rz rotasyonlari
                for q in range(self.n_qubits):
                    gate = self._rz_gate(self.thetas[param_idx].to(torch.complex128))
                    state = self._apply_single_qubit_gate(state, gate, q)
                    param_idx += 1

                # CZ entanglement (son katmandan sonra degil)
                if rep < self.reps:
                    for q in range(self.n_qubits - 1):
                        state = self._apply_cz(state, q, q + 1)

            # 3. Pauli-Z beklenen degerleri
            # <Z_q> = sum_i |alpha_i|^2 * (-1)^bit_q(i)
            probs = (state * state.conj()).real  # |alpha_i|^2
            z_expectations = []
            for q in range(self.n_qubits):
                # Qubit q'nun 0 oldugu durumlar: +1, 1 oldugu durumlar: -1
                signs = torch.ones(self.dim, device=device)
                for i in range(self.dim):
                    if (i >> (self.n_qubits - 1 - q)) & 1:
                        signs[i] = -1.0
                z_q = (probs * signs).sum()
                z_expectations.append(z_q)

            results.append(torch.stack(z_expectations))

        return torch.stack(results).float()  # (batch, n_qubits)


class QuantumAttentionFast(nn.Module):
    """
    Fast hybrid quantum-classical model.

    PyTorch-native simulasyon ile EfficientSU2-equivalent circuit.
    Training Qiskit'ten ~500x hizli (autograd vs parameter-shift).

    IBM QPU'ya transfer icin: egitilmis self.circuit.thetas parametreleri
    ayni EfficientSU2 circuit'ine aktarilabilir.
    """

    def __init__(self, n_qubits=8, reps=2, identity_init=True):
        super().__init__()
        self.circuit = QuantumCircuitLayer(n_qubits, reps, identity_init)
        self.classifier = nn.Linear(n_qubits, 2)

    def forward(self, x):
        q_out = self.circuit(x)
        return self.classifier(q_out)

    def count_params(self):
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        quantum = self.circuit.thetas.numel()
        classical = total - quantum
        return {"total": total, "quantum": quantum, "classical": classical}
