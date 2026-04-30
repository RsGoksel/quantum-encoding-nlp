"""
Quantum Encoding Strategies — Phase 6 Literature Contribution
==============================================================
NLP embedding'leri icin 4 farkli kuantum encoding stratejisinin
sistematik karsilastirmasi.

ENCODING STRATEJILERI:
1. Angle Encoding: Ry(sigmoid(x_i)*pi) — mevcut, basit
2. Dense Angle Encoding: Ry(x_i) + Rz(x_j) ayni qubit — 2x feature/qubit
3. IQP Encoding: Rz(x_i) + RZZ(x_i*x_j) — feature interaction
4. Data Re-uploading: Her var layer oncesi Ry(x_i) tekrar — yuksek expressibility

HER ENCODING ICIN:
- Ayni variational ansatz (EfficientSU2: Ry+Rz+CZ)
- Ayni measurement (Pauli-Z)
- Ayni sigmoid scaling (encoding-preprocessing uyumu)
- Identity ve random init destegi

QUANTUM ATTENTION MEKANIZMASI:
- Mevcut: circuit output → Linear → prediction (basit classification)
- Yeni: circuit output → softmax → attention weights → modulate features → Linear → prediction
- Bu, Proposal 2'deki "interference + softmax" mekanizmasinin implementasyonu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ===========================================================================
# Temel Quantum Circuit Layer (encoding-agnostic variational + measurement)
# ===========================================================================

class QuantumCircuitBase(nn.Module):
    """
    Encoding-agnostic quantum circuit base class.

    Alt siniflar sadece _encode() metodunu override eder.
    Variational katmanlar ve measurement tum encoding'ler icin ayni.
    """

    def __init__(self, n_qubits=8, reps=1, identity_init=True):
        super().__init__()
        self.n_qubits = n_qubits
        self.reps = reps
        self.dim = 2 ** n_qubits

        # Variational parametreler: her rep icin n_qubits Ry + n_qubits Rz + final layer
        n_params = n_qubits * 2 * (reps + 1)
        if identity_init:
            self.thetas = nn.Parameter(torch.zeros(n_params))
        else:
            self.thetas = nn.Parameter(torch.rand(n_params) * 2 * math.pi)

        # Z measurement icin isaret vektoru (precompute)
        self._precompute_signs()

    def _precompute_signs(self):
        """Pauli-Z measurement isaret vektorlerini onceden hesapla."""
        signs = torch.zeros(self.n_qubits, self.dim)
        for q in range(self.n_qubits):
            for i in range(self.dim):
                if (i >> (self.n_qubits - 1 - q)) & 1:
                    signs[q, i] = -1.0
                else:
                    signs[q, i] = 1.0
        self.register_buffer("z_signs", signs)

    def _ry_gate(self, theta):
        """Ry(theta) rotation gate."""
        c = torch.cos(theta / 2)
        s = torch.sin(theta / 2)
        return torch.stack([
            torch.stack([c, -s]),
            torch.stack([s, c]),
        ]).to(torch.complex128)

    def _rz_gate(self, theta):
        """Rz(theta) rotation gate."""
        phase = theta / 2
        zero = torch.zeros_like(phase, dtype=torch.complex128)
        return torch.stack([
            torch.stack([torch.exp(-1j * phase), zero]),
            torch.stack([zero, torch.exp(1j * phase)]),
        ])

    def _apply_single_qubit_gate(self, state, gate, qubit):
        """Tek qubit'e gate uygula (tensordot yontemi)."""
        n = self.n_qubits
        shape = [2] * n
        state = state.reshape(shape)
        state = torch.tensordot(gate, state, dims=([1], [qubit]))
        perm = list(range(1, qubit + 1)) + [0] + list(range(qubit + 1, n))
        state = state.permute(perm)
        return state.reshape(-1)

    def _apply_cz(self, state, q1, q2):
        """CZ gate: |11> bilesenlerine -1 faz."""
        n = self.n_qubits
        shape = [2] * n
        state = state.reshape(shape)
        idx = [slice(None)] * n
        idx[q1] = 1
        idx[q2] = 1
        state = state.clone()
        state[tuple(idx)] = -state[tuple(idx)]
        return state.reshape(-1)

    def _apply_rzz(self, state, q1, q2, theta):
        """
        RZZ(theta) gate: exp(-i * theta/2 * Z_q1 Z_q2)
        |00> → e^(-i*theta/2) |00>
        |01> → e^(+i*theta/2) |01>
        |10> → e^(+i*theta/2) |10>
        |11> → e^(-i*theta/2) |11>
        """
        n = self.n_qubits
        shape = [2] * n
        state = state.reshape(shape)

        phase = theta / 2
        # |00> ve |11>: e^(-i*phase), |01> ve |10>: e^(+i*phase)
        state = state.clone()

        # |00> durumlar
        idx_00 = [slice(None)] * n
        idx_00[q1] = 0
        idx_00[q2] = 0
        state[tuple(idx_00)] = state[tuple(idx_00)] * torch.exp(-1j * phase)

        # |01> durumlar
        idx_01 = [slice(None)] * n
        idx_01[q1] = 0
        idx_01[q2] = 1
        state[tuple(idx_01)] = state[tuple(idx_01)] * torch.exp(1j * phase)

        # |10> durumlar
        idx_10 = [slice(None)] * n
        idx_10[q1] = 1
        idx_10[q2] = 0
        state[tuple(idx_10)] = state[tuple(idx_10)] * torch.exp(1j * phase)

        # |11> durumlar
        idx_11 = [slice(None)] * n
        idx_11[q1] = 1
        idx_11[q2] = 1
        state[tuple(idx_11)] = state[tuple(idx_11)] * torch.exp(-1j * phase)

        return state.reshape(-1)

    def _encode(self, state, x_sample):
        """
        Encoding adimi — ALT SINIFLAR OVERRIDE EDER.

        Args:
            state: (dim,) complex128 — |0...0> baslangic durumu
            x_sample: (input_dim,) — tek sample'in PCA features'lari

        Returns:
            state: (dim,) — encoding uygulanmis durum
        """
        raise NotImplementedError

    def _variational_and_measure(self, state, device):
        """Variational katmanlar + Pauli-Z measurement (tum encoding'ler icin ayni)."""
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

        # Pauli-Z measurement
        probs = (state * state.conj()).real
        z_exp = (probs.unsqueeze(0) * self.z_signs).sum(dim=1)
        return z_exp

    def forward(self, x):
        """
        Args:
            x: (batch, input_dim) — PCA features
        Returns:
            (batch, n_qubits) — Pauli-Z beklenen degerleri [-1, 1]
        """
        batch_size = x.shape[0]
        device = x.device

        results = []
        for b in range(batch_size):
            # |0...0> baslangic durumu
            state = torch.zeros(self.dim, dtype=torch.complex128, device=device)
            state[0] = 1.0

            # Encoding
            state = self._encode(state, x[b])

            # Variational + Measurement
            z_exp = self._variational_and_measure(state, device)
            results.append(z_exp)

        return torch.stack(results).float()


# ===========================================================================
# Encoding 1: Angle Encoding (mevcut yaklasim)
# ===========================================================================

class AngleEncodingCircuit(QuantumCircuitBase):
    """
    Angle Encoding: Her qubit'e Ry(sigmoid(x_i)*pi) ile input yukleme.

    - 8 feature → 8 qubit (1:1 mapping)
    - Basit, lineer encoding
    - Sigmoid scaling ile [0, pi] araligina normalize
    - Phase 4'te test edildi: %69 accuracy
    """

    def _encode(self, state, x_sample):
        x_scaled = torch.sigmoid(x_sample) * math.pi
        for q in range(self.n_qubits):
            gate = self._ry_gate(x_scaled[q].to(torch.complex128))
            state = self._apply_single_qubit_gate(state, gate, q)
        return state


# ===========================================================================
# Encoding 2: Dense Angle Encoding
# ===========================================================================

class DenseAngleEncodingCircuit(QuantumCircuitBase):
    """
    Dense Angle Encoding: Ry(x_i) + Rz(x_j) ayni qubit'e.

    - 8 feature → 4 qubit (2 feature/qubit)
    - Ry ve Rz farkli eksenlerde: ortogonal bilgi tasiyabilir
    - Daha az qubit → daha kucuk Hilbert uzayi (2^4=16 vs 2^8=256)
    - Avantaj: Daha az qubit = daha az noise (QPU'da)
    - Dezavantaj: Daha az expressibility

    Qubit mapping:
    - Qubit 0: Ry(x_0), Rz(x_1)
    - Qubit 1: Ry(x_2), Rz(x_3)
    - Qubit 2: Ry(x_4), Rz(x_5)
    - Qubit 3: Ry(x_6), Rz(x_7)
    """

    def __init__(self, n_features=8, reps=1, identity_init=True):
        # 8 feature / 2 = 4 qubit
        n_qubits = n_features // 2
        super().__init__(n_qubits=n_qubits, reps=reps, identity_init=identity_init)
        self.n_features = n_features

    def _encode(self, state, x_sample):
        x_scaled = torch.sigmoid(x_sample) * math.pi
        for q in range(self.n_qubits):
            # Ry(x_{2q}) — first feature
            ry_gate = self._ry_gate(x_scaled[2 * q].to(torch.complex128))
            state = self._apply_single_qubit_gate(state, ry_gate, q)
            # Rz(x_{2q+1}) — second feature
            rz_gate = self._rz_gate(x_scaled[2 * q + 1].to(torch.complex128))
            state = self._apply_single_qubit_gate(state, rz_gate, q)
        return state


# ===========================================================================
# Encoding 3: IQP (Instantaneous Quantum Polynomial) Encoding
# ===========================================================================

class IQPEncodingCircuit(QuantumCircuitBase):
    """
    IQP Encoding: Rz(x_i) individual + RZZ(x_i * x_j) pairwise interaction.

    - 8 feature → 8 qubit
    - Hadamard → Rz(x_i) → RZZ(x_i*x_j) nearest-neighbor
    - Feature interaction'lari yakalayabilir (non-linear)
    - ZZFeatureMap'in PyTorch-native karsiligi

    Circuit:
    H|0> → Rz(x_0) → RZZ(x_0*x_1) → ... → variational layers
    H|0> → Rz(x_1) → RZZ(x_1*x_2) → ...
    ...

    Neden IQP?
    - Havlicek et al. (2019): IQP circuit'ler klasik olarak simule edilmesi zor
    - Feature interaction terimleri (x_i * x_j) non-linear kernel olusturur
    - Literaturde quantum kernel methods ile basarili sonuclar
    """

    def _encode(self, state, x_sample):
        x_scaled = torch.sigmoid(x_sample) * math.pi

        # Hadamard tum qubit'lere (|0> → |+>)
        h_gate = torch.tensor([
            [1, 1],
            [1, -1],
        ], dtype=torch.complex128) / math.sqrt(2)
        for q in range(self.n_qubits):
            state = self._apply_single_qubit_gate(state, h_gate, q)

        # Rz(x_i) her qubit'e
        for q in range(self.n_qubits):
            gate = self._rz_gate(x_scaled[q].to(torch.complex128))
            state = self._apply_single_qubit_gate(state, gate, q)

        # RZZ(x_i * x_j) nearest-neighbor interaction
        for q in range(self.n_qubits - 1):
            interaction = x_scaled[q] * x_scaled[q + 1]
            state = self._apply_rzz(state, q, q + 1, interaction.to(torch.complex128))

        return state


# ===========================================================================
# Encoding 4: Data Re-uploading
# ===========================================================================

class DataReuploadingCircuit(QuantumCircuitBase):
    """
    Data Re-uploading: Input her variational layer oncesi tekrar yuklenir.

    - 8 feature → 8 qubit
    - Her variational layer oncesi Ry(x_i) tekrar encoding
    - Perez-Salinas et al. (2020): "Data re-uploading for a universal quantum classifier"
    - Tek katmanli quantum circuit universal approximator DEGIL
    - Ama data re-uploading ile universal approximator haline gelir

    Circuit:
    |0> → Ry(x_0) → [Ry(t) Rz(t) CZ] → Ry(x_0) → [Ry(t) Rz(t) CZ] → ... → Measure
    |0> → Ry(x_1) → [Ry(t) Rz(t) CZ] → Ry(x_1) → [Ry(t) Rz(t) CZ] → ... → Measure
    ...

    Avantaj: Daha yuksek expressibility
    Dezavantaj: Daha derin circuit → daha fazla noise (QPU'da)
    """

    def _encode(self, state, x_sample):
        # Sadece ilk encoding — re-uploading forward'da yapilir
        x_scaled = torch.sigmoid(x_sample) * math.pi
        for q in range(self.n_qubits):
            gate = self._ry_gate(x_scaled[q].to(torch.complex128))
            state = self._apply_single_qubit_gate(state, gate, q)
        return state

    def forward(self, x):
        """Data re-uploading: her variational layer oncesi input tekrar yuklenir."""
        batch_size = x.shape[0]
        device = x.device

        x_scaled = torch.sigmoid(x) * math.pi

        results = []
        for b in range(batch_size):
            state = torch.zeros(self.dim, dtype=torch.complex128, device=device)
            state[0] = 1.0

            # Ilk encoding
            for q in range(self.n_qubits):
                gate = self._ry_gate(x_scaled[b, q].to(torch.complex128))
                state = self._apply_single_qubit_gate(state, gate, q)

            param_idx = 0
            for rep in range(self.reps + 1):
                # Variational: Ry(theta)
                for q in range(self.n_qubits):
                    gate = self._ry_gate(self.thetas[param_idx].to(torch.complex128))
                    state = self._apply_single_qubit_gate(state, gate, q)
                    param_idx += 1
                # Variational: Rz(theta)
                for q in range(self.n_qubits):
                    gate = self._rz_gate(self.thetas[param_idx].to(torch.complex128))
                    state = self._apply_single_qubit_gate(state, gate, q)
                    param_idx += 1
                # CZ entanglement
                if rep < self.reps:
                    for q in range(self.n_qubits - 1):
                        state = self._apply_cz(state, q, q + 1)

                    # DATA RE-UPLOADING: input tekrar yukle (son rep haric)
                    for q in range(self.n_qubits):
                        gate = self._ry_gate(x_scaled[b, q].to(torch.complex128))
                        state = self._apply_single_qubit_gate(state, gate, q)

            # Pauli-Z measurement
            probs = (state * state.conj()).real
            z_exp = (probs.unsqueeze(0) * self.z_signs).sum(dim=1)
            results.append(z_exp)

        return torch.stack(results).float()


# ===========================================================================
# Quantum Attention Models (encoding + classifier)
# ===========================================================================

class QuantumEncodingModel(nn.Module):
    """
    Quantum model with configurable encoding strategy.

    Basit versiyon: circuit output → Linear(n_qubits, 2) → classification
    Phase 4'teki QuantumAttentionFast ile ayni yapi, farkli encoding.
    """

    def __init__(self, encoding="angle", n_qubits=8, reps=1, identity_init=True):
        super().__init__()
        self.encoding_name = encoding

        if encoding == "angle":
            self.circuit = AngleEncodingCircuit(n_qubits=n_qubits, reps=reps, identity_init=identity_init)
            out_dim = n_qubits
        elif encoding == "dense_angle":
            self.circuit = DenseAngleEncodingCircuit(n_features=n_qubits, reps=reps, identity_init=identity_init)
            out_dim = n_qubits // 2  # 4 qubit output
        elif encoding == "iqp":
            self.circuit = IQPEncodingCircuit(n_qubits=n_qubits, reps=reps, identity_init=identity_init)
            out_dim = n_qubits
        elif encoding == "re-uploading":
            self.circuit = DataReuploadingCircuit(n_qubits=n_qubits, reps=reps, identity_init=identity_init)
            out_dim = n_qubits
        else:
            raise ValueError(f"Unknown encoding: {encoding}")

        self.classifier = nn.Linear(out_dim, 2)

    def forward(self, x):
        q_out = self.circuit(x)
        return self.classifier(q_out)

    def count_params(self):
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        quantum = self.circuit.thetas.numel()
        classical = total - quantum
        return {"total": total, "quantum": quantum, "classical": classical}


class QuantumAttentionModel(nn.Module):
    """
    TRUE Quantum Attention — Proposal 2'deki "interference + softmax" mekanizmasi.

    Mevcut basit model:
        PCA features → Quantum Circuit → Linear → prediction

    Bu model:
        PCA features → Quantum Circuit → softmax → attention weights
                    ↘                        ↓
                     → features * attention weights → Linear → prediction

    Nasil calisir:
    1. Quantum circuit 8-dim Pauli-Z beklenen degerler uretir [-1, 1]
    2. Bu degerler (shifted + scaled) softmax'ten gecirilir → attention weights [0, 1]
    3. Attention weights, orijinal PCA features'lari modulate eder
    4. Modulated features → Linear classifier → prediction

    Neden "quantum attention"?
    - Quantum circuit, input'a bagli olarak her feature'a ne kadar "dikkat" verilecegini ogrenir
    - Klasik attention'da bu Q*K^T/sqrt(d) ile yapilir
    - Quantum attention'da bu EfficientSU2 circuit'in parametrik donusumu ile yapilir
    - Potansiyel avantaj: Quantum entanglement, feature'lar arasi non-local korelasyonlari yakalayabilir
    """

    def __init__(self, encoding="angle", n_qubits=8, reps=1, identity_init=True):
        super().__init__()
        self.encoding_name = encoding
        self.n_qubits = n_qubits

        if encoding == "angle":
            self.circuit = AngleEncodingCircuit(n_qubits=n_qubits, reps=reps, identity_init=identity_init)
            self.attn_dim = n_qubits
        elif encoding == "dense_angle":
            self.circuit = DenseAngleEncodingCircuit(n_features=n_qubits, reps=reps, identity_init=identity_init)
            self.attn_dim = n_qubits // 2
        elif encoding == "iqp":
            self.circuit = IQPEncodingCircuit(n_qubits=n_qubits, reps=reps, identity_init=identity_init)
            self.attn_dim = n_qubits
        elif encoding == "re-uploading":
            self.circuit = DataReuploadingCircuit(n_qubits=n_qubits, reps=reps, identity_init=identity_init)
            self.attn_dim = n_qubits
        else:
            raise ValueError(f"Unknown encoding: {encoding}")

        # Attention weight projection (quantum output → feature dimension)
        if self.attn_dim != n_qubits:
            self.attn_proj = nn.Linear(self.attn_dim, n_qubits)
        else:
            self.attn_proj = None

        # Temperature parameter for attention sharpness
        self.temperature = nn.Parameter(torch.tensor(1.0))

        # Classifier on modulated features
        self.classifier = nn.Linear(n_qubits, 2)

    def forward(self, x):
        # x: (batch, 8) — PCA features

        # 1. Quantum circuit → raw attention scores
        q_out = self.circuit(x)  # (batch, attn_dim), values in [-1, 1]

        # 2. Project to feature dimension if needed (dense_angle: 4→8)
        if self.attn_proj is not None:
            q_out = self.attn_proj(q_out)

        # 3. Shift from [-1,1] to [0,2], then softmax with temperature
        attn_scores = (q_out + 1.0) / self.temperature
        attn_weights = F.softmax(attn_scores, dim=-1)  # (batch, 8), sums to 1

        # 4. Modulate original features with attention weights
        # Element-wise multiplication: weighted features
        modulated = x * attn_weights * self.n_qubits  # scale factor to maintain magnitude

        # 5. Classify
        return self.classifier(modulated)

    def count_params(self):
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        quantum = self.circuit.thetas.numel()
        classical = total - quantum
        return {"total": total, "quantum": quantum, "classical": classical}
