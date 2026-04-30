"""
Classical Attention Model
=========================
PCA embedding -> Learned Attention -> Linear -> Softmax

Bu model quantum attention'in "klasik karsiligi" — ayni parametre
budgetinde klasik attention ne yapar onu gosterir.

Attention mekanizmasi:
- 8-dim input'a 8-dim query/key/value projection uygula
- Attention score = softmax(Q * K^T / sqrt(d))
- Output = Attention * V
- Toplam ~200 parametre (quantum modelin 48 parametresiyle karsilastir)

Adil karsilastirma icin parametre sayisini quantum modele
yakin tutmak onemli. Bu yuzden tek-head, dusuk boyutlu attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ClassicalAttention(nn.Module):
    """
    Basit self-attention + linear classifier.

    Attention nasil calisir:
    1. Input x (batch, 8) -> Q, K, V projection'lari (her biri 8->8 linear)
    2. Score = Q * K^T / sqrt(8)  — benzerlik olcumu
    3. Weight = softmax(score)    — normalize et
    4. Context = Weight * V       — agirlikli toplam
    5. Output = Linear(context)   — siniflandirma

    Neden single-head?
    - Quantum modelimiz ~48 parametre
    - Multi-head attention yuzlerce parametre gerektirir
    - Adil karsilastirma icin parametre sayisini yakin tutuyoruz
    """

    def __init__(self, input_dim=8, num_classes=2):
        super().__init__()
        self.input_dim = input_dim

        # Q, K, V projections (her biri 8->8)
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)

        # Output projection + classifier
        self.out_proj = nn.Linear(input_dim, input_dim)
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        # x shape: (batch, 8)
        # Self-attention: ayni input Q, K, V olarak kullanilir
        q = self.query(x)   # (batch, 8)
        k = self.key(x)     # (batch, 8)
        v = self.value(x)   # (batch, 8)

        # Attention score: dot product / sqrt(d)
        # Burada her ornek kendi kendisiyle attention yapiyor (single token)
        # Score = sum(q * k) / sqrt(8) -> skaler
        score = (q * k).sum(dim=-1, keepdim=True) / math.sqrt(self.input_dim)
        weight = torch.sigmoid(score)  # 0-1 arasi gate

        # Gated attention: weight * v
        context = weight * v  # (batch, 8)

        # Output projection
        out = self.out_proj(context)  # (batch, 8)

        # Classifier
        return self.classifier(out)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
