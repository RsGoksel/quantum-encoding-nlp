"""
Classical Baseline Model
========================
En basit model: PCA embedding -> Linear -> Softmax

Bu model "alt sinir" — bundan kotu olmak kabul edilemez.
8 boyutlu PCA vektoru dogrudan 2-sinifli linear katmana girer.

Parametre sayisi: 8*2 + 2 = 18 (weight + bias)
"""

import torch
import torch.nn as nn


class ClassicalBaseline(nn.Module):
    """
    PCA(768->8) embedding'leri dogrudan linear katmana verir.

    Forward:
        x: (batch, 8)  — PCA-reduced DistilBERT embeddings
        -> (batch, 2)  — logits (positive/negative)
    """

    def __init__(self, input_dim=8, num_classes=2):
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.classifier(x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
