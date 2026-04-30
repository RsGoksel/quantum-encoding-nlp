"""SST-2 only preprocessing — IMDb zaten tamamlandi."""
import os, sys, time, torch
os.environ["PYTHONIOENCODING"] = "utf-8"
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
sys.path.insert(0, str(PROJECT_ROOT))
from data.preprocess import extract_embeddings, apply_pca

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Dataset
from datasets import load_dataset
print("SST-2 yukleniyor...")
t0 = time.time()
sst2 = load_dataset("glue", "sst2")
print(f"  -> Sure: {time.time()-t0:.1f}s")

# Model
from transformers import DistilBertModel, DistilBertTokenizer
print("DistilBERT yukleniyor...")
t0 = time.time()
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased")
model.eval()
model.to(device)
print(f"  -> Sure: {time.time()-t0:.1f}s")

# SST-2 split
train_texts = sst2["train"]["sentence"]
val_texts = sst2["validation"]["sentence"]
train_labels = torch.tensor(sst2["train"]["label"], dtype=torch.long)
val_labels = torch.tensor(sst2["validation"]["label"], dtype=torch.long)

n_test = len(train_texts) // 7
gen = torch.Generator().manual_seed(42)
perm = torch.randperm(len(train_texts), generator=gen)
test_idx = perm[:n_test].tolist()
train_idx = perm[n_test:].tolist()

test_texts_split = [train_texts[i] for i in test_idx]
test_labels = train_labels[test_idx]
train_texts_split = [train_texts[i] for i in train_idx]
train_labels_split = train_labels[train_idx]

print(f"Split: train={len(train_texts_split)}, val={len(val_texts)}, test={len(test_texts_split)}")

# Embeddings
t0 = time.time()
print("\n[Train] Embedding cikartiliyor...")
train_emb = extract_embeddings(train_texts_split, model, tokenizer, device, desc="[Train]")
print(f"  -> {train_emb.shape}, sure: {time.time()-t0:.1f}s")

t0 = time.time()
print("\n[Val] Embedding cikartiliyor...")
val_emb = extract_embeddings(val_texts, model, tokenizer, device, desc="[Val]")
print(f"  -> {val_emb.shape}, sure: {time.time()-t0:.1f}s")

t0 = time.time()
print("\n[Test] Embedding cikartiliyor...")
test_emb = extract_embeddings(test_texts_split, model, tokenizer, device, desc="[Test]")
print(f"  -> {test_emb.shape}, sure: {time.time()-t0:.1f}s")

# PCA
train_pca, val_pca, test_pca, var_ratio = apply_pca(train_emb, val_emb, test_emb)

# Kaydet
save_path = DATA_DIR / "sst2_embeddings.pt"
torch.save({
    "train_X": train_pca, "train_y": train_labels_split,
    "val_X": val_pca, "val_y": val_labels,
    "test_X": test_pca, "test_y": test_labels,
    "pca_variance": var_ratio, "original_dim": 768, "reduced_dim": 8,
}, save_path)
print(f"\nKaydedildi: {save_path} ({save_path.stat().st_size / 1024:.0f} KB)")
print("TAMAMLANDI!")
