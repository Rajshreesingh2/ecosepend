"""
Phase 3 - DistilBERT Fine-tuning
Fine-tunes DistilBERT on our Indian transaction dataset.
Upgrades from TF-IDF baseline to transformer-based classification.
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import os
import pickle

# ── Config ──
MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 64
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5

print("📦 Loading data...")
df = pd.read_csv("data/processed/transactions_labelled.csv")

# ── Encode labels ──
le = LabelEncoder()
df["label"] = le.fit_transform(df["category"])
print(f"✅ Categories: {list(le.classes_)}")

# ── Split ──
X_train, X_test, y_train, y_test = train_test_split(
    df["transaction_text"].tolist(),
    df["label"].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

# ── Dataset class ──
class TransactionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx]
        }

# ── Load tokenizer + model ──
print("\n⏳ Loading DistilBERT (downloads ~260MB first time)...")
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(le.classes_)
)

# ── Create datasets ──
train_dataset = TransactionDataset(X_train, y_train, tokenizer)
test_dataset  = TransactionDataset(X_test,  y_test,  tokenizer)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE)

# ── Optimizer ──
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# ── Training loop ──
print(f"\n🔥 Training for {EPOCHS} epochs...\n")
device = torch.device("cpu")
model.to(device)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for batch_num, batch in enumerate(train_loader):
        optimizer.zero_grad()

        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if batch_num % 10 == 0:
            print(f"  Epoch {epoch+1} | Batch {batch_num}/{len(train_loader)} | Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    print(f"\n✅ Epoch {epoch+1} complete | Avg Loss: {avg_loss:.4f}\n")

# ── Evaluation ──
print("📊 Evaluating on test set...")
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print("\n✅ DistilBERT Results:")
print(classification_report(all_labels, all_preds, target_names=le.classes_))

# ── Save model ──
os.makedirs("models/distilbert", exist_ok=True)
model.save_pretrained("models/distilbert")
tokenizer.save_pretrained("models/distilbert")

with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("💾 DistilBERT model saved to models/distilbert/")
print("\n🎉 Phase 3 DistilBERT training complete!")