"""
Tests DistilBERT on messy real-world Indian transactions
that TF-IDF would struggle to classify correctly.
"""

import torch
import pickle
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# ── Load model ──
print("⏳ Loading DistilBERT...")
tokenizer = DistilBertTokenizer.from_pretrained("models/distilbert")
model = DistilBertForSequenceClassification.from_pretrained("models/distilbert")
model.eval()

with open("models/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

def predict(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=64
    )
    with torch.no_grad():
        outputs = model(**inputs)
    pred = torch.argmax(outputs.logits, dim=1).item()
    confidence = torch.softmax(outputs.logits, dim=1).max().item()
    return le.classes_[pred], confidence

# ── Test on messy real-world text ──
test_cases = [
    # Easy ones
    "Zomato Payment UPI",
    "BPCL FUEL STATION 04/03",
    # Messy ones TF-IDF struggles with
    "paid for lunch with colleagues",
    "filled up the tank on highway",
    "bought new headphones online",
    "monthly streaming service fee",
    "doctor visit charges",
    "college semester fees paid",
    "electricity due date payment",
    # Your real transaction
    "Zomato Limited UPI zomato-order@ptybl"
]

print("\n🧪 DistilBERT Predictions:\n")
for text in test_cases:
    category, confidence = predict(text)
    print(f"  '{text}'")
    print(f"   → {category} ({confidence:.0%} confident)\n")