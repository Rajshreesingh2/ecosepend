"""
Phase 3 - NLP Module
Trains a TF-IDF + Logistic Regression baseline classifier first.
Then we'll upgrade to DistilBERT on top of this.

Why baseline first?
- Trains in seconds on CPU
- Gives us a benchmark F1 score to beat with DistilBERT
- Real ML engineers always build a baseline before heavy models
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import pickle
import os

# ── Load data ──
df = pd.read_csv("data/processed/transactions_labelled.csv")
print(f"✅ Loaded {len(df)} transactions")
print(f"Categories: {df['category'].unique()}\n")

# ── Split into train and test ──
X_train, X_test, y_train, y_test = train_test_split(
    df["transaction_text"],
    df["category"],
    test_size=0.2,
    random_state=42,
    stratify=df["category"]   # equal split per category
)

print(f"Training on {len(X_train)} samples")
print(f"Testing on  {len(X_test)} samples\n")

# ── Step 1: TF-IDF Vectorizer ──
# Converts text → numbers that ML can understand
# Example: "Zomato order" → [0.0, 0.8, 0.0, 0.4, ...]
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),   # captures "Zomato order" as one feature
    max_features=5000
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ── Step 2: Train Logistic Regression ──
print("🔄 Training baseline model...")
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_vec, y_train)

# ── Step 3: Evaluate ──
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n✅ Baseline Model Results:")
print(f"   Accuracy: {accuracy:.2%}\n")
print("📊 Classification Report:")
print(classification_report(y_test, y_pred))

# ── Step 4: Save model ──
os.makedirs("models", exist_ok=True)
with open("models/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
with open("models/baseline_classifier.pkl", "wb") as f:
    pickle.dump(model, f)
print("💾 Model saved to models/")

# ── Step 5: Test on your real transaction ──
print("\n🧪 Testing on your real Zomato transaction:")
test_transactions = [
    "Zomato Payment UPI",
    "BPCL FUEL STATION",
    "Airtel recharge on GPay",
    "Netflix subscription",
    "Apollo pharmacy order"
]

for text in test_transactions:
    vec = vectorizer.transform([text])
    prediction = model.predict(vec)[0]
    confidence = model.predict_proba(vec).max()
    print(f"  '{text}'")
    print(f"   → {prediction} ({confidence:.0%} confident)\n")