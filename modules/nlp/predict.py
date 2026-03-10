"""
Connects Phase 2 CV output → Phase 3 NLP classifier
Reads transactions.csv from Phase 2 and classifies each one
"""

import pandas as pd
import pickle

# ── Load saved model ──
with open("models/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
with open("models/baseline_classifier.pkl", "rb") as f:
    model = pickle.load(f)

# ── Load Phase 2 output ──
df = pd.read_csv("data/processed/transactions.csv")
print("📥 Phase 2 output loaded:")
print(df.to_string())

# ── Classify each transaction ──
print("\n🧠 NLP Classification Results:\n")
for _, row in df.iterrows():
    text = str(row.get("item", ""))
    vec = vectorizer.transform([text])
    category = model.predict(vec)[0]
    confidence = model.predict_proba(vec).max()

    print(f"  Transaction : {text}")
    print(f"  Amount      : ₹{row.get('amount', 'N/A')}")
    print(f"  Category    : {category} ({confidence:.0%} confident)")
    print(f"  Date        : {row.get('date', 'N/A')}")
    print()