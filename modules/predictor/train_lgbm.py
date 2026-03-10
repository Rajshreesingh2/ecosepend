"""
Phase 4 - LightGBM CO₂ Predictor
Trains LightGBM to predict monthly CO₂ from spending patterns.
Then uses SHAP to explain which categories drive the score.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import shap
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# ── Load data ──
df = pd.read_csv("data/processed/monthly_spending.csv")
print(f"✅ Loaded {len(df)} monthly records\n")

# ── Features and target ──
feature_cols = [
    "spend_transport", "spend_food", "spend_shopping",
    "spend_utilities", "spend_health", "spend_entertainment",
    "spend_education"
]

X = df[feature_cols]
y = df["total_co2_kg"]

# ── Train/test split ──
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training on {len(X_train)} samples")
print(f"Testing on  {len(X_test)} samples\n")

# ── Train LightGBM ──
print("🔄 Training LightGBM...")
model = lgb.LGBMRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
    verbose=-1
)
model.fit(X_train, y_train)

# ── Evaluate ──
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)

print(f"✅ Model Results:")
print(f"   MAE : {mae:.3f} kg CO₂  (avg prediction error)")
print(f"   R²  : {r2:.3f}          (1.0 = perfect)\n")

# ── SHAP Explainability ──
print("🔍 Calculating SHAP values...")
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# ── Save SHAP summary plot ──
os.makedirs("data/processed/plots", exist_ok=True)
plt.figure()
shap.summary_plot(
    shap_values, X_test,
    feature_names=feature_cols,
    show=False
)
plt.tight_layout()
plt.savefig("data/processed/plots/shap_summary.png")
plt.close()
print("✅ SHAP plot saved to data/processed/plots/shap_summary.png\n")

# ── Save model ──
os.makedirs("models", exist_ok=True)
with open("models/lgbm_co2_predictor.pkl", "wb") as f:
    pickle.dump(model, f)
with open("models/shap_explainer.pkl", "wb") as f:
    pickle.dump(explainer, f)
print("💾 Model saved to models/\n")

# ── Test on a real user prediction ──
print("🧪 Sample prediction — Rajshree's spending this month:\n")
print(f"  {'Category':15} {'Spent':>8}")
print(f"  {'-'*25}")
sample_spending = {
    "spend_transport":     800,
    "spend_food":          565,
    "spend_shopping":      1200,
    "spend_utilities":     500,
    "spend_health":        300,
    "spend_entertainment": 299,
    "spend_education":     999
}
for k, v in sample_spending.items():
    print(f"  {k.replace('spend_','').title():15} ₹{v:>7}")

sample = pd.DataFrame([sample_spending])
predicted_co2 = model.predict(sample)[0]
print(f"\n  🌍 Predicted CO₂: {predicted_co2:.2f} kg this month")

# ── SHAP explanation ──
shap_vals = explainer.shap_values(sample)[0]
contributions = dict(zip(feature_cols, shap_vals))
sorted_contrib = sorted(
    contributions.items(),
    key=lambda x: abs(x[1]),
    reverse=True
)

total_impact = sum(abs(v) for _, v in sorted_contrib)

print(f"\n  📊 What's driving your CO₂:\n")
for feature, value in sorted_contrib:
    category   = feature.replace("spend_", "").title()
    percentage = abs(value) / total_impact * 100
    bar        = "█" * int(percentage / 5)
    print(f"    {category:15} {bar:20} {percentage:.1f}%")

top = sorted_contrib[0][0].replace("spend_", "").title()
print(f"\n  💡 Top driver  : {top}")
print(f"  💡 Tip         : Reduce {top} spending to lower your footprint!")
print(f"\n  🌱 Monthly CO₂ goal: 15 kg  |  Your score: {predicted_co2:.2f} kg")
if predicted_co2 > 15:
    over = predicted_co2 - 15
    print(f"  ⚠️  You are {over:.2f} kg over your goal this month!")
else:
    print(f"  ✅ You are within your monthly CO₂ goal. Great job!")