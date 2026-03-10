import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import os

os.makedirs("mlflow/experiments", exist_ok=True)
mlflow.set_tracking_uri("mlflow/experiments")
mlflow.set_experiment("ecospend-co2-predictor")

CO2_FACTORS = {
    "transport": 0.00231, "food": 0.00189, "shopping": 0.00143,
    "utilities": 0.00298, "health": 0.00089,
    "entertainment": 0.00065, "education": 0.00045
}

# Generate data
np.random.seed(42)
n = 1200
data = {
    "transport":     np.random.uniform(200, 3000, n),
    "food":          np.random.uniform(300, 5000, n),
    "shopping":      np.random.uniform(0, 8000, n),
    "utilities":     np.random.uniform(200, 2000, n),
    "health":        np.random.uniform(0, 3000, n),
    "entertainment": np.random.uniform(0, 2000, n),
    "education":     np.random.uniform(0, 5000, n),
}
df = pd.DataFrame(data)
df["co2_kg"] = sum(df[c] * CO2_FACTORS[c] for c in CO2_FACTORS)
df["co2_kg"] += np.random.normal(0, 0.5, n)

X = df.drop("co2_kg", axis=1)
y = df["co2_kg"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Experiment with different params
experiments = [
    {"n_estimators": 100, "learning_rate": 0.1,  "max_depth": 4},
    {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 5},
    {"n_estimators": 300, "learning_rate": 0.03, "max_depth": 6},
]

best_mae = float("inf")
best_model = None
best_params = None

for params in experiments:
    with mlflow.start_run():
        model = lgb.LGBMRegressor(**params, random_state=42, verbose=-1)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        r2  = r2_score(y_test, preds)

        # Log to MLflow
        mlflow.log_params(params)
        mlflow.log_metric("mae", round(mae, 4))
        mlflow.log_metric("r2",  round(r2,  4))
        mlflow.sklearn.log_model(model, "lgbm_model")

        print(f"Params: {params}")
        print(f"  MAE: {mae:.4f} | R²: {r2:.4f}")

        if mae < best_mae:
            best_mae   = mae
            best_model = model
            best_params = params

print(f"\n🏆 Best model: {best_params}")
print(f"   MAE: {best_mae:.4f}")

# Save best model
with open("models/lgbm_co2_predictor.pkl", "wb") as f:
    pickle.dump(best_model, f)
print("✅ Best model saved to models/lgbm_co2_predictor.pkl")