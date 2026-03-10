"""
Phase 4 - Feature Engineering
Maps spending categories to CO₂ emission factors
and engineers features for LightGBM training.
"""

import pandas as pd
import numpy as np
import os

# ── DEFRA Carbon Emission Factors (kg CO₂ per ₹1 spent) ──
EMISSION_FACTORS = {
    "Transport":     0.00231,
    "Food":          0.00189,
    "Shopping":      0.00143,
    "Utilities":     0.00298,
    "Health":        0.00089,
    "Entertainment": 0.00065,
    "Education":     0.00045
}

def calculate_co2(category: str, amount: float) -> float:
    """Calculate CO₂ for a single transaction."""
    factor = EMISSION_FACTORS.get(category, 0.001)
    return round(amount * factor, 4)

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes transaction dataframe.
    Returns feature-engineered dataframe ready for LightGBM.
    """
    # ── Add CO₂ per transaction ──
    df["co2_kg"] = df.apply(
        lambda row: calculate_co2(row["category"], row["amount"]), axis=1
    )

    # ── One-hot encode categories ──
    category_dummies = pd.get_dummies(df["category"], prefix="cat")
    df = pd.concat([df, category_dummies], axis=1)

    # ── Amount features ──
    df["amount_log"] = np.log1p(df["amount"])
    df["amount_squared"] = df["amount"] ** 2

    # ── CO₂ intensity (CO₂ per rupee) ──
    df["co2_intensity"] = df["co2_kg"] / (df["amount"] + 1)

    return df

def generate_monthly_data(n_users=200, n_months=6):
    """
    Generates synthetic monthly spending data per user.
    This is what LightGBM trains on.
    """
    import random
    random.seed(42)
    np.random.seed(42)

    categories = list(EMISSION_FACTORS.keys())
    rows = []

    for user_id in range(n_users):
        # Each user has a spending personality
        dominant_category = random.choice(categories)

        for month in range(1, n_months + 1):
            monthly_data = {"user_id": user_id, "month": month}
            total_co2 = 0

            for category in categories:
                # Dominant category gets more spending
                if category == dominant_category:
                    amount = random.randint(2000, 8000)
                else:
                    amount = random.randint(200, 3000)

                co2 = calculate_co2(category, amount)
                monthly_data[f"spend_{category.lower()}"] = amount
                monthly_data[f"co2_{category.lower()}"] = co2
                total_co2 += co2

            monthly_data["total_co2_kg"] = round(total_co2, 2)
            rows.append(monthly_data)

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    print("⚙️  Generating monthly spending dataset...\n")
    df = generate_monthly_data(n_users=200, n_months=6)

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/monthly_spending.csv", index=False)

    print(f"✅ Generated {len(df)} monthly records")
    print(f"   Users: 200, Months: 6 each")
    print(f"\nSample row:")
    print(df.iloc[0].to_string())
    print(f"\nCO₂ stats:")
    print(df["total_co2_kg"].describe().round(2))