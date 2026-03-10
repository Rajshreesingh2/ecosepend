import pandas as pd
import numpy as np
import os

np.random.seed(42)

users = [f"user_{i}" for i in range(1, 101)]
actions = [
    "switched_to_metro", "ordered_veg_meal", "bought_secondhand",
    "reduced_ac_temp", "used_reusable_bag", "carpooled",
    "chose_local_produce", "air_dried_clothes", "turned_off_standby",
    "took_shorter_shower"
]

rows = []
for user in users:
    n = np.random.randint(3, 8)
    chosen = np.random.choice(actions, n, replace=False)
    for action in chosen:
        rating = np.random.choice([3, 4, 5], p=[0.2, 0.4, 0.4])
        rows.append({"user_id": user, "action": action, "rating": rating})

df = pd.DataFrame(rows)
os.makedirs("data/processed", exist_ok=True)
df.to_csv("data/processed/user_actions.csv", index=False)
print(f"✅ Generated {len(df)} user-action ratings for {len(users)} users")
print(df.head(10))