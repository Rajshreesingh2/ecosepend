import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("data/processed/user_actions.csv")

# Build user-action matrix
matrix = df.pivot_table(index="user_id", columns="action", values="rating", fill_value=0)

# Item-item similarity (collaborative filtering)
item_sim = cosine_similarity(matrix.T)
item_sim_df = pd.DataFrame(item_sim, index=matrix.columns, columns=matrix.columns)

# User-item similarity
user_sim = cosine_similarity(matrix)
user_sim_df = pd.DataFrame(user_sim, index=matrix.index, columns=matrix.index)

os.makedirs("models", exist_ok=True)
with open("models/recsys_matrix.pkl", "wb") as f:
    pickle.dump(matrix, f)
with open("models/item_similarity.pkl", "wb") as f:
    pickle.dump(item_sim_df, f)
with open("models/user_similarity.pkl", "wb") as f:
    pickle.dump(user_sim_df, f)

print("✅ RecSys model trained and saved!")
print(f"   Matrix shape: {matrix.shape}")
print(f"   Item similarity matrix: {item_sim_df.shape}")