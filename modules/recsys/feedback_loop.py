import pandas as pd
import numpy as np
import pickle
import os

def update_feedback(user_id, action, new_rating):
    df = pd.read_csv("data/processed/user_actions.csv")

    mask = (df["user_id"] == user_id) & (df["action"] == action)
    if mask.any():
        df.loc[mask, "rating"] = new_rating
        print(f"✅ Updated {user_id} → '{action}' rating to {new_rating}")
    else:
        new_row = pd.DataFrame([{"user_id": user_id, "action": action, "rating": new_rating}])
        df = pd.concat([df, new_row], ignore_index=True)
        print(f"✅ Added new feedback: {user_id} → '{action}' rating {new_rating}")

    df.to_csv("data/processed/user_actions.csv", index=False)

    # Retrain silently
    matrix = df.pivot_table(index="user_id", columns="action", values="rating", fill_value=0)
    from sklearn.metrics.pairwise import cosine_similarity
    user_sim = cosine_similarity(matrix)
    user_sim_df = pd.DataFrame(user_sim, index=matrix.index, columns=matrix.index)

    with open("models/recsys_matrix.pkl", "wb") as f:
        pickle.dump(matrix, f)
    with open("models/user_similarity.pkl", "wb") as f:
        pickle.dump(user_sim_df, f)
    print("🔄 Model retrained with new feedback!")

if __name__ == "__main__":
    update_feedback("user_1", "switched_to_metro", 5)
    update_feedback("user_1", "carpooled", 4)
    update_feedback("new_user_rajshree", "ordered_veg_meal", 5)