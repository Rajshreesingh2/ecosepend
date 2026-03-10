import pandas as pd
import numpy as np
import pickle

CO2_SAVINGS = {
    "switched_to_metro":     18.0,
    "ordered_veg_meal":       6.0,
    "bought_secondhand":      3.0,
    "reduced_ac_temp":        8.0,
    "used_reusable_bag":      1.5,
    "carpooled":             10.0,
    "chose_local_produce":    4.0,
    "air_dried_clothes":      2.0,
    "turned_off_standby":     3.5,
    "took_shorter_shower":    1.0
}

with open("models/recsys_matrix.pkl", "rb") as f:
    matrix = pickle.load(f)
with open("models/user_similarity.pkl", "rb") as f:
    user_sim_df = pickle.load(f)

def recommend(user_id, top_n=5):
    if user_id not in matrix.index:
        # Cold start — recommend highest CO2 savings
        recs = sorted(CO2_SAVINGS.items(), key=lambda x: x[1], reverse=True)[:top_n]
        print(f"\n🆕 Cold start recommendations for {user_id}:")
    else:
        user_row = matrix.loc[user_id]
        already_done = user_row[user_row > 0].index.tolist()

        # Find similar users
        similar_users = user_sim_df[user_id].sort_values(ascending=False)[1:11].index
        sim_matrix = matrix.loc[similar_users]

        # Score actions not yet done
        scores = sim_matrix.mean(axis=0)
        unseen = scores.drop(labels=already_done, errors="ignore")
        top_actions = unseen.sort_values(ascending=False).head(top_n).index.tolist()
        recs = [(a, CO2_SAVINGS.get(a, 0)) for a in top_actions]

        print(f"\n👤 Recommendations for {user_id}:")
        print(f"   Already done: {already_done}")

    print(f"\n   {'Action':<30} {'CO₂ Saved':>10}")
    print(f"   {'-'*42}")
    total = 0
    for action, saving in recs:
        print(f"   {action:<30} {saving:>8.1f} kg/mo")
        total += saving
    print(f"   {'-'*42}")
    print(f"   {'TOTAL POTENTIAL SAVING':<30} {total:>8.1f} kg/mo")
    return recs

if __name__ == "__main__":
    recommend("user_1")
    recommend("user_42")
    recommend("new_user_999")  # cold start test