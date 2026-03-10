from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle, os, sys, shutil
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = FastAPI(
    title="EcoSpend AI API",
    description="Carbon footprint predictor from spending habits",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load models once at startup ──
@app.on_event("startup")
def load_models():
    global lgbm_model, label_encoder, tfidf, baseline_clf, recsys_matrix, user_sim_df

    with open("models/lgbm_co2_predictor.pkl", "rb") as f:
        lgbm_model = pickle.load(f)
    with open("models/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    with open("models/tfidf_vectorizer.pkl", "rb") as f:
        tfidf = pickle.load(f)
    with open("models/baseline_classifier.pkl", "rb") as f:
        baseline_clf = pickle.load(f)
    with open("models/recsys_matrix.pkl", "rb") as f:
        recsys_matrix = pickle.load(f)
    with open("models/user_similarity.pkl", "rb") as f:
        user_sim_df = pickle.load(f)
    print("✅ All models loaded!")

# ── Health Check ──
@app.get("/")
def root():
    return {
        "app": "EcoSpend AI",
        "status": "running",
        "version": "1.0.0",
        "modules": ["CV", "NLP", "Predictor", "RecSys", "Forecast"]
    }

@app.get("/health")
def health():
    return {"status": "ok"}

# ── NLP: Classify transaction ──
class TransactionIn(BaseModel):
    description: str

@app.post("/classify")
def classify_transaction(tx: TransactionIn):
    vec = tfidf.transform([tx.description])
    pred = baseline_clf.predict(vec)[0]
    proba = baseline_clf.predict_proba(vec)[0]
    confidence = round(float(proba.max()) * 100, 1)

    CO2_FACTORS = {
        "Transport": 0.00231, "Food": 0.00189, "Shopping": 0.00143,
        "Utilities": 0.00298, "Health": 0.00089,
        "Entertainment": 0.00065, "Education": 0.00045
    }
    return {
        "description": tx.description,
        "category": pred,
        "confidence": confidence,
        "co2_factor": CO2_FACTORS.get(pred, 0)
    }

# ── Predictor: Predict monthly CO2 ──
class SpendingIn(BaseModel):
    transport: float = 0
    food: float = 0
    shopping: float = 0
    utilities: float = 0
    health: float = 0
    entertainment: float = 0
    education: float = 0

@app.post("/predict-co2")
def predict_co2(data: SpendingIn):
    CO2_FACTORS = {
        "transport": 0.00231, "food": 0.00189, "shopping": 0.00143,
        "utilities": 0.00298, "health": 0.00089,
        "entertainment": 0.00065, "education": 0.00045
    }
    spend = data.dict()
    total_spend = sum(spend.values())
    raw_co2 = {k: round(v * CO2_FACTORS[k], 3) for k, v in spend.items()}

    features = pd.DataFrame([spend])
    predicted_co2 = round(float(lgbm_model.predict(features)[0]), 2)

    breakdown = {}
    for cat, co2 in raw_co2.items():
        pct = round((co2 / sum(raw_co2.values())) * 100, 1) if sum(raw_co2.values()) > 0 else 0
        breakdown[cat] = {"spend": spend[cat], "co2_kg": co2, "percentage": pct}

    return {
        "predicted_co2_kg": predicted_co2,
        "total_spend": total_spend,
        "goal_kg": 15.0,
        "within_goal": predicted_co2 <= 15.0,
        "eco_score": "A+" if predicted_co2 < 10 else "A" if predicted_co2 < 13 else "B" if predicted_co2 < 16 else "C",
        "breakdown": breakdown
    }

# ── RecSys: Get recommendations ──
CO2_SAVINGS = {
    "switched_to_metro": 18.0, "ordered_veg_meal": 6.0,
    "bought_secondhand": 3.0, "reduced_ac_temp": 8.0,
    "used_reusable_bag": 1.5, "carpooled": 10.0,
    "chose_local_produce": 4.0, "air_dried_clothes": 2.0,
    "turned_off_standby": 3.5, "took_shorter_shower": 1.0
}

@app.get("/recommend/{user_id}")
def get_recommendations(user_id: str, top_n: int = 5):
    if user_id not in recsys_matrix.index:
        recs = sorted(CO2_SAVINGS.items(), key=lambda x: x[1], reverse=True)[:top_n]
        cold_start = True
    else:
        user_row = recsys_matrix.loc[user_id]
        already_done = user_row[user_row > 0].index.tolist()
        similar_users = user_sim_df[user_id].sort_values(ascending=False)[1:11].index
        scores = recsys_matrix.loc[similar_users].mean(axis=0)
        unseen = scores.drop(labels=already_done, errors="ignore")
        top_actions = unseen.sort_values(ascending=False).head(top_n).index.tolist()
        recs = [(a, CO2_SAVINGS.get(a, 0)) for a in top_actions]
        cold_start = False

    return {
        "user_id": user_id,
        "cold_start": cold_start,
        "recommendations": [
            {"action": a, "co2_saving_kg_per_month": s} for a, s in recs
        ],
        "total_potential_saving_kg": round(sum(s for _, s in recs), 1)
    }

# ── Feedback: Update user rating ──
class FeedbackIn(BaseModel):
    user_id: str
    action: str
    rating: int

@app.post("/feedback")
def submit_feedback(fb: FeedbackIn):
    if fb.rating < 1 or fb.rating > 5:
        raise HTTPException(status_code=400, detail="Rating must be 1-5")
    if fb.action not in CO2_SAVINGS:
        raise HTTPException(status_code=400, detail=f"Unknown action: {fb.action}")

    df = pd.read_csv("data/processed/user_actions.csv")
    mask = (df["user_id"] == fb.user_id) & (df["action"] == fb.action)
    if mask.any():
        df.loc[mask, "rating"] = fb.rating
    else:
        new_row = pd.DataFrame([{"user_id": fb.user_id, "action": fb.action, "rating": fb.rating}])
        df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv("data/processed/user_actions.csv", index=False)

    return {"status": "success", "message": f"Feedback saved for {fb.user_id}"}

# ── CV: Upload receipt ──
@app.post("/scan-receipt")
async def scan_receipt(file: UploadFile = File(...)):
    allowed = ["image/jpeg", "image/png", "image/jpg"]
    if file.content_type not in allowed:
        raise HTTPException(status_code=400, detail="Only JPG/PNG images allowed")

    save_path = f"data/raw/uploaded_{file.filename}"
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        from modules.cv.receipt_parser import parse_receipt
        result = parse_receipt(save_path)
    except Exception as e:
        result = {"note": "OCR parsing pending", "file_saved": save_path}

    return {"status": "uploaded", "file": save_path, "parsed": result}

# ── Summary: Full pipeline for a user ──
@app.get("/summary/{user_id}")
def get_summary(user_id: str):
    sample_spend = {
        "transport": 800, "food": 565, "shopping": 1200,
        "utilities": 500, "health": 300,
        "entertainment": 299, "education": 999
    }
    CO2_FACTORS = {
        "transport": 0.00231, "food": 0.00189, "shopping": 0.00143,
        "utilities": 0.00298, "health": 0.00089,
        "entertainment": 0.00065, "education": 0.00045
    }
    features = pd.DataFrame([sample_spend])
    predicted_co2 = round(float(lgbm_model.predict(features)[0]), 2)

    recs_resp = get_recommendations(user_id, top_n=3)

    return {
        "user_id": user_id,
        "month": "March 2026",
        "predicted_co2_kg": predicted_co2,
        "eco_score": "A+" if predicted_co2 < 10 else "A",
        "total_spend": sum(sample_spend.values()),
        "top_recommendations": recs_resp["recommendations"],
        "within_goal": predicted_co2 <= 15.0
    }