# 🌍 EcoSpend AI
### *Turn your spending into climate action*

> A full-stack AI system that scans receipts, classifies transactions, predicts your monthly CO₂ footprint, forecasts future trends, and recommends personalised eco-actions — built entirely on CPU, deployed live, ₹0 cost.



---

## 🧠 What It Does

```
📸 Receipt / UPI Screenshot
        ↓
   🔍 CV Module          →  Extracts items + amounts using OpenCV + EasyOCR
        ↓
   💬 NLP Module         →  Classifies transaction category + intent (DistilBERT)
        ↓
   📊 CO₂ Predictor      →  Maps spending → carbon footprint (LightGBM + SHAP)
        ↓
   📈 Forecast           →  Predicts next 3 months of CO₂ (Facebook Prophet)
        ↓
   🎯 Recommendations    →  Personalised eco-actions that learn from you (RecSys)
```

---

## 🤖 The 5 AI Modules

| # | Module | Task | Model | Status |
|---|--------|------|-------|--------|
| 1 | **CV — Receipt Scanner** | Photo → structured items + amounts | OpenCV + EasyOCR | ✅ Complete |
| 2 | **NLP — Transaction Classifier** | Text → category + intent | DistilBERT (fine-tuned) | 🔨 In Progress |
| 3 | **ML — CO₂ Predictor** | Spending → CO₂ score + explanation | LightGBM + SHAP | ⏳ Pending |
| 4 | **Time Series — Forecast** | Past CO₂ → 3-month prediction | Facebook Prophet | ⏳ Pending |
| 5 | **RecSys — Eco Actions** | CO₂ profile → ranked actions | Hybrid CF + Feedback Loop | ⏳ Pending |

---

## 🏗️ Tech Stack

| Layer | Tools |
|-------|-------|
| Computer Vision | OpenCV · EasyOCR |
| NLP | HuggingFace Transformers · DistilBERT · TF-IDF |
| Core ML | Scikit-learn · LightGBM · SHAP |
| Time Series | Facebook Prophet |
| RecSys | Surprise · Matrix Factorisation |
| API | FastAPI |
| UI | Streamlit |
| MLOps | MLflow · Evidently AI · Docker · GitHub Actions |
| Deploy | Render · HuggingFace Spaces |

---

## 📂 Project Structure

```
ecospend-ai/
├── data/
│   ├── raw/                  # Receipt images, UPI screenshots
│   └── processed/            # Cleaned CSVs, feature-engineered data
├── modules/
│   ├── cv/
│   │   ├── ocr_pipeline.py       # OpenCV preprocessing + EasyOCR
│   │   └── receipt_parser.py     # Structured output extractor
│   ├── nlp/
│   │   ├── classifier.py         # DistilBERT fine-tuning
│   │   └── evaluate.py           # F1, accuracy metrics
│   ├── predictor/
│   │   ├── train_lgbm.py         # LightGBM CO₂ regression
│   │   └── shap_explain.py       # Explainability layer
│   ├── timeseries/
│   │   └── prophet_model.py      # Facebook Prophet forecasting
│   └── recommender/
│       ├── hybrid.py             # CF + content-based scorer
│       └── feedback.py           # Human-in-the-loop retraining
├── api/
│   └── main.py               # FastAPI — all 5 modules wired
├── ui/
│   └── app.py                # Streamlit dashboard
├── monitoring/
│   └── drift.py              # Evidently AI drift detection
├── tests/                    # pytest unit tests
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/Rajshreesingh2/ecosepend.git
cd ecosepend

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the CV module on a receipt
python modules/cv/receipt_parser.py

# 5. Run tests
pytest tests/ -v
```

---

## 📸 CV Module — Sample Output

Input: A Zomato UPI payment screenshot

```python
{
  "item"     : "Zomato Payment",
  "category" : "Food",
  "amount"   : 565,
  "date"     : "09 Mar 2026",
  "upi_ref"  : "201609095612",
  "from"     : "Rajshree Singh",
  "to"       : "Zomato Limited",
  "bank"     : "YES BANK"
}
```

---

## 🗓️ Build Roadmap

| Phase | What | Duration | Status |
|-------|------|----------|--------|
| 1 | Foundation + EDA | Week 1–2 | ✅ Done |
| 2 | CV Receipt Scanner | Week 3–4 | ✅ Done |
| 3 | NLP Classifier | Week 5–7 | 🔨 Active |
| 4 | CO₂ Predictor + SHAP | Week 8–9 | ⏳ Pending |
| 5 | Time Series Forecast | Week 10–11 | ⏳ Pending |
| 6 | RecSys + Feedback Loop | Week 12–13 | ⏳ Pending |
| 7 | API + UI | Week 14–15 | ⏳ Pending |
| 8 | MLOps | Week 16–17 | ⏳ Pending |
| 9 | Deploy | Week 18 | ⏳ Pending |

---

## 💰 Total Cost

**₹0 / $0** — Every tool, dataset, and hosting service is completely free.

---

## 👩‍💻 Author

**Rajshree Singh** — Building EcoSpend AI as a placement project to demonstrate full-stack ML engineering skills.

GitHub: https://github.com/Rajshreesingh2
