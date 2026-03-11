import gradio as gr
import pickle
import pandas as pd
import numpy as np
import cv2
import easyocr
import re
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Load models
with open("models/lgbm_co2_predictor.pkl", "rb") as f:
    lgbm_model = pickle.load(f)
with open("models/tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)
with open("models/baseline_classifier.pkl", "rb") as f:
    baseline_clf = pickle.load(f)
with open("models/recsys_matrix.pkl", "rb") as f:
    recsys_matrix = pickle.load(f)
with open("models/user_similarity.pkl", "rb") as f:
    user_sim_df = pickle.load(f)

reader = easyocr.Reader(['en'], gpu=False)

CO2_FACTORS = {
    "Transport": 0.00231, "Food": 0.00189, "Shopping": 0.00143,
    "Utilities": 0.00298, "Health": 0.00089,
    "Entertainment": 0.00065, "Education": 0.00045
}

CO2_SAVINGS = {
    "Switch to metro/bus": 18.0,
    "Order vegetarian meals": 6.0,
    "Buy secondhand": 3.0,
    "Reduce AC temperature": 8.0,
    "Use reusable bags": 1.5,
    "Carpool to work": 10.0,
    "Choose local produce": 4.0,
    "Air dry clothes": 2.0,
    "Turn off standby devices": 3.5,
    "Take shorter showers": 1.0
}

# Tab 1: OCR Receipt Scanner
def scan_receipt(image):
    if image is None:
        return "Please upload an image first.", "", ""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        results = reader.readtext(thresh)
        text = " ".join([r[1] for r in results])

        # Smart amount extraction
        amount = None

        # Try word-to-number for Indian receipts ("Five Hundred Sixty Five")
        word_map = {
            'zero':0,'one':1,'two':2,'three':3,'four':4,'five':5,
            'six':6,'seven':7,'eight':8,'nine':9,'ten':10,
            'eleven':11,'twelve':12,'thirteen':13,'fourteen':14,'fifteen':15,
            'sixteen':16,'seventeen':17,'eighteen':18,'nineteen':19,'twenty':20,
            'thirty':30,'forty':40,'fifty':50,'sixty':60,'seventy':70,
            'eighty':80,'ninety':90,'hundred':100,'thousand':1000
        }
        words = text.lower().split()
        nums = []
        i = 0
        while i < len(words):
            w = re.sub(r'[^a-z]', '', words[i])
            if w in word_map:
                nums.append(word_map[w])
            i += 1
        if nums:
            total = 0
            current = 0
            for n in nums:
                if n == 100:
                    current *= 100
                elif n == 1000:
                    total += current * 1000
                    current = 0
                else:
                    current += n
            word_amount = total + current
            if 10 <= word_amount <= 100000:
                amount = float(word_amount)

        # Fallback: regex for digits near amount keywords
        if not amount:
            patterns = [
                r'(?:amount|total|paid|rs\.?|inr)[^\d]*(\d{2,6})',
                r'(\d{3,6})\s*(?:rs|inr|rupees)',
            ]
            for pattern in patterns:
                match = re.search(pattern, text.lower())
                if match:
                    amt = float(match.group(1).replace(',', ''))
                    if 10 <= amt <= 100000:
                        amount = amt
                        break

        # Last fallback: pick most reasonable number
        if not amount:
            all_nums = [float(n.replace(',','')) for n in re.findall(r'\b\d{3,6}\b', text)]
            valid = [n for n in all_nums if 10 <= n <= 100000]
            if valid:
                amount = min(valid)

        # Classify category
        vec = tfidf.transform([text])
        category = baseline_clf.predict(vec)[0]
        co2 = round(amount * CO2_FACTORS.get(category, 0.001), 3) if amount else 0

        result = f"""## Receipt Scanned Successfully!

**Raw Text Detected:**
{text[:400]}

---
**Category:** {category}
**Amount:** Rs. {amount if amount else 'Not detected'}
**CO2 Impact:** {co2} kg/month
"""
        return result, category, f"Rs. {amount}" if amount else "Not detected"

    except Exception as e:
        return f"Error: {str(e)}", "", ""


# Tab 2: NLP Classifier
def classify_text(description):
    if not description.strip():
        return "Please enter a transaction description."

    vec = tfidf.transform([description])
    category = baseline_clf.predict(vec)[0]
    proba = baseline_clf.predict_proba(vec)[0]
    confidence = round(float(proba.max()) * 100, 1)
    classes = baseline_clf.classes_

    bars = ""
    for cls, prob in sorted(zip(classes, proba), key=lambda x: x[1], reverse=True):
        pct = round(prob * 100, 1)
        bar = "|" * int(pct / 5)
        bars += f"\n**{cls}** {bar} {pct}%"

    return f"""## Classification Result

**Input:** {description}
**Predicted Category:** {category}
**Confidence:** {confidence}%

### All Category Probabilities:
{bars}
"""


# Tab 3: CO2 Predictor
def predict_co2(transport, food, shopping, utilities, health, entertainment, education):
    spend = {
        "transport": transport, "food": food, "shopping": shopping,
        "utilities": utilities, "health": health,
        "entertainment": entertainment, "education": education
    }

    features = pd.DataFrame([spend])
    predicted = round(float(lgbm_model.predict(features)[0]), 2)
    total_spend = sum(spend.values())

    raw_co2 = {k: round(v * list(CO2_FACTORS.values())[i], 3)
               for i, (k, v) in enumerate(spend.items())}
    total_raw = sum(raw_co2.values())

    score = "A+" if predicted < 10 else "A" if predicted < 13 else "B" if predicted < 16 else "C"
    goal_pct = round((predicted / 15) * 100, 1)
    status = "Within goal!" if predicted <= 15 else "Exceeds goal - take action!"

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.patch.set_facecolor('#f0f7f2')
    cats = list(raw_co2.keys())
    vals = list(raw_co2.values())
    colors = ['#e74c3c','#f39c12','#9b59b6','#e67e22','#3498db','#1abc9c','#2ecc71']
    ax1.barh(cats, vals, color=colors)
    ax1.set_title('CO2 by Category (kg)', fontweight='bold')
    ax1.set_facecolor('#f8fdf9')
    ax2.pie(vals, labels=cats, colors=colors, autopct='%1.1f%%', startangle=90)
    ax2.set_title('CO2 Distribution', fontweight='bold')
    plt.tight_layout()
    chart_path = "co2_breakdown.png"
    plt.savefig(chart_path, dpi=100, bbox_inches='tight')
    plt.close()

    result = f"""## CO2 Prediction Result

| Metric | Value |
|--------|-------|
| **Predicted CO2** | {predicted} kg/month |
| **Total Spend** | Rs. {total_spend:,} |
| **Eco Score** | {score} |
| **Goal Progress** | {goal_pct}% of 15 kg |
| **Status** | {status} |

### Category Breakdown:
"""
    for cat, co2 in raw_co2.items():
        pct = round((co2/total_raw)*100, 1) if total_raw > 0 else 0
        bar = "|" * int(pct/5)
        result += f"\n**{cat.title()}** {bar} {co2} kg ({pct}%)"

    return result, chart_path


# Tab 4: Recommendations
def get_recommendations(user_id):
    if not user_id.strip():
        user_id = "new_user"

    if user_id in recsys_matrix.index:
        user_row = recsys_matrix.loc[user_id]
        already_done = user_row[user_row > 0].index.tolist()
        similar_users = user_sim_df[user_id].sort_values(ascending=False)[1:11].index
        scores = recsys_matrix.loc[similar_users].mean(axis=0)
        unseen = scores.drop(labels=already_done, errors="ignore")
        top_keys = unseen.sort_values(ascending=False).head(5).index.tolist()
        recs = [(a, CO2_SAVINGS.get(a, 2.0)) for a in top_keys]
        mode = f"Personalised for {user_id}"
    else:
        recs = sorted(CO2_SAVINGS.items(), key=lambda x: x[1], reverse=True)[:5]
        mode = "Default recommendations (new user)"

    total = sum(s for _, s in recs)
    result = f"## Eco Action Recommendations\n\n**{mode}**\n\n"
    result += "| Action | CO2 Saved per Month |\n|--------|--------------------|\n"
    for action, saving in recs:
        result += f"| {action} | {saving} kg |\n"
    result += f"\n**Total potential saving: {total} kg CO2/month**"
    result += f"\n\nEquivalent to planting {int(total/2)} trees!"
    return result


# Build Gradio UI
with gr.Blocks(title="EcoSpend AI") as demo:

    gr.Markdown("""
    # EcoSpend AI
    ### AI-powered Carbon Footprint Predictor from Spending Habits
    Built with LightGBM | DistilBERT | EasyOCR | Collaborative Filtering
    """)

    with gr.Tabs():

        with gr.Tab("Scan Receipt"):
            gr.Markdown("### Upload a UPI payment screenshot or receipt image")
            with gr.Row():
                img_input = gr.Image(label="Upload Receipt Image", type="numpy")
                with gr.Column():
                    ocr_output = gr.Markdown()
                    ocr_cat = gr.Textbox(label="Category detected")
                    ocr_amt = gr.Textbox(label="Amount detected")
            scan_btn = gr.Button("Scan Receipt", variant="primary")
            scan_btn.click(scan_receipt, inputs=img_input, outputs=[ocr_output, ocr_cat, ocr_amt])

        with gr.Tab("Classify Transaction"):
            gr.Markdown("### Type any transaction to auto-categorise it")
            tx_input = gr.Textbox(
                label="Transaction Description",
                placeholder="e.g. Zomato food delivery, BPCL petrol, Amazon shopping...",
                lines=2
            )
            classify_btn = gr.Button("Classify", variant="primary")
            classify_output = gr.Markdown()
            classify_btn.click(classify_text, inputs=tx_input, outputs=classify_output)
            gr.Examples(
                examples=[
                    ["Zomato food delivery"],
                    ["BPCL petrol pump"],
                    ["Amazon shopping"],
                    ["Airtel bill payment"],
                    ["Apollo pharmacy"],
                    ["Uber cab ride"],
                    ["Udemy online course"]
                ],
                inputs=tx_input
            )

        with gr.Tab("Predict CO2"):
            gr.Markdown("### Enter monthly spending to predict your carbon footprint")
            with gr.Row():
                with gr.Column():
                    t  = gr.Slider(0, 5000, value=800,  label="Transport (Rs.)")
                    f  = gr.Slider(0, 5000, value=565,  label="Food (Rs.)")
                    s  = gr.Slider(0, 8000, value=1200, label="Shopping (Rs.)")
                    u  = gr.Slider(0, 3000, value=500,  label="Utilities (Rs.)")
                    h  = gr.Slider(0, 3000, value=300,  label="Health (Rs.)")
                    e  = gr.Slider(0, 2000, value=299,  label="Entertainment (Rs.)")
                    ed = gr.Slider(0, 5000, value=999,  label="Education (Rs.)")
                with gr.Column():
                    co2_output = gr.Markdown()
                    co2_chart  = gr.Image(label="CO2 Breakdown Chart")
            predict_btn = gr.Button("Predict My CO2", variant="primary")
            predict_btn.click(predict_co2, inputs=[t,f,s,u,h,e,ed], outputs=[co2_output, co2_chart])

        with gr.Tab("Recommendations"):
            gr.Markdown("### Get personalised eco action recommendations")
            user_input = gr.Textbox(
                label="User ID",
                placeholder="e.g. user_1, user_42, or leave blank for new user",
                value="user_1"
            )
            rec_btn = gr.Button("Get Recommendations", variant="primary")
            rec_output = gr.Markdown()
            rec_btn.click(get_recommendations, inputs=user_input, outputs=rec_output)

    gr.Markdown("""
    ---
    Built by **Rajshree Singh** |
    [GitHub](https://github.com/Rajshreesingh2/ecosepend) |
    [HuggingFace](https://huggingface.co/spaces/singhrajshree/ecospend-ai)
    """)

if __name__ == "__main__":
    demo.launch(share=True)