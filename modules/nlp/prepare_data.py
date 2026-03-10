"""
Phase 3 - Data Preparation
Generates realistic Indian UPI transaction dataset for NLP training.
Combines with creditcard.csv amounts for realistic values.
"""

import pandas as pd
import random
from faker import Faker

fake = Faker('en_IN')
random.seed(42)

# ── Transaction templates ──
transactions = {
    "Food": [
        "Zomato order", "Swiggy delivery", "McDonald's payment",
        "Dominos pizza", "Blinkit groceries", "BigBasket order",
        "Zepto delivery", "Restaurant bill", "Cafe Coffee Day",
        "Starbucks payment"
    ],
    "Transport": [
        "Uber ride", "Ola cab booking", "BMTC bus pass",
        "BPCL fuel station", "Indian Oil petrol", "Metro card recharge",
        "Rapido bike ride", "InDrive cab", "HP fuel pump",
        "Auto rickshaw UPI"
    ],
    "Shopping": [
        "Amazon purchase", "Flipkart order", "Myntra clothing",
        "Meesho order", "Nykaa cosmetics", "Ajio fashion",
        "Reliance Digital", "Croma electronics", "DMart shopping",
        "Lifestyle store"
    ],
    "Utilities": [
        "Electricity bill payment", "Airtel recharge", "Jio recharge",
        "Vi mobile recharge", "BWSSB water bill", "Piped gas payment",
        "Broadband bill", "DTH recharge", "Municipal tax",
        "Society maintenance"
    ],
    "Health": [
        "Apollo pharmacy", "MedPlus medicines", "1mg order",
        "Netmeds delivery", "Doctor consultation fee", "Practo appointment",
        "PharmEasy order", "Hospital OPD payment", "Cult.fit membership",
        "Gym monthly fee"
    ],
    "Entertainment": [
        "Netflix subscription", "Amazon Prime", "Hotstar premium",
        "Spotify premium", "BookMyShow tickets", "PVR cinemas",
        "Steam game purchase", "YouTube premium", "Zee5 subscription",
        "SonyLIV subscription"
    ],
    "Education": [
        "Udemy course purchase", "Coursera subscription", "BYJU's fee",
        "Unacademy subscription", "College fee payment", "Books purchase",
        "Coding Ninjas course", "GeeksforGeeks subscription",
        "LeetCode premium", "Internshala course"
    ]
}

# ── Amount ranges per category ──
amount_ranges = {
    "Food":          (80, 800),
    "Transport":     (50, 1500),
    "Shopping":      (200, 5000),
    "Utilities":     (100, 3000),
    "Health":        (100, 2000),
    "Entertainment": (99, 999),
    "Education":     (500, 10000)
}

# ── Generate dataset ──
rows = []
for _ in range(1500):
    category = random.choice(list(transactions.keys()))
    merchant = random.choice(transactions[category])
    amount = random.randint(*amount_ranges[category])

    # Add some noise to make it realistic
    suffix = random.choice([
        "", " UPI", " payment", f" {fake.numerify('######')}",
        f" on {random.choice(['Paytm', 'PhonePe', 'GPay', 'BHIM'])}"
    ])

    rows.append({
        "transaction_text": merchant + suffix,
        "amount": amount,
        "category": category
    })

df = pd.DataFrame(rows)

# ── Shuffle ──
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# ── Save ──
df.to_csv("data/processed/transactions_labelled.csv", index=False)
print(f"✅ Generated {len(df)} transactions")
print(f"\nCategory distribution:\n{df['category'].value_counts()}")
print(f"\nSample:\n{df.head(10).to_string()}")