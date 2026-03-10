import re
import cv2
import easyocr
from typing import List, Dict


def parse_upi_screenshot(ocr_results: list) -> Dict:
    """
    Takes raw EasyOCR output from a UPI screenshot.
    Returns a clean structured dictionary.
    """
    # Extract just the text from EasyOCR results (confidence > 50%)
    lines = [text for (bbox, text, confidence) in ocr_results if confidence > 0.5]

    # ── DEBUG: see all lines EasyOCR found ──
    print("\n🔍 DEBUG — All lines found by EasyOCR:\n")
    for i, line in enumerate(lines):
        print(f"  [{i}] {line}")

    data = {
        "item": None,
        "category": None,
        "amount": None,
        "date": None,
        "upi_ref": None,
        "from": None,
        "to": None,
        "bank": None
    }

    for i, line in enumerate(lines):

        # ── Amount ──
       # ── Amount ──
        # "Rupees Five Hundred Sixty Five Only" → 565
        if "rupees" in line.lower() and "only" in line.lower():
            word_to_num = {
                "zero":0,"one":1,"two":2,"three":3,"four":4,"five":5,
                "six":6,"seven":7,"eight":8,"nine":9,"ten":10,
                "eleven":11,"twelve":12,"thirteen":13,"fourteen":14,
                "fifteen":15,"sixteen":16,"seventeen":17,"eighteen":18,
                "nineteen":19,"twenty":20,"thirty":30,"forty":40,
                "fifty":50,"sixty":60,"seventy":70,"eighty":80,"ninety":90,
                "hundred":100,"thousand":1000
            }
            words = line.lower().replace("rupees","").replace("only","").split()
            total = 0
            current = 0
            for word in words:
                if word in word_to_num:
                    val = word_to_num[word]
                    if val == 100:
                        current *= 100
                    elif val == 1000:
                        total += current * 1000
                        current = 0
                    else:
                        current += val
            total += current
            if total > 0:
                data["amount"] = total

        # ── Date ──
        date_match = re.search(r'\d{2}\s\w+\s\d{4}', line)
        if date_match:
            data["date"] = date_match.group()

        # ── UPI Ref ──
        if "upi ref" in line.lower() and i + 1 < len(lines):
            ref_match = re.search(r'\d{10,}', lines[i + 1])
            if ref_match:
                data["upi_ref"] = ref_match.group()

        # ── Also catch UPI ref as a standalone long number ──
        long_number = re.fullmatch(r'\d{10,}', line.strip())
        if long_number and not data["upi_ref"]:
            data["upi_ref"] = line.strip()

        # ── Merchant / Item ──
        if "zomato" in line.lower():
            data["item"] = "Zomato Payment"
            data["category"] = "Food"

        if "swiggy" in line.lower():
            data["item"] = "Swiggy Payment"
            data["category"] = "Food"

        if "bpcl" in line.lower() or "fuel" in line.lower():
            data["item"] = "Fuel"
            data["category"] = "Transport"

        if "uber" in line.lower() or "ola" in line.lower():
            data["item"] = "Cab Ride"
            data["category"] = "Transport"

        if "amazon" in line.lower() or "flipkart" in line.lower():
            data["item"] = "Online Shopping"
            data["category"] = "Shopping"

        # ── Sender ──
        if "from" in line.lower() and i + 1 < len(lines):
            data["from"] = lines[i + 1]

        # ── Receiver ──
        if line.lower() == "to" and i + 1 < len(lines):
            data["to"] = lines[i + 1]

        # ── Bank ──
        for bank in ["union bank", "hdfc", "sbi", "icici", "axis", "yes bank", "kotak", "paytm"]:
            if bank in line.lower():
                data["bank"] = line

    return data


if __name__ == "__main__":

    # ── Load and preprocess image ──
    img_path = "data/raw/test_receipt.jpg"
    img = cv2.imread(img_path)

    if img is None:
        print("❌ Image not found! Check path:", img_path)
        exit()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # ── Run OCR ──
    print("⏳ Running OCR... (takes 20-30 seconds first time)")
    reader = easyocr.Reader(['en'])
    results = reader.readtext(binary)

    # ── Parse ──
    parsed = parse_upi_screenshot(results)

    # ── Print structured output ──
    print("\n✅ Final Structured Output:\n")
    for key, value in parsed.items():
        if value:
            print(f"  {key:10} → {value}")

    # ── Show what's still missing ──
    missing = [key for key, value in parsed.items() if not value]
    if missing:
        print(f"\n⚠️  Could not extract: {', '.join(missing)}")
        print("   (Check DEBUG lines above to fix these manually)\n")