import csv
import cv2
import easyocr
from receipt_parser import parse_upi_screenshot

def save_to_csv(parsed: dict, output_path: str):
    """Saves parsed receipt data to CSV for Phase 3 NLP module."""
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=parsed.keys())
        writer.writeheader()
        writer.writerow(parsed)
    print(f"✅ Saved to {output_path}")

if __name__ == "__main__":
    img = cv2.imread("data/raw/test_receipt.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255,
                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    reader = easyocr.Reader(['en'])
    results = reader.readtext(binary)

    parsed = parse_upi_screenshot(results)
    save_to_csv(parsed, "data/processed/transactions.csv")