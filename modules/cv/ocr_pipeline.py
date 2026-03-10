import cv2
import easyocr
import matplotlib.pyplot as plt

# ── Step 1: Load the image ──
image_path = "data/raw/test_receipt.jpg"
img = cv2.imread(image_path)

# ── Step 2: Preprocess with OpenCV ──
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# ── Step 3: Run EasyOCR ──
print("🔍 Reading text from image...")
reader = easyocr.Reader(['en'])
results = reader.readtext(binary)

# ── Step 4: Print what it found ──
print("\n📄 Text found in your image:\n")
for (bbox, text, confidence) in results:
    print(f"  {confidence:.0%} confident → {text}")
    