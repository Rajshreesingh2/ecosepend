import sys
sys.path.append("modules/cv")
from receipt_parser import parse_upi_screenshot

def make_fake_ocr(lines):
    """Helper — converts plain text lines into EasyOCR format"""
    return [([[0,0],[1,0],[1,1],[0,1]], text, 0.99) for text in lines]

def test_item_extracted():
    ocr = make_fake_ocr(["Zomato Payment", "Food"])
    result = parse_upi_screenshot(ocr)
    assert result["item"] == "Zomato Payment"

def test_category_extracted():
    ocr = make_fake_ocr(["Zomato Payment", "Food"])
    result = parse_upi_screenshot(ocr)
    assert result["category"] == "Food"

def test_amount_extracted():
    ocr = make_fake_ocr(["Rupees Five Hundred Sixty Five Only"])
    result = parse_upi_screenshot(ocr)
    assert result["amount"] == 565

def test_date_extracted():
    ocr = make_fake_ocr(["09 Mar 2026"])
    result = parse_upi_screenshot(ocr)
    assert result["date"] == "09 Mar 2026"

def test_empty_input():
    result = parse_upi_screenshot([])
    assert result["item"] is None