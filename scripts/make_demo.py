"""Make explicitly FICTIONAL pages for smoke testing (not real evaluation)."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

BASE = Path(__file__).resolve().parents[1] / "assets"
FONT_HI = "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf"
FONT_EN = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def font(path: str, size: int):
    return ImageFont.truetype(path, size)


def build():
    BASE.mkdir(parents=True, exist_ok=True)
    inputs = [
        ("notice_deadline.png", [
            ("DEMO DOCUMENT — NOT AN OFFICIAL NOTICE", "en"),
            ("नगर सेवा कार्यालय / City Services Office", "hi"),
            ("आवेदन जमा करने की अंतिम तिथि: 25 अक्टूबर 2026", "hi"),
            ("Application deadline: 25 October 2026", "en"),
            ("Helpline: 1800-000-DEMO (fictional)", "en"),
        ]),
        ("invoice.png", [
            ("DEMO DOCUMENT — FICTIONAL INVOICE", "en"),
            ("Bharat Stationery (fictional)", "en"),
            ("Invoice no: DEMO-1042", "en"),
            ("Item: Notebooks      Amount: INR 860", "en"),
            ("Total payable: INR 860", "en"),
        ]),
        ("circular.png", [
            ("DEMO DOCUMENT — FICTIONAL CIRCULAR", "en"),
            ("परिपत्र / Circular: 2026-07", "hi"),
            ("समीक्षा बैठक: 4 नवंबर 2026", "hi"),
            ("Review meeting: 4 November 2026", "en"),
            ("Location: Hall B", "en"),
        ]),
    ]
    for fname, lines in inputs:
        image = Image.new("RGB", (1050, 750), "#fffdfa")
        draw = ImageDraw.Draw(image)
        draw.rectangle((40, 40, 1010, 705), outline="#d1d5db", width=3)
        for i, (text, language) in enumerate(lines):
            draw.text((75, 100 + 95 * i), text, font=font(FONT_HI if language == "hi" else FONT_EN, 27), fill="#111827")
        image.save(BASE / fname)
    print("Generated three synthetic, labeled DEMO documents in", BASE)


if __name__ == "__main__":
    build()
