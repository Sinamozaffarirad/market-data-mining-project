from pathlib import Path
from PIL import Image, ImageOps


BASE = Path(r"C:\UNI\BA Final Project\report\_season6_captures")

# Each crop is deliberately framed around the rendered dashboard card(s), excluding
# duplicated loading states and browser-only margins.
CROPS = {
    "fig6-01.png": ("overview-full.png", (70, 85, 1360, 850)),
    "fig6-02.png": ("overview-full.png", (70, 980, 1360, 1660)),
    "fig6-03.png": ("overview-full.png", (70, 1620, 1360, 2190)),
    "fig6-04.png": ("products-full.png", (70, 1040, 1360, 1480)),
    "fig6-05.png": ("products-full.png", (70, 1420, 1360, 2050)),
    "fig6-06.png": ("products-full.png", (70, 2460, 1360, 3180)),
    "fig6-07.png": ("time-trading-full.png", (70, 1050, 1360, 1470)),
    "fig6-08.png": ("time-trading-full.png", (70, 1900, 1360, 2370)),
    "fig6-09.png": ("time-trading-full.png", (70, 1450, 1360, 1930)),
    "fig6-10.png": ("customers-full.png", (70, 1070, 705, 1530)),
    "fig6-11.png": ("customers-full.png", (700, 1070, 1360, 1530)),
    "fig6-12.png": ("stores-full.png", (70, 1060, 1360, 1540)),
    "fig6-13.png": ("growth-loyalty-full.png", (70, 1070, 1360, 1510)),
    "fig6-14.png": ("growth-loyalty-full.png", (70, 1490, 1360, 2080)),
    "fig6-15.png": ("growth-loyalty-full.png", (70, 2040, 1360, 2820)),
    "fig6-16.png": ("pricing-promotions-full.png", (70, 1070, 705, 1530)),
    "fig6-17.png": ("pricing-promotions-full.png", (700, 1070, 1360, 1530)),
    "fig6-18.png": ("significance-results-full.png", (70, 3820, 1360, 4800)),
    "fig6-19.png": ("significance-results-full.png", (70, 1240, 1360, 3150)),
    "fig6-20.png": ("significance-expanded-full.png", (70, 5000, 1360, 5680)),
}


for target, (source, box) in CROPS.items():
    image = Image.open(BASE / source).convert("RGB")
    cropped = image.crop(box)
    cropped.save(BASE / target, "PNG", optimize=True)
    print(f"{target}: {cropped.size}")
