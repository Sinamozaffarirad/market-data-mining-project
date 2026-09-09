from pathlib import Path
import io
import shutil
import zipfile
import xml.etree.ElementTree as ET
from PIL import Image, ImageOps

SOURCE = Path(r"C:\UNI\BA Final Project\report\Project_Report_frontmatter_footers_fixed.docx")
OUTPUT = Path(r"C:\UNI\BA Final Project\report\Project_Report_season6_screenshots.docx")
CAPTURES = Path(r"C:\UNI\BA Final Project\report\_season6_captures")
REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"


def fit_to_slot(image: Image.Image, size: tuple[int, int]) -> bytes:
    """Fill the existing Word image slot without distortion."""
    fitted = ImageOps.fit(image.convert("RGB"), size, method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
    data = io.BytesIO()
    fitted.save(data, "PNG", optimize=True)
    return data.getvalue()


with zipfile.ZipFile(SOURCE, "r") as source:
    rels = ET.fromstring(source.read("word/_rels/document.xml.rels"))
    relationships = {e.get("Id"): e.get("Target") for e in rels.findall(f"{{{REL_NS}}}Relationship")}
    replacements = {}
    for figure in range(1, 21):
        rid = f"rId{56 + figure}"
        part = "word/" + relationships[rid]
        original = Image.open(io.BytesIO(source.read(part)))
        capture = Image.open(CAPTURES / f"fig6-{figure:02d}.png")
        replacements[part] = fit_to_slot(capture, original.size)
        print(f"{rid} -> {part}: {original.size}")

    with zipfile.ZipFile(OUTPUT, "w", zipfile.ZIP_DEFLATED) as destination:
        for item in source.infolist():
            payload = replacements.get(item.filename, source.read(item.filename))
            destination.writestr(item, payload)

print(f"Created {OUTPUT}")
