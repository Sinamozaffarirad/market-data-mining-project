from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET

DOCX = Path(r"C:\UNI\BA Final Project\report\Project_Report_frontmatter_footers_fixed.docx")
NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
}

with zipfile.ZipFile(DOCX) as archive:
    document = ET.fromstring(archive.read("word/document.xml"))
    rels = ET.fromstring(archive.read("word/_rels/document.xml.rels"))
    targets = {e.get("Id"): e.get("Target") for e in rels}
    for p_index, paragraph in enumerate(document.findall(".//w:p", NS)):
        for blip in paragraph.findall(".//a:blip", NS):
            rid = blip.get("{%s}embed" % NS["r"])
            if rid and rid.startswith("rId") and rid[3:].isdigit() and 57 <= int(rid[3:]) <= 76:
                extent = paragraph.find(".//wp:extent", NS)
                print(rid, targets[rid], "paragraph", p_index,
                      "extent", (extent.get("cx"), extent.get("cy")) if extent is not None else None)
