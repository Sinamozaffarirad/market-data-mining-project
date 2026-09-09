from copy import deepcopy
from pathlib import Path
import re
from zipfile import ZIP_DEFLATED, ZipFile

from lxml import etree


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PR_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
CT_NS = "http://schemas.openxmlformats.org/package/2006/content-types"
NS = {"w": W_NS, "r": R_NS, "pr": PR_NS, "ct": CT_NS}
FOOTER_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/footer"
FOOTER_CT = "application/vnd.openxmlformats-officedocument.wordprocessingml.footer+xml"


def footer_xml(template: bytes, label: str) -> bytes:
    root = etree.fromstring(template)
    texts = root.findall(".//w:t", NS)
    if not texts:
        raise RuntimeError("Footer template contains no text run.")
    texts[0].text = label
    for extra in texts[1:]:
        extra.text = ""
    return etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)


def main(source: str, output: str) -> None:
    labels = [
        "یک", "دو", "سه", "چهار", "پنج", "شش", "هفت", "هشت", "نه", "ده", "یازده",
        "دوازده", "سیزده", "چهارده", "پانزده", "شانزده", "هفده", "هجده", "نوزده", "بیست",
        "بیست و یک", "بیست و دو",
    ]
    with ZipFile(Path(source)) as input_zip:
        document = etree.fromstring(input_zip.read("word/document.xml"))
        relationships = etree.fromstring(input_zip.read("word/_rels/document.xml.rels"))
        content_types = etree.fromstring(input_zip.read("[Content_Types].xml"))
        sections = document.findall(".//w:sectPr", NS)
        if len(sections) != 28:
            raise RuntimeError(f"Expected 28 sections, found {len(sections)}.")

        existing_ids = {rel.get("Id") for rel in relationships.findall("pr:Relationship", NS)}
        next_id = 80
        existing_footer_numbers = [
            int(match.group(1))
            for name in input_zip.namelist()
            if (match := re.fullmatch(r"word/footer(\d+)\.xml", name))
        ]
        next_footer_number = max(existing_footer_numbers, default=0) + 1
        footer_parts = []
        template = input_zip.read("word/footer8.xml")
        for index, label in enumerate(labels):
            section = sections[5 + index]
            for existing in section.findall("w:footerReference", NS):
                if existing.get(f"{{{W_NS}}}type") == "default":
                    section.remove(existing)
            while f"rId{next_id}" in existing_ids:
                next_id += 1
            relationship_id = f"rId{next_id}"
            existing_ids.add(relationship_id)
            part_number = next_footer_number + index
            part_name = f"word/footer{part_number}.xml"
            footer_parts.append((part_name, footer_xml(template, label)))
            etree.SubElement(
                relationships,
                f"{{{PR_NS}}}Relationship",
                Id=relationship_id,
                Type=FOOTER_REL,
                Target=f"footer{part_number}.xml",
            )
            reference = etree.Element(f"{{{W_NS}}}footerReference")
            reference.set(f"{{{W_NS}}}type", "default")
            reference.set(f"{{{R_NS}}}id", relationship_id)
            section.insert(0, reference)
            etree.SubElement(
                content_types,
                f"{{{CT_NS}}}Override",
                PartName=f"/word/footer{part_number}.xml",
                ContentType=FOOTER_CT,
            )
            next_id += 1

        replacements = {
            "word/document.xml": etree.tostring(document, xml_declaration=True, encoding="UTF-8", standalone=True),
            "word/_rels/document.xml.rels": etree.tostring(relationships, xml_declaration=True, encoding="UTF-8", standalone=True),
            "[Content_Types].xml": etree.tostring(content_types, xml_declaration=True, encoding="UTF-8", standalone=True),
        }
        with ZipFile(Path(output), "w", ZIP_DEFLATED) as output_zip:
            for item in input_zip.infolist():
                output_zip.writestr(item, replacements.get(item.filename, input_zip.read(item.filename)))
            for part_name, data in footer_parts:
                output_zip.writestr(part_name, data)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("output")
    args = parser.parse_args()
    main(args.source, args.output)
