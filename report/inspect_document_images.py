from pathlib import Path
from zipfile import ZipFile

from lxml import etree


NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}
RID = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed"


def main(path: str) -> None:
    with ZipFile(Path(path)) as archive:
        root = etree.fromstring(archive.read("word/document.xml"))
        paragraphs = root.xpath(".//w:p", namespaces=NS)
        for index, paragraph in enumerate(paragraphs, 1):
            if not paragraph.xpath(".//w:drawing", namespaces=NS):
                continue
            text = "".join(paragraph.itertext()).strip().replace("\n", " ")
            props = [
                (prop.get("name"), prop.get("descr"))
                for prop in paragraph.xpath(".//wp:docPr", namespaces=NS)
            ]
            rels = [blip.get(RID) for blip in paragraph.xpath(".//a:blip", namespaces=NS)]
            before = " ".join("".join(item.itertext()).strip() for item in paragraphs[max(0, index - 3):index - 1])
            after = " ".join("".join(item.itertext()).strip() for item in paragraphs[index:index + 2])
            print(index, "|", props, "| BEFORE:", repr(before[:220]), "| AFTER:", repr(after[:300]), "|", rels)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    main(parser.parse_args().path)
