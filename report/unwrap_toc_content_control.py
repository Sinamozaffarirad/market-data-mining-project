from copy import deepcopy
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from lxml import etree


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": W_NS}


def main(source: str, output: str) -> None:
    source_path = Path(source)
    output_path = Path(output)
    with ZipFile(source_path) as source_zip:
        xml = source_zip.read("word/document.xml")
        root = etree.fromstring(xml)
        body = root.find("w:body", NS)
        toc_sdt = next(
            child
            for child in body
            if child.tag == f"{{{W_NS}}}sdt" and "فهرست مطالب" in "".join(child.itertext())
        )
        content = toc_sdt.find("w:sdtContent", NS)
        position = body.index(toc_sdt)
        for child in list(content):
            body.insert(position, deepcopy(child))
            position += 1
        body.remove(toc_sdt)
        updated_xml = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)

        with ZipFile(output_path, "w", ZIP_DEFLATED) as output_zip:
            for item in source_zip.infolist():
                output_zip.writestr(item, updated_xml if item.filename == "word/document.xml" else source_zip.read(item.filename))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("output")
    args = parser.parse_args()
    main(args.source, args.output)
