from pathlib import Path
import json
import re
import zipfile
import xml.etree.ElementTree as ET
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent.parent
results = []
pattern = re.compile(r'\b(chatgpt|claude|codex|copilot|openai|anthropic)\b', re.I)
for p in sorted((ROOT / 'report').glob('*.docx')):
    with zipfile.ZipFile(p) as z:
        broken = z.testzip()
        document = ET.fromstring(z.read('word/document.xml'))
        text = '\n'.join(n.text or '' for n in document.iter() if n.tag.endswith('}t'))
        metadata = z.read('docProps/core.xml').decode('utf-8') if 'docProps/core.xml' in z.namelist() else ''
        comments = z.read('word/comments.xml').decode('utf-8') if 'word/comments.xml' in z.namelist() else ''
        results.append({'path': p.relative_to(ROOT).as_posix(), 'zip_error': broken,
            'text_characters': len(text), 'text_tool_names': pattern.findall(text),
            'metadata_tool_names': pattern.findall(metadata), 'comment_tool_names': pattern.findall(comments),
            'tracked_insertions': len(document.findall('.//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}ins')),
            'tracked_deletions': len(document.findall('.//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}del')),
            'has_comments': bool(comments)})
        if p.name == 'Project_Report.docx':
            (ROOT / 'cleanup_audit' / 'report_text.txt').write_text(text, encoding='utf-8')
for name in ('Project_Report.pdf', 'Format Project.pdf'):
    p = ROOT / 'report' / name
    reader = PdfReader(p)
    pages = [page.extract_text() or '' for page in reader.pages]
    results.append({'path': p.relative_to(ROOT).as_posix(), 'pages': len(pages),
        'text_tool_names': pattern.findall('\n'.join(pages)),
        'metadata_tool_names': pattern.findall(str(reader.metadata)),
        'attachments': list(reader.attachments)})
with zipfile.ZipFile(ROOT / 'dashboardMarket.pbix') as z:
    results.append({'path': 'dashboardMarket.pbix', 'zip_error': z.testzip(), 'entries': z.namelist()})
(ROOT / 'cleanup_audit' / 'document_inspection.json').write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')
for row in results:
    if Path(row['path']).name in ('Project_Report.docx', 'Project_Report.pdf', 'Format Project.pdf', 'dashboardMarket.pbix'):
        print(json.dumps(row, ensure_ascii=False))
