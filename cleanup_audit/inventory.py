from pathlib import Path
import ast
import csv
import hashlib
import io
import json
import os
import re
import tokenize
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
AUDIT = ROOT / 'cleanup_audit'
SKIP = {'cleanup_audit', 'Project_Clean'}
TEXT = {'.py', '.html', '.css', '.js', '.sql', '.md', '.txt', '.json', '.ps1', '.ris'}
rows = []
reviews = []
for base, dirs, files in os.walk(ROOT, followlinks=False):
    dirs[:] = sorted(d for d in dirs if not (Path(base) == ROOT and d in SKIP))
    for name in sorted(files):
        p = Path(base) / name
        rel = p.relative_to(ROOT).as_posix()
        row = {'path': rel, 'bytes': p.stat().st_size, 'sha256': ''}
        own = rel.split('/')[0] not in {'.git', '.venv'} and '__pycache__' not in p.parts
        if own:
            with p.open('rb') as f:
                row['sha256'] = hashlib.file_digest(f, 'sha256').hexdigest()
        rows.append(row)
        if own and p.suffix.lower() in TEXT and p.stat().st_size < 5_000_000:
            source = p.read_text(encoding='utf-8-sig', errors='replace')
            review = {'path': rel, 'lines': len(source.splitlines()), 'decode_replacements': source.count('\ufffd')}
            review['tool_name_lines'] = [i for i, line in enumerate(source.splitlines(), 1)
                if re.search(r'\b(chatgpt|claude|codex|copilot|openai|anthropic)\b', line, re.I)]
            if p.suffix == '.py':
                try:
                    tree = ast.parse(source)
                    review['syntax'] = 'pass'
                    review['definitions'] = [{'name': n.name, 'line': n.lineno, 'lines': n.end_lineno-n.lineno+1}
                        for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
                    review['imports'] = sorted({n.module or '' for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} |
                        {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names})
                    review['bare_except'] = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.ExceptHandler) and n.type is None]
                    review['comments'] = [{'line': t.start[0], 'text': t.string} for t in tokenize.generate_tokens(io.StringIO(source).readline) if t.type == tokenize.COMMENT]
                except (SyntaxError, tokenize.TokenError) as exc:
                    review['syntax'] = str(exc)
            reviews.append(review)
with (AUDIT / 'source_inventory.csv').open('w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['path', 'bytes', 'sha256'])
    writer.writeheader()
    writer.writerows(rows)
(AUDIT / 'source_review.json').write_text(json.dumps(reviews, indent=2, ensure_ascii=False), encoding='utf-8')
print(json.dumps({'files': len(rows), 'bytes': sum(r['bytes'] for r in rows), 'text_files_reviewed': len(reviews),
    'python_files': sum(r['path'].endswith('.py') for r in reviews),
    'syntax_errors': [{'path': r['path'], 'error': r['syntax']} for r in reviews if r.get('syntax', 'pass') != 'pass'],
    'top_level': dict(Counter(r['path'].split('/')[0] for r in rows))}, indent=2))
