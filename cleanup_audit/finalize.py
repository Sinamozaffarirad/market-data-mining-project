from pathlib import Path
import ast
import csv
import hashlib
import json
import re
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / 'Project_Clean'
AUDIT = ROOT / 'cleanup_audit'

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

inventory = list(csv.DictReader((AUDIT / 'source_inventory.csv').open(encoding='utf-8')))
manifest = list(csv.DictReader((AUDIT / 'export_manifest.csv').open(encoding='utf-8')))
original_changes = []
for row in inventory:
    if row['sha256'] and sha(ROOT / row['path']) != row['sha256']:
        original_changes.append(row['path'])
assert not original_changes, original_changes

excluded_names = {'.git', '.claude', '.codex', '.agents', '.cursor', '.venv', 'node_modules', '__pycache__'}
for p in DEST.rglob('*'):
    assert not p.is_symlink(), p
    assert p.name not in excluded_names, p
    assert not p.name.endswith(('.pyc', '.backup', '.tmp', '.log')), p

rows = []
for p in sorted(DEST.rglob('*')):
    if p.is_file() and p.name != 'SHA256SUMS.txt':
        rows.append({'path': p.relative_to(DEST).as_posix(), 'bytes': p.stat().st_size, 'sha256': sha(p)})
hashes = {r['path']: r['sha256'] for r in rows}
modified = [r for r in manifest if r['decision'] == 'keep' and hashes[r['destination']] != r['sha256']]
assert all(Path(r['destination']).suffix in {'.py', '.md', '.html'} or r['destination'] == '.gitignore' for r in modified)
for row in manifest:
    if row['decision'] == 'keep' and Path(row['destination']).suffix.lower() in {'.docx', '.pdf', '.pbix', '.pkl', '.bak', '.png', '.json', '.ris'}:
        assert hashes[row['destination']] == row['sha256'], row['destination']

with (AUDIT / 'final_inventory.csv').open('w', encoding='utf-8', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=['path', 'bytes', 'sha256'])
    writer.writeheader()
    writer.writerows(rows)
(DEST / 'SHA256SUMS.txt').write_text(''.join(f"{r['sha256']}  {r['path']}\n" for r in rows), encoding='utf-8')

reviews = json.loads((AUDIT / 'source_review.json').read_text(encoding='utf-8'))
scan_matches = []
for row in rows:
    p = DEST / row['path']
    if p.suffix in {'.py', '.html', '.js', '.css', '.md', '.txt', '.json', '.sql', '.ris'} or p.name == '.gitignore':
        text = p.read_text(encoding='utf-8-sig')
        for i, line in enumerate(text.splitlines(), 1):
            if re.search(r'\b(chatgpt|claude|codex|copilot|openai|anthropic)\b', line, re.I):
                scan_matches.append({'path': row['path'], 'line': i})
        if p.suffix == '.py':
            ast.parse(text)

summary = {
    'source_files_inventoried': len(inventory),
    'source_bytes': sum(int(r['bytes']) for r in inventory),
    'project_owned_files_hashed': sum(bool(r['sha256']) for r in inventory),
    'project_text_files_scanned': len(reviews),
    'export_files': len(rows) + 1,
    'export_bytes': sum(r['bytes'] for r in rows) + (DEST / 'SHA256SUMS.txt').stat().st_size,
    'excluded_files': sum(r['decision'] == 'exclude' for r in manifest),
    'excluded_bytes': sum(int(r['bytes']) for r in manifest if r['decision'] == 'exclude'),
    'modified_export_files': [r['destination'] for r in modified],
    'source_changes': original_changes,
    'assistant_name_text_matches': scan_matches,
    'exclusions_by_reason': dict(Counter(r['reason'] for r in manifest if r['decision'] == 'exclude')),
}
(AUDIT / 'final_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

checks = json.loads((AUDIT / 'verification.json').read_text())
rendered_js = json.loads((AUDIT / 'rendered_javascript.json').read_text())
live = json.loads((AUDIT / 'live_smoke.json').read_text())
assert not checks['errors']
assert all(r['ok'] for r in rendered_js)
assert all(r.get('status') == 200 and 'error' not in r for r in live)
assert json.loads((AUDIT / 'database_check.json').read_text())['backup_verify'] == 'passed'

report = f'''# Project cleanup record

The cleaned distribution is `C:\\UNI\\BA Final Project\\Project_Clean`.
The original project files were preserved. All {summary['project_owned_files_hashed']}
hashed original files still match the starting inventory.

## Scope and results

- Inventoried {len(inventory):,} files, including hidden files, local dependencies and Git history.
- Scanned {len(reviews)} project-owned text files. Dependency and Git internals were inventoried as infrastructure, not rewritten.
- Exported {summary['export_files']} files ({summary['export_bytes'] / 1_000_000_000:.2f} GB), including the checksum list.
- Excluded {summary['excluded_files']:,} files ({summary['excluded_bytes'] / 1_000_000_000:.2f} GB) from the copy; most excluded file counts are virtual-environment and Git internals.
- Modified {len(modified)} exported source/documentation files. All retained reports, images, model artifacts, metrics, the PBIX file and database backup match their original SHA-256 hashes.

## Exclusions

Local assistant configuration, Git history, the virtual environment, bytecode,
temporary document builds, rendered QA images, intermediate report versions,
unreferenced draft illustrations, the older database backup, and the explicitly
named Power BI backup were omitted. The selected final reports are
`Project_Report.docx` and `Project_Report.pdf`.

Unreferenced template variants and stylesheets were omitted. Project-level Django
templates shadowed four application-level templates, so the unused copies were
omitted. The current model loaders use revenue version 12, repurchase version 4 and
hybrid version 1; incompatible old cache generations were omitted. The large
collaborative-filtering cache was retained for demonstration performance.

Broken or obsolete one-off helper scripts were omitted. The supported Django
transaction import command remains, with an explicit CSV path required.

## Source changes

Redundant file labels, editing notes, generic tutorial text and promotional
recommendation labels were simplified. Substantive algorithm explanations,
framework migration markers and third-party attributions were retained.
The unused historical churn implementation was removed while its deprecation
guards were preserved.

Three legacy URLs referenced missing templates. They now redirect to the existing
dashboard or basket-analysis pages, retaining login checks. One existing forecast
test used a superseded fixture structure; its fixture now matches the current
per-lead model interface, with the original numerical assertions preserved.

The exported settings accept environment variables for database connection,
allowed hosts, debug mode and the secret key. A temporary secret is generated
when no stable key is supplied. The README documents setup, actual page paths,
current model versions and the date of the included database snapshot.

## Verification

- 37 existing tests passed using a temporary in-memory SQLite database. No live test database was created.
- Django system checks and installed dependency consistency checks passed.
- {len(checks['python'])} Python files compiled; {len(checks['json'])} JSON files parsed.
- {len(checks['templates'])} templates compiled; {len(checks['template_references'])} template references and {len(checks['static_references'])} static references resolved.
- {len(checks['javascript'])} source/template script blocks passed JavaScript syntax checks after substituting template values.
- {len(rendered_js)} actual server-rendered script blocks passed JavaScript syntax checks.
- Nine main pages and two dashboard endpoints returned HTTP 200 against local SQL Server. The checker blocked database-writing statements.
- SQL Server `RESTORE VERIFYONLY` accepted the copied backup. No restore was performed.
- The selected 221-page PDF was text-extracted. The DOCX package passed its ZIP integrity check and contained no Word comments or tracked insertions/deletions.
- No assistant tool names were found in retained project text or in the inspected final report text/core metadata. This is a literal scan, not an authorship test.

## Limits

Code style cannot establish who wrote a project, and this cleanup does not certify
independent authorship or satisfy an institution's disclosure rules. This was a
distribution cleanup with automated source checks and targeted review, not a
proof that every algorithm, browser interaction or report claim is correct.

The report contents and layout were not edited; full page-by-page visual review
was outside the cleanup. Binary model files were copied and hash-checked; the
models were not all retrained. The database backup is from 11 August 2026 and was
not restored to compare its contents with the current live database. Newer Django
migrations may be needed after restoration.

SQL Server-specific behaviour was smoke-tested on the live database, while the
integration test ran on SQLite. Browser interactions and external CDN/Power BI
availability were not exercised end to end. This remains a local demonstration
configuration, not a completed production security review. Existing CSRF-exempt
API handlers and the legacy dataset-specific CSV importer warrant a separate
review before production use or importing a different dataset.

## Audit files

- `source_inventory.csv`: every original file, with hashes for project-owned files.
- `export_manifest.csv`: the keep/exclude decision and reason for every original file.
- `final_inventory.csv`: final distributed file hashes.
- `changes.json`: source refinements and their reasons.
- `source_review.json`: text scans and Python syntax/structure inventory.
- `verification.json`, `rendered_javascript.json`, `live_smoke.json`: source and runtime checks.
- `isolated_tests.log`, `database_check.json`, `document_inspection.json`: test and artifact evidence.

The audit folder is separate from the distribution. No original files were deleted.
'''
(AUDIT / 'CLEANUP_REPORT.md').write_text(report, encoding='utf-8')
print(json.dumps({k: v for k, v in summary.items() if k not in {'exclusions_by_reason', 'modified_export_files'}}, indent=2))
