"""Build a clean distribution without changing the source project."""
from pathlib import Path
import csv
import hashlib
import json
import shutil
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / 'Project_Clean'
AUDIT = ROOT / 'cleanup_audit'
if DEST.exists():
    raise SystemExit('Destination already exists; refusing to overwrite it.')

UNUSED = {
    'extract_script.py': 'Malformed, unreferenced one-off template extraction script',
    'test_future_prediction.py': 'Obsolete manual demonstration for the replaced predictive pipeline',
    'validate_results.py': 'Obsolete diagnostic with invalid imports, absolute path and SQL LIMIT syntax',
    'simple_validation.py': 'One-off screenshot comparison with hardcoded expected values and old SQL instance',
    'Website/market/update_feedback.py': 'One-off template patch already applied',
    'Website/market/static/css/market_analysis.css': 'Unreferenced stylesheet superseded by the active v2 theme',
    'Website/market/static/css/market_analysis_v3.css': 'Unreferenced stylesheet',
    'Website/market/static/images/University_of_Kashan_Logo.png': 'Unreferenced image; active templates use kuLogo.png',
    'Website/market/templates/admin/dunnhumby/data_manipulation_enhanced.html': 'Unreferenced alternate template',
}
REPORT_KEEP = {'Project_Report.docx', 'Project_Report.pdf', 'ReportOfDatasets.docx', 'Format Project.pdf', 'manabe-zotero.ris'}

def classify(rel):
    parts = Path(rel).parts
    top = parts[0]
    name = parts[-1]
    if top == '.git': return None, 'Local version-control history retained in original'
    if top == '.venv': return None, 'Machine-specific virtual environment; recreate from requirements'
    if '__pycache__' in parts or name.endswith(('.pyc', '.pyo')): return None, 'Generated Python bytecode'
    if top in {'.claude', '.codex', '.agents', '.cursor'}: return None, 'Local assistant configuration'
    if top == 'tmp': return None, 'Temporary report builds, extracts and QA renders'
    if top == 'output': return None, 'Partial report draft superseded by selected final report'
    if top == 'Claude outputs': return None, 'Unreferenced draft illustrations outside application/report dependencies'
    if top == 'work': return None, 'Older database backup; newer 2026-08-11 backup retained'
    if top == '_db_restore_tmp': return 'database/' + name, 'Latest available database backup retained for restoration'
    if top == 'report':
        if len(parts) == 2 and name in REPORT_KEEP: return rel, 'Selected final report or supporting reference'
        return None, 'Superseded report draft, document patch helper or QA output'
    if name == 'dashboardMarket.before-dashboard.pbix': return None, 'Explicit backup of the current Power BI dashboard'
    if top.startswith('text of conversation '): return None, 'Private working notes not required to run the project'
    if name.endswith(('.backup', '.tmp', '.log')) or name.startswith('~$'): return None, 'Backup, temporary or lock file'
    if rel in UNUSED: return None, UNUSED[rel]
    if name.endswith('_modern.html'): return None, 'Unreferenced alternative template family'
    prefix = 'Website/market/dunnhumby/templates/admin/dunnhumby/'
    if rel.startswith(prefix): return None, 'Shadowed by project-level template with the same loader name'
    if rel.startswith('Website/market/ml_models_cache/time_series/') and not name.startswith('product_revenue_v12_'):
        return None, 'Obsolete forecast artifact; current loader only opens version 12'
    if rel.startswith('Website/market/ml_models_cache/') and len(parts) == 4 and not name.startswith('repurchase_v4_'):
        return None, 'Obsolete predictive artifact; current loader only opens repurchase version 4'
    return rel, 'Project source, active asset, test or trained model'

rows = list(csv.DictReader((AUDIT / 'source_inventory.csv').open(encoding='utf-8')))
manifest = []
DEST.mkdir()
for row in rows:
    target, reason = classify(row['path'])
    manifest.append({**row, 'destination': target or '', 'decision': 'keep' if target else 'exclude', 'reason': reason})
    if target:
        dst = DEST / target
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / row['path'], dst)
        with dst.open('rb') as f:
            actual = hashlib.file_digest(f, 'sha256').hexdigest()
        if actual != row['sha256']:
            raise RuntimeError(f'Copy verification failed: {target}')
with (AUDIT / 'export_manifest.csv').open('w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(manifest[0]))
    writer.writeheader()
    writer.writerows(manifest)
summary = {decision: {'files': sum(r['decision'] == decision for r in manifest),
    'bytes': sum(int(r['bytes']) for r in manifest if r['decision'] == decision)} for decision in ('keep', 'exclude')}
(AUDIT / 'export_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps(summary, indent=2))
