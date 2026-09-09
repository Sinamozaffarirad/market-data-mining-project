from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent.parent
changes_path = ROOT / 'cleanup_audit/changes.json'
changes = json.loads(changes_path.read_text(encoding='utf-8'))
replacements = {
    'AI-Powered Product Recommendations': 'Product Recommendations',
    'AI Recommendations Ready': 'Recommendations Ready',
    'AI-powered product recommendations': 'product recommendations',
    'AI Analysis': 'Recommendation Analysis',
}
for rel in ['Website/market/dunnhumby/views.py', 'Website/market/templates/site/dunnhumby/basket_analysis.html']:
    p = ROOT / 'Project_Clean' / rel
    text = p.read_text(encoding='utf-8')
    for old, new in replacements.items():
        text = text.replace(old, new)
    p.write_text(text, encoding='utf-8')
    changes.append({'path': rel, 'reason': 'Use functional recommendation labels instead of promotional wording; preserve machine-learning behaviour'})
changes_path.write_text(json.dumps(changes, indent=2, ensure_ascii=False), encoding='utf-8')
