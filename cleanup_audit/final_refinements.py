from pathlib import Path
import ast
import json

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / 'Project_Clean'
changes_path = ROOT / 'cleanup_audit/changes.json'
changes = json.loads(changes_path.read_text(encoding='utf-8'))

def replace(rel, old, new, reason):
    p = DEST / rel
    source = p.read_text(encoding='utf-8')
    assert old in source, rel
    result = source.replace(old, new)
    if p.suffix == '.py' and reason.startswith('Comment'):
        assert ast.dump(ast.parse(source)) == ast.dump(ast.parse(result))
    p.write_text(result, encoding='utf-8')
    changes.append({'path': rel, 'reason': reason})

rel = 'Website/market/dunnhumby/analytics.py'
p = DEST / rel
source = p.read_text(encoding='utf-8')
marker = '# Kept only as historical reference while this project transitions to the'
assert marker in source
before = ast.parse(source)
source = source[:source.index(marker)].rstrip() + '\n'
after = ast.parse(source)
assert len(before.body) == len(after.body) + 1
assert before.body[-1].name == '_legacy_build_churn_feature_set'
p.write_text(source, encoding='utf-8')
changes.append({'path': rel, 'reason': 'Remove unreferenced historical churn implementation; retain explicit deprecation guards and tests'})

replace('Website/market/customers/urls.py',
    "    # The 'name' has been changed from 'customer_search' to 'search'\n", '',
    'Comment cleanup: remove an obsolete editing note')
replace('Website/market/dunnhumby/views.py',
    '    # ۲. دریافت داده‌ها از دیتابیس (بدون تغییر)',
    '    # دریافت خلاصه هر بخش مشتریان',
    'Comment cleanup: describe the query instead of editing history')
replace('Website/market/dunnhumby/views.py',
    '"segments": segments_list,  # <-- از لیست مرتب‌شده جدید استفاده می‌کنیم',
    '"segments": segments_list,',
    'Comment cleanup: remove an obsolete editing note')
replace('Website/market/dunnhumby/templatetags/market_icons.py',
    '''    """A tab icon matching the page's icon in the navigation bar.

    Browsers show a generic globe for a site with no icon, so every tab looked
    alike once a few were open. An emoji drawn into an inline SVG needs no file
    and no extra request.
    """''',
    '    """Return an inline SVG favicon for the page\'s navigation emoji."""',
    'Use a concise favicon docstring')
for old, new in [('/* Additional modern enhancements */', '/* Table transitions */'),
                 ('// Enhanced Modal Helper', '// Shared modal helper'),
                 ('<!-- Enhanced Modal -->', '<!-- Shared modal -->')]:
    replace('Website/market/templates/site/base.html', old, new, 'Comment cleanup: describe the component directly')
for rel, reason in [('README.md', 'Replace stale feature claims, paths and version information with verified local setup instructions'),
                    ('.gitignore', 'Reduce generic rules to this project and omit local assistant-specific configuration')]:
    changes.append({'path': rel, 'reason': reason})
changes_path.write_text(json.dumps(changes, indent=2, ensure_ascii=False), encoding='utf-8')
print(f'{len(changes)} change records in total.')
