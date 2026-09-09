from pathlib import Path
import ast
import csv
import io
import json
import re
import tokenize

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / 'Project_Clean'
AUDIT = ROOT / 'cleanup_audit'
changes = []

def edit(rel, old, new, reason):
    p = DEST / rel
    text = p.read_text(encoding='utf-8-sig')
    if old not in text:
        raise ValueError(f'Text not found: {rel}: {old[:70]}')
    p.write_text(text.replace(old, new), encoding='utf-8')
    changes.append({'path': rel, 'reason': reason})

for p in sorted((DEST / 'Website').rglob('*.py')):
    source = p.read_text(encoding='utf-8-sig')
    lines = source.splitlines(keepends=True)
    removed = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type != tokenize.COMMENT:
            continue
        comment = token.string
        if (re.fullmatch(r'#\s*[\w/]+\.py\s*', comment)
                or re.fullmatch(r'#\s*[-=]{3,}\s*', comment)
                or 'بخش اضافه شده' in comment or 'پایان بخش اضافه شده' in comment
                or comment.strip() in {'# Create your models here.', '# Create your views here.',
                    '# Create your tests here.', '# Register your models here.',
                    '# Setup Django environment', '# Configure Django settings',
                    '# Add the project root to the Python path'}):
            row, col = token.start
            lines[row - 1] = lines[row - 1][:col] + lines[row - 1][token.end[1]:]
            removed.append(row)
    updated = ''.join(lines)
    if updated != source:
        assert ast.dump(ast.parse(updated)) == ast.dump(ast.parse(source)), p
        p.write_text(updated, encoding='utf-8')
        changes.append({'path': p.relative_to(DEST).as_posix(), 'reason': 'Remove redundant file labels and editing/scaffolding comments', 'lines': removed})

edit('Website/market/customers/ml/cf_cache.py',
    'Precomputed, on-disk cache of the collaborative-filtering similarity\nmatrix. dunnhumby.collab_filter.get_cf_recommendations rebuilds the whole\nmatrix from 2.6M transactions and recomputes cosine_similarity on every\nsingle call - that\'s why recommendation pages were taking minutes.\nThis module builds it ONCE (via `python manage.py build_cf_cache`) and\nevery request afterward just loads the cached file into memory.',
    'Persist the household-product and similarity matrices for recommendations.\nBuild the cache with `python manage.py build_cf_cache`; requests load it once\nper process to avoid recalculating similarities for every recommendation.',
    'Document cache lifecycle concisely')
edit('Website/market/dunnhumby/bi_views.py',
    "    # The two tests above answer a question about one pair. These ask whether the\n    # dimension as a whole separates the groups, which is what the professor's\n    # ANOVA / Kruskal-Wallis pairing is for: the same question, one assuming\n    # normality and one not.",
    '    # Compare all groups using parametric ANOVA and rank-based Kruskal-Wallis.',
    'Replace conversational comment with the statistical purpose')
edit('Website/market/dunnhumby/admin.py',
    'Enhanced database manipulation interface with comprehensive CRUD operations',
    'Handle database maintenance and record operations.', 'Use a factual docstring')
edit('Website/market/dunnhumby/admin.py',
    'Generate association rules (enhanced implementation)',
    'Generate association rules from basket co-occurrence.', 'Use a factual docstring')
edit('Website/market/dunnhumby/management/commands/import_transactions.py',
    'default=r"C:\\Local Disk D\\UNI Files\\Final Project\\transaction_data.csv",',
    'required=True,', 'Require an explicit CSV path instead of a stale machine-specific default')
edit('Website/market/dunnhumby/tests.py',
    '{"equal": ZeroCorrectionModel(), "weighted": ZeroCorrectionModel()},',
    '{\n                "models": {lead: {\n                    "equal_product_weight": ZeroCorrectionModel(),\n                    "log_revenue_weight": ZeroCorrectionModel(),\n                } for lead in (1, 2, 3)},\n                "deepest_lead": 3,\n            },',
    'Update regression fixture to the existing per-lead model bundle; preserve forecast assertions')

# The old pages reference absent core templates. Preserve their public URLs and
# authentication requirements, directing them to the maintained analytics pages.
p = DEST / 'Website/market/core/views.py'
p.write_text('''from django.contrib.auth.decorators import login_required
from django.shortcuts import redirect


def home(request):
    return redirect("dunnhumby_site:index")


@login_required
def dashboard(request):
    return redirect("dunnhumby_site:bi_dashboard")


@login_required
def analytics(request):
    return redirect("dunnhumby_site:basket_analysis")


@login_required
def reports(request):
    return redirect("dunnhumby_site:bi_dashboard")
''', encoding='utf-8')
changes.append({'path': 'Website/market/core/views.py', 'reason': 'Preserve legacy routes with redirects to existing pages instead of missing templates'})

# Remove only unused imports in modules containing no implementation.
for rel in ['catalog/views.py', 'catalog/tests.py', 'core/admin.py', 'core/models.py', 'core/tests.py']:
    p = DEST / 'Website/market' / rel
    if all(isinstance(n, (ast.Import, ast.ImportFrom)) for n in ast.parse(p.read_text()).body):
        p.write_text('', encoding='utf-8')
        changes.append({'path': p.relative_to(DEST).as_posix(), 'reason': 'Remove unused scaffold imports; retain Django module'})

for rel in ['Website/market/market/settings.py', 'Website/market/market/urls.py']:
    p = DEST / rel
    source = p.read_text(encoding='utf-8')
    doc = ast.parse(source).body[0]
    if isinstance(doc, ast.Expr) and isinstance(doc.value, ast.Constant) and isinstance(doc.value.value, str):
        description = 'Configuration for the local market analysis application.' if p.name == 'settings.py' else 'Application URL routes.'
        lines = source.splitlines(keepends=True)
        p.write_text('"""' + description + '"""\n' + ''.join(lines[doc.end_lineno:]), encoding='utf-8')
        changes.append({'path': rel, 'reason': 'Replace generic setup tutorial with a module description'})

edit('Website/market/market/settings.py',
    'SECRET_KEY = "django-insecure--7&=%p1gosrs=_zv(+immez89e$w*@x1lb&9rad&s_ym*qur0&"',
    'SECRET_KEY = os.getenv("DJANGO_SECRET_KEY") or get_random_secret_key()',
    'Do not distribute a hardcoded application signing key; allow a stable environment key')
edit('Website/market/market/settings.py', 'from pathlib import Path',
    'from pathlib import Path\n\nfrom django.core.management.utils import get_random_secret_key',
    'Support local secret-key generation')
edit('Website/market/market/settings.py', 'DEBUG = True',
    'DEBUG = os.getenv("DJANGO_DEBUG", "1").lower() in {"1", "true", "yes"}',
    'Make local debug mode configurable')
edit('Website/market/market/settings.py', 'ALLOWED_HOSTS = []',
    'ALLOWED_HOSTS = [host.strip() for host in os.getenv(\n    "DJANGO_ALLOWED_HOSTS", "localhost,127.0.0.1,[::1]"\n).split(",") if host.strip()]', 'Allow explicit hosts while keeping local defaults')
edit('Website/market/market/settings.py', '"NAME": "marketdb",',
    '"NAME": os.getenv("DB_NAME", "marketdb"),', 'Make database name portable')
edit('Website/market/market/settings.py', '"HOST": "localhost",\n        # "HOST": "localhost\\SQLEXPRESS",',
    '"HOST": os.getenv("DB_HOST", "localhost"),', 'Make SQL Server instance portable')
edit('Website/market/market/settings.py', '"driver": "ODBC Driver 17 for SQL Server",',
    '"driver": os.getenv("DB_DRIVER", "ODBC Driver 17 for SQL Server"),', 'Make ODBC driver configurable')
edit('Website/market/market/settings.py',
    '# 3rd-party\n    # \'django_extensions\',\n    # local apps',
    '# Project applications', 'Remove commented-out unused application')

# The standalone importer is an obsolete duplicate: it has an invalid import
# layout and mixes uppercase and lowercase input columns. The management command
# remains the supported entry point.
obsolete = DEST / 'Website/import_transactions.py'
assert obsolete.resolve().is_relative_to(DEST.resolve())
obsolete.unlink()
manifest_path = AUDIT / 'export_manifest.csv'
rows = list(csv.DictReader(manifest_path.open(encoding='utf-8')))
for row in rows:
    if row['path'] == 'Website/import_transactions.py':
        row.update(destination='', decision='exclude', reason='Obsolete duplicate importer with invalid imports and inconsistent column names; management command retained')
with manifest_path.open('w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
(AUDIT / 'changes.json').write_text(json.dumps(changes, indent=2, ensure_ascii=False), encoding='utf-8')
print(f'Applied {len(changes)} documented refinements.')
