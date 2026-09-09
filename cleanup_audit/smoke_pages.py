from pathlib import Path
import json
import os
import re
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'Project_Clean/Website/market'))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'market.settings')
import django
django.setup()
from django.conf import settings
settings.ALLOWED_HOSTS += ['testserver']
from django.db import connection
from django.test import RequestFactory
from django.urls import resolve
from django.contrib.messages.storage.fallback import FallbackStorage

def read_only(execute, sql, params, many, context):
    statement = re.sub(r'--[^\n]*|/\*.*?\*/', '', sql, flags=re.S).strip().lstrip(';').lstrip()
    if not re.match(r'^(SELECT|WITH)\b', statement, re.I):
        raise RuntimeError('Blocked non-read query during smoke check: ' + statement[:80])
    if re.search(r'\b(INSERT|UPDATE|DELETE|MERGE|DROP|ALTER|TRUNCATE|EXEC|CREATE)\b', statement, re.I):
        raise RuntimeError('Blocked potential mutation during smoke check')
    return execute(sql, params, many, context)

factory = RequestFactory()
user = SimpleNamespace(is_authenticated=True, is_staff=True, is_superuser=True,
                       username='local-review', pk=0, id=0, get_full_name=lambda: '', get_username=lambda: 'local-review',
                       has_perm=lambda *a: True, has_module_perms=lambda *a: True)
paths = ['/analysis/', '/analysis/basket-analysis/', '/analysis/association-rules/',
         '/analysis/customer-segments/', '/analysis/bi-dashboard/', '/analysis/data-management/',
         '/customers/search/', '/analysis/product-recommender/', '/analysis/customer-retention/',
         '/analysis/api/bi/kpis/', '/analysis/api/bi/top-products/']
if len(sys.argv) > 1:
    paths = sys.argv[1:]
    results = [r for r in json.loads((ROOT / 'cleanup_audit/live_smoke.json').read_text()) if r['path'] not in paths]
else:
    results = []
for path in paths:
    start = time.monotonic()
    record = {'path': path}
    try:
        request = factory.get(path)
        request.user = user
        request.session = {}
        request._messages = FallbackStorage(request)
        match = resolve(path)
        request.resolver_match = match
        with connection.execute_wrapper(read_only):
            response = match.func(request, *match.args, **match.kwargs)
            if hasattr(response, 'render'):
                response.render()
        record.update(status=response.status_code, bytes=len(response.content))
        if response.get('Content-Type', '').startswith('text/html'):
            filename = path.strip('/').replace('/', '_') + '.html'
            folder = ROOT / 'cleanup_audit/rendered_pages'
            folder.mkdir(exist_ok=True)
            (folder / filename).write_bytes(response.content)
        elif response.status_code >= 400:
            record['error'] = response.content.decode('utf-8', errors='replace')[:500]
    except Exception as exc:
        record['error'] = str(exc)
    record['seconds'] = round(time.monotonic() - start, 2)
    results.append(record)
    print(json.dumps(record), flush=True)
(ROOT / 'cleanup_audit/live_smoke.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
connection.close()
sys.exit(any(r.get('status') != 200 or 'error' in r for r in results))
