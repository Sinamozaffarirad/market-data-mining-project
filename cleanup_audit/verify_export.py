from pathlib import Path
import ast
import importlib
import json
import os
import re
import subprocess
import sys
from html.parser import HTMLParser

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / 'Project_Clean'
PROJECT = DEST / 'Website/market'
AUDIT = ROOT / 'cleanup_audit'
sys.dont_write_bytecode = True
sys.path.insert(0, str(PROJECT))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'market.settings')
import django
django.setup()
from django.template import engines
from django.template.loader import get_template
from django.contrib.staticfiles import finders
from django.test import RequestFactory
from django.urls import resolve, reverse
from types import SimpleNamespace

result = {'python': [], 'json': [], 'templates': [], 'template_references': [],
          'static_references': [], 'javascript': [], 'legacy_redirects': [], 'errors': []}
for p in sorted(PROJECT.rglob('*.py')):
    rel = p.relative_to(DEST).as_posix()
    try:
        tree = ast.parse(p.read_text(encoding='utf-8-sig'), filename=rel)
        compile(tree, rel, 'exec')
        result['python'].append(rel)
    except Exception as exc:
        result['errors'].append({'file': rel, 'error': str(exc)})
for p in sorted(PROJECT.rglob('*.json')):
    try:
        json.loads(p.read_text(encoding='utf-8-sig'))
        result['json'].append(p.relative_to(DEST).as_posix())
    except Exception as exc:
        result['errors'].append({'file': str(p), 'error': str(exc)})

for p in sorted(PROJECT.rglob('*.html')):
    rel = p.relative_to(DEST).as_posix()
    source = p.read_text(encoding='utf-8-sig')
    try:
        engines['django'].from_string(source)
        result['templates'].append(rel)
    except Exception as exc:
        result['errors'].append({'file': rel, 'error': str(exc)})
    for match in re.finditer(r'{%\s*(?:extends|include)\s+[\'"]([^\'"]+)', source):
        name = match.group(1)
        try:
            get_template(name)
            result['template_references'].append({'file': rel, 'template': name})
        except Exception as exc:
            result['errors'].append({'file': rel, 'template': name, 'error': str(exc)})
    for match in re.finditer(r'{%\s*static\s+[\'"]([^\'"]+)', source):
        name = match.group(1)
        found = finders.find(name)
        result['static_references'].append({'file': rel, 'asset': name, 'found': bool(found)})
        if not found:
            result['errors'].append({'file': rel, 'missing_static': name})

for p in sorted(PROJECT.rglob('*.py')):
    tree = ast.parse(p.read_text(encoding='utf-8-sig'))
    for n in ast.walk(tree):
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value.endswith('.html') and '\n' not in n.value:
            try:
                get_template(n.value)
                result['template_references'].append({'file': p.relative_to(DEST).as_posix(), 'template': n.value})
            except Exception as exc:
                result['errors'].append({'file': p.relative_to(DEST).as_posix(), 'template': n.value, 'error': str(exc)})

node = r'C:\Users\sinam\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe'
scripts = AUDIT / 'javascript'
scripts.mkdir(exist_ok=True)

class Scripts(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=False)
        self.capture = False
        self.buffers = []
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'script' and 'src' not in attrs and attrs.get('type', '') in ('', 'text/javascript', 'module'):
            self.capture = True
            self.buffers.append('')
    def handle_endtag(self, tag):
        if tag == 'script': self.capture = False
    def handle_data(self, data):
        if self.capture: self.buffers[-1] += data

for p in sorted(PROJECT.rglob('*')):
    if not p.is_file() or p.suffix not in ('.html', '.js'): continue
    source = p.read_text(encoding='utf-8-sig')
    if p.suffix == '.html':
        # Syntax-check template JS after replacing server-side values. This does
        # not substitute for executing every browser interaction.
        from dunnhumby.templatetags.market_icons import department_styles_json
        source = re.sub(r'{%\s*department_styles_json\s*%}', lambda m: str(department_styles_json()), source)
        source = re.sub(r'{%\s*commodity_departments_json\s*%}', '{}', source)
        source = re.sub(r'{%.*?%}', '', source, flags=re.S)
        source = re.sub(r'{{.*?}}', '0', source, flags=re.S)
        parser = Scripts()
        parser.feed(source)
        chunks = parser.buffers
    else:
        chunks = [source]
    for i, chunk in enumerate(chunks):
        if not chunk.strip(): continue
        temp = scripts / (p.relative_to(PROJECT).as_posix().replace('/', '_') + f'_{i}.js')
        temp.write_text(chunk, encoding='utf-8')
        check = subprocess.run([node, '--check', str(temp)], capture_output=True, text=True)
        record = {'file': p.relative_to(DEST).as_posix(), 'script': i, 'ok': check.returncode == 0}
        result['javascript'].append(record)
        if check.returncode:
            result['errors'].append({**record, 'error': check.stderr})

factory = RequestFactory()
for path, name in [('/dashboard/', 'dunnhumby_site:bi_dashboard'),
                   ('/analytics/', 'dunnhumby_site:basket_analysis'),
                   ('/reports/', 'dunnhumby_site:bi_dashboard')]:
    request = factory.get(path)
    request.user = SimpleNamespace(is_authenticated=True)
    response = resolve(path).func(request)
    ok = response.status_code == 302 and response.url == reverse(name)
    result['legacy_redirects'].append({'path': path, 'target': response.url, 'ok': ok})
    if not ok: result['errors'].append({'legacy_route': path})

(AUDIT / 'verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps({k: (len(v) if k != 'errors' else v) for k, v in result.items()}, indent=2))
sys.exit(bool(result['errors']))
