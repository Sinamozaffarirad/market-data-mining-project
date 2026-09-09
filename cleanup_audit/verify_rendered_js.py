from pathlib import Path
from html.parser import HTMLParser
import json
import subprocess
import sys

AUDIT = Path(__file__).resolve().parent
NODE = r'C:\Users\sinam\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe'

class Parser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=False)
        self.active = False
        self.scripts = []
    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == 'script' and 'src' not in attributes and attributes.get('type', '') in ('', 'text/javascript', 'module'):
            self.active = True
            self.scripts.append('')
    def handle_endtag(self, tag):
        if tag == 'script': self.active = False
    def handle_data(self, text):
        if self.active: self.scripts[-1] += text

results = []
for p in sorted((AUDIT / 'rendered_pages').glob('*.html')):
    parser = Parser()
    parser.feed(p.read_text(encoding='utf-8'))
    for index, script in enumerate(parser.scripts):
        if not script.strip(): continue
        output = AUDIT / 'javascript' / f'rendered_{p.stem}_{index}.js'
        output.write_text(script, encoding='utf-8')
        run = subprocess.run([NODE, '--check', str(output)], capture_output=True, text=True)
        results.append({'page': p.name, 'script': index, 'ok': run.returncode == 0, 'error': run.stderr})
(AUDIT / 'rendered_javascript.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
errors = [r for r in results if not r['ok']]
print(json.dumps({'scripts': len(results), 'errors': errors}, indent=2))
sys.exit(bool(errors))
