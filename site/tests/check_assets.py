"""Verify public HTML references against the source tree or Pages artifact."""
from html.parser import HTMLParser
from pathlib import Path
import sys
from urllib.parse import unquote, urlsplit


class References(HTMLParser):
    def __init__(self):
        super().__init__()
        self.urls = []

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key in ('src', 'href') and value:
                self.urls.append(value)


root = Path(sys.argv[1] if len(sys.argv) > 1 else '.').resolve()
failures = []
count = 0
for page in ['index.html', 'builder.html', 'hardware.html']:
    source = (root / page).read_text(encoding='utf-8')
    parser = References()
    parser.feed(source)
    if 'sync-tank-banner' in source:
        failures.append(f'{page}: removed monochrome banner referenced')
    for ref in parser.urls:
        parsed = urlsplit(ref)
        if parsed.scheme or parsed.netloc or not parsed.path:
            continue
        path = (root / unquote(parsed.path)).resolve()
        count += 1
        if not path.is_relative_to(root) or not path.is_file():
            failures.append(f'{page}: missing or unsafe local reference {ref}')
for required in ['sync/static/vendor/three.module.js', 'sync/static/vendor/OrbitControls.js', 'site/THIRD_PARTY.md']:
    if not (root / required).is_file():
        failures.append(f'Missing 3D dependency: {required}')
if failures:
    raise SystemExit('\n'.join(failures))
print(f'PASS {count} local page references and vendored 3D dependencies')
