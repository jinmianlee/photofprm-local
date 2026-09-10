"""Check the exact installed requirements against PyPI release advisories.

Read-only network requests to pypi.org; does not import any audited package.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import time
import urllib.request
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]


def check(line):
    name, version = line.strip().split('==')
    for attempt in range(4):
        try:
            with urllib.request.urlopen(f'https://pypi.org/pypi/{name}/{version}/json', timeout=25) as response:
                metadata = json.load(response)
            return {'name': name, 'version': version, 'vulnerabilities': metadata.get('vulnerabilities', [])}
        except Exception as error:
            failure = str(error)
            time.sleep(.4)
    return {'name': name, 'version': version, 'error': failure}


if __name__ == '__main__':
    lines = [line for line in (ROOT/'requirements-lock.txt').read_text('utf-8-sig').splitlines() if '==' in line]
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(check, lines))
    report = {'checked_at': datetime.now(timezone.utc).isoformat(), 'source': 'PyPI release JSON',
              'packages': results, 'unknown': [r['name'] for r in results if r.get('error')],
              'affected': [r for r in results if r.get('vulnerabilities')]}
    (ROOT/'data'/'python-advisory-audit.json').write_text(json.dumps(report,indent=2), 'utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='packages'}, indent=2))
