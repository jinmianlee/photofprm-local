"""Transparent foreground launcher. Close with Ctrl+C in this console."""
from pathlib import Path
import os
import sys
import threading
import urllib.request
import json
import webbrowser


def main():
    root = Path(__file__).resolve().parent
    os.chdir(root)
    sys.stdout.reconfigure(encoding='utf-8')
    print('PhotoForm Local - http://127.0.0.1:8765', flush=True)
    print('The local server runs in this console. Press Ctrl+C to stop.', flush=True)
    if not (root / 'dist' / 'index.html').is_file():
        raise SystemExit('Build missing. Run npm install and npm run build first.')
    url = 'http://127.0.0.1:8765'
    try:
        local_http = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with local_http.open(url+'/api/system', timeout=1) as response:
            existing = json.load(response)
        if existing.get('app') == 'PhotoForm Local':
            if '--no-browser' not in sys.argv:
                webbrowser.open(url)
            return
    except (OSError, ValueError):
        pass
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    os.environ['PYTHONUTF8'] = '1'
    import uvicorn
    if '--no-browser' not in sys.argv:
        timer = threading.Timer(1.5, lambda: webbrowser.open(url))
        timer.daemon = True
        timer.start()
    uvicorn.run('backend.app:app', host='127.0.0.1', port=8765, log_level='info')


if __name__ == '__main__':
    main()
