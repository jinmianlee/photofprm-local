"""Transparent foreground launcher. Close with Ctrl+C in this console."""
from pathlib import Path
import os
import sys
import threading
import urllib.request
import json
import webbrowser
import argparse
import socket
import secrets


def main():
    parser = argparse.ArgumentParser(description='Run PhotoForm with local model files.')
    parser.add_argument('--lan', action='store_true', help='Allow paired browsers on the local network')
    parser.add_argument('--allow-host', action='append', default=[], help='Additional local IP or hostname')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    os.chdir(root)
    sys.stdout.reconfigure(encoding='utf-8')
    url = f'http://127.0.0.1:{args.port}'
    print(f'PhotoForm Local - {url}', flush=True)
    print('The local server runs in this console. Press Ctrl+C to stop.', flush=True)
    if not (root / 'dist' / 'index.html').is_file():
        raise SystemExit('Build missing. Run npm install and npm run build first.')
    try:
        local_http = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with local_http.open(url+'/api/system', timeout=1) as response:
            existing = json.load(response)
        if existing.get('app') == 'PhotoForm Local':
            if args.lan and existing.get('network_mode') != 'lan':
                raise SystemExit('A local-only server is already running. Stop it with Ctrl+C, then start LAN mode.')
            print('PhotoForm is already running; opening the existing service.', flush=True)
            if not args.no_browser:
                webbrowser.open(url)
            return
    except (OSError, ValueError):
        pass
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    os.environ['PYTHONUTF8'] = '1'
    if args.lan:
        hosts = set(args.allow_host)
        hosts.update(item[4][0] for item in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET))
        hosts.discard('127.0.0.1')
        key_path = root / 'data' / 'lan-key.txt'
        key_path.parent.mkdir(exist_ok=True)
        if not key_path.exists():
            key_path.write_text(secrets.token_urlsafe(24), encoding='ascii')
        key = key_path.read_text('ascii').strip()
        if len(key) < 24:
            raise SystemExit('Invalid data/lan-key.txt. Remove that file and restart to create a new access code.')
        os.environ.update(PHOTOFORM_LAN='1', PHOTOFORM_ALLOWED_HOSTS=','.join(sorted(hosts)), PHOTOFORM_LAN_KEY=key)
        print('LAN addresses: '+', '.join(f'http://{host}:{args.port}' for host in sorted(hosts)), flush=True)
        print('LAN access code: '+key, flush=True)
        print('Use a trusted private network. Keep this console running. Do not forward this port to the Internet.', flush=True)
    else:
        os.environ.pop('PHOTOFORM_LAN', None)
        os.environ.pop('PHOTOFORM_LAN_KEY', None)
    import uvicorn
    if not args.no_browser:
        timer = threading.Timer(1.5, lambda: webbrowser.open(url))
        timer.daemon = True
        timer.start()
    uvicorn.run('backend.app:app', host='0.0.0.0' if args.lan else '127.0.0.1', port=args.port, log_level='info')


if __name__ == '__main__':
    main()
