"""Exercise the actual loopback upload, inference and export endpoints."""
import json
import time
from pathlib import Path
import httpx
import argparse

ROOT = Path(__file__).resolve().parents[1]
options = {'size_mm': 95, 'colors': 4, 'pitch_mm': .8, 'min_feature_mm': .8,
           'photo_pitch_deg': 25, 'photo_yaw_deg': 7}
started = time.monotonic()
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--existing', action='store_true', help='Verify the recorded job after a server restart')
parser.add_argument('--photo', type=Path, help='Local reference photo; required for a new validation run')
args = parser.parse_args()
if not args.existing and (args.photo is None or not args.photo.is_file()):
    parser.error('Provide an existing local image with --photo, or use --existing.')
photo = args.photo
with httpx.Client(base_url='http://127.0.0.1:8765', trust_env=False, timeout=60,
                  headers={'X-PhotoForm-Client': 'local'}) as client:
    assert client.get('/api/system').json()['single_photo']['ready']
    target = ROOT / 'data/cat-app-validation.json'
    if args.existing:
        record = json.loads(target.read_text('utf-8'))
        job = record['id']
        record['verification_resumed_after_interruption'] = True
    else:
        with photo.open('rb') as stream:
            response = client.post('/api/upload', data={'kind': 'single', 'options': json.dumps(options)},
                                   files={'files': ('cat.png', stream, 'image/png')})
        response.raise_for_status()
        job = response.json()['id']
        record = {'id': job, 'input': str(photo), 'options': options, 'state': 'running'}
    target.write_text(json.dumps(record, indent=2, ensure_ascii=False), 'utf-8')
    print('JOB', job, flush=True)
    previous = None
    while time.monotonic()-started < 1800:
        response = client.get(f'/api/jobs/{job}')
        response.raise_for_status()
        state = response.json()
        message = (state['state'], state['progress'], state['message'])
        if message != previous:
            print(message, flush=True)
            previous = message
        if state['state'] in ('failed', 'cancelled'):
            record.update(state=state['state'], message=state['message'])
            target.write_text(json.dumps(record, indent=2, ensure_ascii=False), 'utf-8')
            raise RuntimeError(state['message'])
        if state['state'] == 'complete':
            for name in ('colored.3mf', 'combined.stl', 'print_bundle.zip', 'preview.glb'):
                download = client.get(f'/api/jobs/{job}/files/{name}')
                download.raise_for_status()
                assert len(download.content) > 1000
            record.update(state='complete', verification_elapsed_seconds=round(time.monotonic()-started, 2),
                          downloads_verified=True, report=state['report'])
            target.write_text(json.dumps(record, indent=2, ensure_ascii=False), 'utf-8')
            print('LOCAL_CAT_FLOW_OK', job, record['verification_elapsed_seconds'], flush=True)
            break
        time.sleep(5)
    else:
        raise TimeoutError('Cat flow did not complete within 30 minutes')
