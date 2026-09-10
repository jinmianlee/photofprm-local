"""Exercise the local upload-region API and verify actual printable downloads."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

import httpx
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT/'data/region-api-validation.json'
API = 'http://127.0.0.1:8765'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['start', 'resume-small-hole', 'finish'])
    args = parser.parse_args()
    with httpx.Client(base_url=API, headers={'X-PhotoForm-Client':'local'}, timeout=60) as client:
        if args.phase == 'start':
            source = ROOT/'data/2412cfadbbde4c04a068dfd582f05a6a/foreground.png'
            region = {'x0':555/1280, 'y0':130/1339, 'x1':900/1280, 'y1':590/1339}
            options = {'size_mm':95, 'colors':3, 'color_style':'flat', 'mesh_resolution':255,
                       'photo_auto_align':True, 'pitch_mm':.8, 'min_feature_mm':.8}
            with source.open('rb') as file:
                response = client.post('/api/upload', data={'kind':'single', 'options':json.dumps(options),
                    'region':json.dumps(region)}, files={'files':('portrait.png', file, 'image/png')})
            response.raise_for_status()
            result = {'state':'started', 'job_id':response.json()['id'], 'region':region,
                      'source':str(source), 'options':options}
        elif args.phase == 'resume-small-hole':
            result = json.loads(RECORD.read_text('utf-8'))
            response = client.get('/api/jobs/'+result['job_id'])
            response.raise_for_status()
            assert response.json()['state'] == 'failed'
            result['initial_failure'] = response.json()['message']
            result['options']['repair_small_holes'] = True
            response = client.post('/api/jobs/'+result['job_id']+'/resume', json=result['options'])
            response.raise_for_status()
            result['state'] = 'resumed_with_bounded_small_hole_repair'
        else:
            result = json.loads(RECORD.read_text('utf-8'))
            job_id = result['job_id']
            response = client.get('/api/jobs/'+job_id)
            response.raise_for_status()
            status = response.json()
            if status['state'] != 'complete':
                print(json.dumps({'state':status['state'], 'message':status['message']}, ensure_ascii=False))
                return 2
            assert status['report']['generation']['input_region']['pixel_bounds_after_exif'] == [555,130,900,590]
            folder = ROOT/'data'/job_id
            assert (folder/'upload_0000.png').read_bytes() == Path(result['source']).read_bytes()
            assert hashlib.sha256((folder/'foreground.png').read_bytes()).digest() == hashlib.sha256((ROOT/'data/detail-head-only/foreground.png').read_bytes()).digest()
            downloads = ROOT/'data/region-download-validation'
            downloads.mkdir(exist_ok=True)
            solids = []
            for name in ['combined.stl']+[p['file'] for p in status['report']['parts']]:
                response = client.get(f'/api/jobs/{job_id}/files/{name}')
                response.raise_for_status()
                target = downloads/name
                target.write_bytes(response.content)
                mesh = trimesh.load(target, force='mesh', process=True)
                assert mesh.is_watertight and mesh.is_winding_consistent and mesh.volume > 0
                assert np.isfinite(mesh.vertices).all()
                solids.append({'file':name, 'faces':len(mesh.faces), 'watertight':True,
                               'winding_consistent':True, 'volume_mm3':float(mesh.volume)})
            for name in ['colored.3mf', 'print_bundle.zip']:
                response = client.get(f'/api/jobs/{job_id}/files/{name}')
                response.raise_for_status()
                (downloads/name).write_bytes(response.content)
                with zipfile.ZipFile(io.BytesIO(response.content)) as bundle:
                    assert bundle.testzip() is None
            result.update(state='validated', elapsed_seconds=status.get('elapsed_seconds'),
                small_hole_repair=status['report']['generation'].get('small_hole_repair'),
                solids=solids, part_count=len(status['report']['parts']),
                dimensions_mm=status['report']['final']['dimensions_mm'],
                inference=json.loads((folder/'inference.json').read_text('utf-8')),
                original_photo_preserved=True, actual_region_matches_experiment=True,
                physical_print_validated=False, likeness_accepted=False,
                automatic_head_body_assembly=False)
        RECORD.write_text(json.dumps(result, ensure_ascii=False, indent=2), 'utf-8')
        print(json.dumps({'state':result['state'], 'job_id':result['job_id']}, ensure_ascii=False))
        return 0


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    raise SystemExit(main())
