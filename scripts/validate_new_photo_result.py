"""Verify an actual local upload's exports and compare the optimized shape.

Run after the job completes: python scripts/validate_new_photo_result.py
This reads only local inputs and the loopback API; it does not start inference.
"""
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile
import xml.etree.ElementTree as ET

import httpx
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.file_io import read_json, write_json


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    target = ROOT / 'data/new-photo-efficient-validation.json'
    record = read_json(target)
    job = ROOT / 'data' / record['job_id']
    standard = ROOT / 'data' / record['standard_job']
    previous = ROOT / 'data' / record['source_job']
    with httpx.Client(base_url='http://127.0.0.1:8765', trust_env=False, timeout=60) as client:
        response = client.get('/api/jobs/' + job.name)
        response.raise_for_status()
        status = response.json()
        assert status['state'] == 'complete', status.get('message')
        assert status['report']['final']['watertight']
        assert len(status['report']['parts']) == 4
        checked = []
        for name in ['combined.stl', 'colored.3mf', 'print_bundle.zip', 'preview.glb',
                     *[part['file'] for part in status['report']['parts']]]:
            response = client.get(f'/api/jobs/{job.name}/files/{name}')
            response.raise_for_status()
            assert hashlib.sha256(response.content).hexdigest() == sha(job / 'output' / name)
            if name.endswith('.stl'):
                mesh = trimesh.load(io.BytesIO(response.content), file_type='stl', force='mesh', process=True)
                assert mesh.is_watertight and mesh.is_winding_consistent
                assert np.isfinite(mesh.vertices).all() and mesh.volume > 0
            if name.endswith(('.zip', '.3mf')):
                with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
                    assert archive.testzip() is None
                    if name.endswith('.3mf'):
                        model = ET.fromstring(archive.read('3D/3dmodel.model'))
                        assert model.attrib['unit'] == 'millimeter'
                        assert len(model.findall('.//{*}mesh')) == 4
            checked.append(name)
    comparisons = {}
    for filename in ('foreground.png', 'shape_latents.safetensors', 'generated_raw.ply'):
        optimized_sha, standard_sha, original_sha = sha(job/filename), sha(standard/filename), sha(previous/filename)
        comparisons[filename] = {'sha256': optimized_sha, 'equals_standard': optimized_sha == standard_sha,
                                 'equals_original': optimized_sha == original_sha}
        assert optimized_sha == standard_sha == original_sha, f'Changed generation: {filename}'
    inference = read_json(job / 'inference.json')
    assert inference['weight_loading'] == 'direct_assignment'
    assert inference['steps'] == 5 and inference['resolution'] == 255
    record.update(validated=True, status='complete', exports_checked=checked,
                  comparison=comparisons, elapsed_seconds=status['elapsed_seconds'],
                  standard_elapsed_seconds=read_json(standard/'status.json')['elapsed_seconds'],
                  phase_seconds=inference['phase_seconds'],
                  standard_phase_seconds=read_json(standard/'inference.json')['phase_seconds'],
                  timings=read_json(job/'timings.json'),
                  physical_print_tested=False, photo_likeness_accepted_by_user=False,
                  notes='Exact foreground, latent and raw mesh bytes preserved; filament colors and photo alignment need review.')
    write_json(target, record)
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
