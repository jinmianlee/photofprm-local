import io
import json
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from backend import app as module


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(module, 'DATA', tmp_path)
    with TestClient(module.app, base_url='http://localhost') as c:
        yield c


H = {'X-PhotoForm-Client':'local'}


def test_local_host_and_csrf_boundaries(client):
    assert client.get('/api/system').status_code == 200
    assert client.get('/api/system',headers={'host':'evil.example'}).status_code == 403
    assert client.post('/api/demo',json={}).status_code == 403
    assert client.post('/api/demo',json={},headers={**H,'Origin':'https://evil.example'}).status_code == 403


def test_validation_and_path_traversal(client):
    assert client.post('/api/demo',json={'size_mm':0},headers=H).status_code == 422
    assert client.post('/api/demo',json={'colors':9},headers=H).status_code == 422
    assert client.post('/api/demo',json={'pitch_mm':.01},headers=H).status_code == 422
    assert client.get('/api/jobs/invalid').status_code == 404
    assert client.get('/api/jobs/'+'f'*32+'/files/secret.txt').status_code == 404


def test_one_photo_cannot_claim_full_reconstruction(client):
    file=io.BytesIO(); Image.new('RGB',(640,640)).save(file,format='JPEG')
    r=client.post('/api/upload',data={'kind':'photos','options':'{}'},files={'files':('one.jpg',file.getvalue(),'image/jpeg')},headers=H)
    assert r.status_code == 422
    assert '8' in r.json()['detail']


def test_unsupported_upload_is_rejected(client):
    r=client.post('/api/upload',data={'kind':'mesh','options':'{}'},files={'files':('bad.exe',b'bad','application/octet-stream')},headers=H)
    assert r.status_code == 422


def test_no_unknown_options(client):
    assert client.post('/api/demo',json={'command':'whoami'},headers=H).status_code == 422


def test_nan_options_do_not_crash_error_serialization(client):
    assert client.post('/api/demo',content='{"size_mm":NaN}',headers={**H,'Content-Type':'application/json'}).status_code == 422


def test_single_photo_requires_validated_engine(client, monkeypatch):
    from backend import single_photo
    monkeypatch.setattr(single_photo, 'capability', lambda: {'ready': False})
    file = io.BytesIO(); Image.new('RGB', (576, 532)).save(file, format='PNG')
    response = client.post('/api/upload', data={'kind': 'single', 'options': '{}'},
                           files={'files': ('cat.png', file.getvalue(), 'image/png')}, headers=H)
    assert response.status_code == 409
    assert not list(module.DATA.glob('*/request.json'))


def test_single_photo_preserves_alpha_and_accepts_cat_resolution(client, monkeypatch):
    from backend import single_photo
    monkeypatch.setattr(single_photo, 'capability', lambda: {'ready': True})
    captured = []
    monkeypatch.setattr(module, 'launch', lambda path, request: captured.append((path, request)) or {'id': path.name})
    file = io.BytesIO(); Image.new('RGBA', (576, 532), (180, 120, 60, 73)).save(file, format='PNG')
    response = client.post('/api/upload', data={'kind': 'single', 'options': '{}'},
                           files={'files': ('cat.png', file.getvalue(), 'image/png')}, headers=H)
    assert response.status_code == 200
    path, request = captured[0]
    assert request['kind'] == 'single'
    with Image.open(path / 'images/photo_0000.png') as image:
        assert image.size == (576, 532)
        assert image.getchannel('A').getextrema() == (73, 73)


def test_generated_photo_source_can_be_recolored(client, monkeypatch):
    old = module.DATA / ('a' * 32); old.mkdir()
    (old / 'request.json').write_text(json.dumps({'kind': 'single', 'source_file': ''}))
    (old / 'source.ply').write_bytes(b'reference mesh')
    (old / 'generation.json').write_text(json.dumps({'model': 'Hunyuan3D-2mini'}))
    (old / 'generated_raw.ply').write_bytes(b'original generated shape')
    (old / 'foreground.png').write_bytes(b'original local foreground')
    captured = []
    monkeypatch.setattr(module, 'launch', lambda path, request: captured.append((path, request)) or {'id': path.name})
    response = client.post(f'/api/jobs/{old.name}/reprocess', json={'photo_pitch_deg': 15, 'photo_yaw_deg': -7}, headers=H)
    assert response.status_code == 200
    path, request = captured[0]
    assert request['kind'] == 'mesh'
    assert (path / 'source.ply').read_bytes() == b'reference mesh'
    assert json.loads((path / 'generation.json').read_text())['model'] == 'Hunyuan3D-2mini'
    assert (path / 'generated_raw.ply').read_bytes() == b'original generated shape'
    assert (path / 'foreground.png').read_bytes() == b'original local foreground'
    assert request['options']['photo_pitch_deg'] == 15 and request['options']['photo_yaw_deg'] == -7


def test_resume_keeps_same_input_and_uses_new_print_options(client, monkeypatch):
    old = module.DATA / ('b' * 32); old.mkdir()
    (old / 'request.json').write_text(json.dumps({'kind': 'single', 'source_file': '', 'options': {}}))
    (old / 'status.json').write_text(json.dumps({'kind': 'single', 'state': 'cancelled', 'id': old.name}))
    (old / 'diffusion_checkpoint.safetensors').write_bytes(b'checkpoint fixture')
    captured = []
    monkeypatch.setattr(module, 'launch', lambda path, request: captured.append((path, request)) or {'id': path.name})
    response = client.post(f'/api/jobs/{old.name}/resume', json={'colors': 3}, headers=H)
    assert response.status_code == 200
    path, request = captured[0]
    assert path == old and request['resume'] is True
    assert request['options']['colors'] == 3
    assert (old / 'diffusion_checkpoint.safetensors').read_bytes() == b'checkpoint fixture'


def test_status_does_not_overwrite_completion_during_exit_race(client, monkeypatch):
    job = module.DATA / ('c'*32); job.mkdir()
    running = {'id': job.name, 'kind': 'single', 'state': 'running', 'progress': 99}
    module.write_json(job / 'status.json', running)
    class JustFinished:
        def poll(self):
            module.write_json(job / 'status.json', {**running, 'state': 'complete', 'progress': 100})
            return 0
    monkeypatch.setitem(module.PROCESSES, job.name, JustFinished())
    assert module.state(job)['state'] == 'complete'
    assert json.loads((job / 'status.json').read_text())['state'] == 'complete'


def test_refine_reuses_image_bound_checkpoint_and_preserves_original(client, monkeypatch):
    old = module.DATA / ('d'*32); old.mkdir()
    module.write_json(old/'status.json', {'id': old.name, 'kind': 'single', 'state': 'complete'})
    (old/'foreground.png').write_bytes(b'original foreground')
    (old/'diffusion_checkpoint.safetensors').write_bytes(b'image-bound checkpoint')
    (old/'input_region.json').write_text('{"mode":"independent_region"}')
    captured = []
    monkeypatch.setattr(module, 'launch', lambda path, request: captured.append((path, request)) or {'id': path.name})
    response = client.post(f'/api/jobs/{old.name}/refine', json={'mesh_resolution': 383, 'color_style': 'flat'}, headers=H)
    assert response.status_code == 200
    path, request = captured[0]
    assert path != old and request['resume'] is True and request['kind'] == 'single'
    assert request['options']['mesh_resolution'] == 383
    assert (path/'foreground.png').read_bytes() == (old/'foreground.png').read_bytes()
    assert (path/'diffusion_checkpoint.safetensors').read_bytes() == b'image-bound checkpoint'
    assert (path/'input_region.json').read_bytes() == (old/'input_region.json').read_bytes()
    assert not (path/'generated_raw.ply').exists()
