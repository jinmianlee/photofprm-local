import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest
from fastapi.testclient import TestClient
from backend import app as module

H = {'X-PhotoForm-Client': 'local'}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(module, 'DATA', tmp_path)
    monkeypatch.setattr(module, 'PROCESSES', {})
    monkeypatch.delenv('PHOTOFORM_LAN', raising=False)
    with TestClient(module.app, base_url='http://localhost') as value:
        yield value


def task(name='a', status='complete'):
    path = module.DATA/(name*32)
    path.mkdir()
    module.write_json(path/'status.json', {'id': path.name, 'kind': 'mesh', 'state': status})
    (path/'source.stl').write_bytes(b'mesh fixture')
    return path


def test_delete_only_selected_job_and_missing_records_remain_manageable(client):
    chosen = task()
    kept = task('b')
    private = module.DATA/'engine-check.json'
    private.write_text('{}')
    assert client.delete('/api/jobs/'+chosen.name).status_code == 403
    files = client.get('/api/jobs/'+chosen.name+'/files').json()
    assert any(f['name'] == 'source.stl' and f['bytes'] == 12 for f in files)
    assert client.delete('/api/jobs/'+chosen.name, headers=H).status_code == 200
    assert not chosen.exists() and kept.is_dir() and private.is_file()
    assert client.delete('/api/jobs/..', headers=H).status_code in (404, 405)
    (kept/'status.json').write_text('broken JSON')
    listed = client.get('/api/jobs').json()
    assert listed[0]['state'] == 'failed' and listed[0]['storage_bytes'] > 0
    assert client.delete('/api/jobs/'+kept.name, headers=H).status_code == 200


def test_history_does_not_hide_older_tasks(client):
    for i in range(40):
        path = module.DATA/f'{i:032x}'
        path.mkdir()
        module.write_json(path/'status.json', {'state': 'cancelled'})
    assert len(client.get('/api/jobs').json()) == 40


def test_resume_rejects_a_different_seed_without_mutating_original_request(client, monkeypatch):
    path=task(status='cancelled')
    module.write_json(path/'status.json',{'id':path.name,'kind':'single','state':'cancelled'})
    request={'kind':'single','options':{'shape_steps':5,'shape_seed':12345}}
    module.write_json(path/'request.json',request)
    (path/'diffusion_checkpoint.safetensors').write_bytes(b'fixture')
    calls=[]
    monkeypatch.setattr(module,'launch',lambda *args:calls.append(args))
    response=client.post('/api/jobs/'+path.name+'/resume',headers=H,json={'shape_seed':42})
    assert response.status_code==409 and not calls
    assert json.loads((path/'request.json').read_text('utf-8'))==request


def test_links_cannot_escape_job_cleanup(client, tmp_path):
    path = task()
    outside = module.DATA/'shared'
    outside.mkdir()
    (outside/'keep.txt').write_text('keep')
    try:
        (path/'linked').symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip('Symlink creation not permitted on this Windows account')
    assert client.delete('/api/jobs/'+path.name, headers=H).status_code == 409
    assert (outside/'keep.txt').read_text() == 'keep'


def test_cancel_waits_for_real_worker_exit_before_deletion(client):
    path = task(status='running')
    child_pid_file = path/'child-pid.txt'
    from backend.process_tree import ManagedProcess
    code = "from backend.process_tree import wait_for_start; wait_for_start(); import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); open(sys.argv[1],'w').write(str(p.pid)); time.sleep(60)"
    process = ManagedProcess([sys.executable, '-c', code, str(child_pid_file)],
                               start_new_session=os.name != 'nt',
                               creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    module.PROCESSES[path.name] = process
    child_handle = None
    try:
        deadline = time.monotonic()+10
        while not child_pid_file.exists() and time.monotonic() < deadline:
            time.sleep(.05)
        child_pid = int(child_pid_file.read_text())
        if os.name == 'nt':
            import ctypes
            kernel = ctypes.WinDLL('kernel32', use_last_error=True)
            kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
            kernel.OpenProcess.restype = ctypes.c_void_p
            kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
            kernel.CloseHandle.argtypes = [ctypes.c_void_p]
            child_handle = kernel.OpenProcess(0x100000, False, child_pid)
            assert child_handle
        assert client.delete('/api/jobs/'+path.name, headers=H).status_code == 409
        response = client.post('/api/jobs/'+path.name+'/cancel', headers=H)
        assert response.status_code == 200 and response.json()['state'] == 'cancelled'
        assert process.poll() is not None and (path/'source.stl').is_file()
        if child_handle:
            assert kernel.WaitForSingleObject(child_handle, 3000) == 0
        assert client.delete('/api/jobs/'+path.name, headers=H).status_code == 200
    finally:
        if child_handle:
            kernel.CloseHandle(child_handle)
        if process.poll() is None:
            process.kill()
            process.wait()


def test_lan_pairing_protects_private_files_and_mutations(client, monkeypatch):
    path = task()
    monkeypatch.setenv('PHOTOFORM_LAN', '1')
    monkeypatch.setenv('PHOTOFORM_ALLOWED_HOSTS', '192.168.1.25')
    monkeypatch.setenv('PHOTOFORM_LAN_KEY', 'test-key-with-sufficient-length')
    headers = {**H, 'Host': '192.168.1.25:8765', 'Origin': 'http://192.168.1.25:8765'}
    assert client.get('/api/jobs', headers=headers).status_code == 401
    assert client.get('/api/jobs/'+path.name+'/files/source.stl', headers=headers).status_code == 401
    assert client.post('/api/connect', json={'key': 'wrong'}, headers=headers).status_code == 401
    response = client.post('/api/connect', json={'key': 'test-key-with-sufficient-length'}, headers=headers)
    assert response.status_code == 200
    cookie = response.cookies.get('photoform_session')
    headers['Cookie'] = 'photoform_session='+cookie
    assert 'HttpOnly' in response.headers['set-cookie'] and 'SameSite=strict' in response.headers['set-cookie']
    assert client.get('/api/jobs', headers=headers).status_code == 200
    assert client.delete('/api/jobs/'+path.name, headers={**headers, 'Origin': 'http://192.168.1.25:8888'}).status_code == 403
    assert client.delete('/api/jobs/'+path.name, headers={**headers, 'Origin': 'https://evil.example'}).status_code == 403
    assert client.get('/api/jobs', headers={**headers, 'Host': 'evil.example'}).status_code == 403
    assert client.delete('/api/jobs/'+path.name, headers=headers).status_code == 200
