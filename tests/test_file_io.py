import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import pytest
from backend.file_io import write_json, replace_with_retry, read_json


@pytest.mark.skipif(os.name != 'nt', reason='Real Windows sharing-violation regression')
def test_progress_replace_waits_for_real_windows_reader(tmp_path):
    target = tmp_path / 'status.json'
    write_json(target, {'state': 'running', 'progress': 5})
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                  wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    # Allow concurrent reading/writing, but block deletion/rename just as a
    # normal open Windows reader does. No ACL or security setting is changed.
    handle = kernel.CreateFileW(str(target), 0x80000000, 3, None, 3, 0, None)
    assert handle != wintypes.HANDLE(-1).value
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(write_json, target, {'state': 'complete', 'progress': 100})
            time.sleep(.15)
            assert not future.done()
            assert json.loads(target.read_text())['progress'] == 5
            kernel.CloseHandle(handle); handle = None
            future.result(timeout=3)
    finally:
        if handle is not None:
            kernel.CloseHandle(handle)
    assert json.loads(target.read_text())['state'] == 'complete'
    assert not list(tmp_path.glob('*.tmp'))


def test_simultaneous_writers_publish_whole_independent_documents(tmp_path):
    target = tmp_path / 'status.json'
    write_json(target, {'writer': -1, 'text': 'initial'})
    barrier = threading.Barrier(4)
    def writer(index):
        barrier.wait()
        for _ in range(15):
            write_json(target, {'writer': index, 'text': str(index)*10000})
            value = read_json(target)
            assert value['text'] == str(value['writer'])*10000
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(writer, range(4)))
    assert not list(tmp_path.glob('*.tmp'))


def test_real_permission_failure_is_not_reported_as_success(tmp_path, monkeypatch):
    target = tmp_path / 'status.json'; target.write_text('original')
    source = tmp_path / 'new.tmp'; source.write_text('new')
    def denied(*args):
        raise PermissionError(13, 'persistently denied')
    monkeypatch.setattr(os, 'replace', denied)
    with pytest.raises(PermissionError):
        replace_with_retry(source, target, timeout=.03)
    assert target.read_text() == 'original'
