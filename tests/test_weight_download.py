import hashlib
import importlib.util
import io
from pathlib import Path
import re

import pytest


@pytest.fixture
def downloader(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('weight_download',
        Path(__file__).resolve().parents[1] / 'scripts/download_hunyuan_weight.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(module, 'DEST', tmp_path / 'model.safetensors')
    monkeypatch.setattr(module, 'PART', tmp_path / 'model.download')
    monkeypatch.setattr(module, 'STATUS', tmp_path / 'status.json')
    monkeypatch.setattr(module.time, 'sleep', lambda _: None)
    return module


def test_connection_interruption_resumes_without_duplicate_bytes(downloader, monkeypatch):
    payload = b'cat-mesh-model' * 180000
    downloader.SIZE = len(payload)
    downloader.SHA = hashlib.sha256(payload).hexdigest()
    offsets = []

    class Response(io.BytesIO):
        def read1(self, n=-1):
            return self.read(n)

        def read(self, n=-1):
            if len(offsets) == 1 and self.tell() >= 1024*1024:
                raise OSError('simulated connection drop')
            return super().read(n)

    def connect(request, timeout):
        start, end = map(int, re.fullmatch(r'bytes=(\d+)-(\d+)', request.get_header('Range')).groups())
        offsets.append(start)
        result = Response(payload[start:end+1])
        result.headers = {'Content-Range': f'bytes {start}-{end}/{len(payload)}'}
        return result

    monkeypatch.setattr(downloader.urllib.request, 'urlopen', connect)
    downloader.main()
    assert offsets == [0, 1024*1024]
    assert downloader.DEST.read_bytes() == payload
    assert not downloader.PART.exists()


def test_corrupted_complete_weight_is_never_accepted(downloader):
    downloader.SIZE = 4
    downloader.SHA = hashlib.sha256(b'good').hexdigest()
    downloader.PART.write_bytes(b'evil')
    with pytest.raises(RuntimeError, match='SHA-256 mismatch'):
        downloader.main()
    assert not downloader.DEST.exists()
    assert downloader.PART.read_bytes() == b'evil'


def test_server_ignoring_range_cannot_corrupt_partial_file(downloader, monkeypatch):
    downloader.SIZE = 8
    downloader.PART.write_bytes(b'good')
    response = io.BytesIO(b'goodtail')
    response.headers = {}
    monkeypatch.setattr(downloader.urllib.request, 'urlopen', lambda *_a, **_k: response)
    with pytest.raises(RuntimeError, match='exact requested byte range'):
        downloader.main()
    assert downloader.PART.read_bytes() == b'good'
