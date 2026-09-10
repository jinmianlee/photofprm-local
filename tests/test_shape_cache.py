import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
from backend.shape_cache import generation_signature, record_shape, completed_shape
from backend import single_photo


def cache(job):
    image = job / 'foreground.png'; image.write_bytes(b'current photo')
    raw = job / 'generated_raw.ply'; raw.write_bytes(b'mesh payload' * 20)
    shape = SimpleNamespace(faces=[0]*10, vertices=[0]*8, extents=np.array([1., 2., 1.]))
    record_shape(job, generation_signature(image), 255, shape)
    return image, raw


def test_reuse_rejects_different_image_resolution_or_changed_mesh(tmp_path):
    image, raw = cache(tmp_path)
    assert completed_shape(tmp_path, image)
    assert not completed_shape(tmp_path, image, resolution=127)
    image.write_bytes(b'another photo')
    assert not completed_shape(tmp_path, image)
    image.write_bytes(b'current photo')
    raw.write_bytes(b'different mesh' * 20)
    assert not completed_shape(tmp_path, image)


def test_resume_finished_shape_never_starts_ai_process(tmp_path, monkeypatch):
    cache(tmp_path)
    monkeypatch.setattr(single_photo, 'capability', lambda: {'ready': True})
    def must_not_run(*args, **kwargs):
        raise AssertionError('Finished geometry restarted the AI process')
    monkeypatch.setattr(single_photo, 'run_stage', must_not_run)
    from backend import photo_color
    prepare = Mock(return_value={'local': True})
    monkeypatch.setattr(photo_color, 'prepare_colored_bust', prepare)
    source, _ = single_photo.reconstruct(tmp_path, {}, lambda *a: None, resume=True)
    assert source == tmp_path / 'source.ply'
    prepare.assert_called_once()
