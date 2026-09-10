import io
import json

import pytest
from PIL import Image
from fastapi.testclient import TestClient
from backend import app as module
from backend.photo_region import PhotoRegion


def test_region_crops_original_pixels_without_resampling():
    image = Image.new('RGBA', (800, 700), (12, 28, 41, 73))
    image.putpixel((310, 150), (70, 80, 90, 255))
    cropped, audit = PhotoRegion(x0=.375, y0=.2, x1=.875, y1=.8).crop(image)
    assert cropped.size == (400, 420)
    assert cropped.getpixel((10, 10)) == (70, 80, 90, 255)
    assert cropped.getpixel((0, 0)) == (12, 28, 41, 73)
    assert audit['pixel_bounds_after_exif'] == [300, 140, 700, 560]
    assert image.size == (800, 700)
    with pytest.raises(ValueError, match='256'):
        PhotoRegion(x0=0, y0=0, x1=.2, y1=.2).crop(image)


@pytest.mark.parametrize('bounds', [dict(x0=0,y0=0,x1=float('nan'),y1=1), dict(x0=.8,y0=0,x1=.4,y1=1), dict(x0=0,y0=0,x1=1.1,y1=1)])
def test_invalid_region_is_rejected(bounds):
    with pytest.raises(ValueError):
        PhotoRegion(**bounds)


def test_api_region_precedes_foreground_and_keeps_original(tmp_path, monkeypatch):
    from backend import single_photo
    monkeypatch.setattr(module, 'DATA', tmp_path)
    monkeypatch.setattr(single_photo, 'capability', lambda: {'ready': True})
    captured = []
    monkeypatch.setattr(module, 'launch', lambda path, request: captured.append((path, request)) or {'id':path.name})
    buffer = io.BytesIO()
    Image.new('RGBA', (800, 700), (90, 150, 30, 87)).save(buffer, format='PNG')
    photo = buffer.getvalue()
    region = {'x0':.375,'y0':.2,'x1':.875,'y1':.8}
    with TestClient(module.app, base_url='http://localhost') as client:
        response = client.post('/api/upload', headers={'X-PhotoForm-Client':'local'},
            data={'kind':'single','options':'{}','region':json.dumps(region)},
            files={'files':('original.png', photo, 'image/png')})
        assert response.status_code == 200, response.text
        path, _ = captured[0]
        assert (path/'upload_0000.png').read_bytes() == photo
        with Image.open(path/'images/photo_0000.png') as image:
            assert image.size == (400, 420)
            assert image.getchannel('A').getextrema() == (87, 87)
        assert json.loads((path/'input_region.json').read_text())['normalized_bounds'] == region
        response = client.post('/api/upload', headers={'X-PhotoForm-Client':'local'},
            data={'kind':'photos','options':'{}','region':json.dumps(region)},
            files={'files':('original.png', photo, 'image/png')})
        assert response.status_code == 422
        assert len(captured) == 1
