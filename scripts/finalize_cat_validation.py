"""Record completed real-cat checks, only after inspecting actual artifacts."""
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import zipfile
import numpy as np
import trimesh
import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.settings import write_json

def read(path):
    return json.loads(path.read_text('utf-8'))

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

data = ROOT / 'data'
app = read(data / 'cat-app-validation.json')
assert app['state'] == 'complete' and app['downloads_verified']
job = data / app['id']
status = read(job / 'status.json')
assert status['state'] == 'complete' and len(status['report']['parts']) == 4
reference = data / 'cat-color-angle25-yaw7'
sliced = read(reference / 'slice-four-colors-positioned/result.json')
assert sliced['return_code'] == 0 and sliced['error_string'] == 'Success.'
plate = sliced['sliced_plates'][0]
assert {p['id'] for p in plate['filaments']} == {1, 2, 3, 4}
assert plate['filament_change_times'] > 0 and not plate['warning_message']
assert abs(sliced['layer_height']-.16) < 1e-5

ns = {'m': 'http://schemas.microsoft.com/3dmanufacturing/core/2015/02'}
with zipfile.ZipFile(reference / 'bambu-import.3mf') as archive:
    config = ET.fromstring(archive.read('Metadata/model_settings.config'))
    model = ET.fromstring(archive.read('3D/3dmodel.model'))
    objects = ET.fromstring(archive.read('3D/Objects/object_1.model'))
part_metadata = config.findall('object/part')
transforms = model.findall('.//m:components/m:component', ns)
records = []
for i, part in enumerate(status['report']['parts']):
    path = job / 'output' / part['file']
    # The native slicer test must concern the same geometry as the live app.
    assert sha(path) == sha(reference / 'output' / part['file'])
    mesh = trimesh.load(path)
    assert mesh.is_watertight and mesh.is_winding_consistent and mesh.volume > 0
    metadata = part_metadata[i]
    assert metadata.find("metadata[@key='extruder']").get('value') == str(i+1)
    stats = metadata.find('mesh_stat').attrib
    assert int(stats['face_count']) == len(mesh.faces)
    assert all(int(v) == 0 for k, v in stats.items() if k != 'face_count')
    component = transforms[i]
    obj = objects.find(f"m:resources/m:object[@id='{component.get('objectid')}']", ns)
    vertices = np.array([[float(v.get(k)) for k in ('x', 'y', 'z')]
                         for v in obj.findall('m:mesh/m:vertices/m:vertex', ns)])
    transform = np.array([float(n) for n in component.get('transform').split()])
    vertices = vertices @ transform[:9].reshape(3, 3) + transform[9:]
    native_bounds = np.array([vertices.min(0), vertices.max(0)])
    assert np.allclose(native_bounds, mesh.bounds, atol=.0001)
    records.append({'file': part['file'], 'sha256': sha(path), 'watertight_after_stl_reload': True,
                    'native_import_repairs': 0, 'relative_placement_preserved': True, 'filament_slot': i+1})
combined = trimesh.load(job / 'output/combined.stl')
assert combined.is_watertight and abs(combined.extents.max()-95) < .0001
assert sha(job / 'output/preview.glb') == sha(reference / 'output/preview.glb')

recolor = read(data / 'cat-recolor-validation.json')
recolored_job = data / recolor['id']
with httpx.Client(base_url='http://127.0.0.1:8765', trust_env=False, timeout=30) as client:
    response = client.get('/api/jobs/' + recolor['id'])
    response.raise_for_status()
    recolored = response.json()
    assert recolored['state'] == 'complete' and len(recolored['report']['parts']) == 3
    assert client.get('/api/jobs/' + recolor['id'] + '/files/colored.3mf').status_code == 200
assert not (recolored_job / 'inference.json').exists()
assert sha(recolored_job / 'generated_raw.ply') == sha(job / 'generated_raw.ply')
assert recolored['report']['generation']['photo_pitch_deg'] == 25
assert recolored['report']['generation']['photo_yaw_deg'] == 7
recolor.update(state='complete', three_parts_verified=True, raw_shape_unchanged=True, inference_not_repeated=True)
write_json(data / 'cat-recolor-validation.json', recolor)

validation = read(data / 'single-photo-validation.json')
validation.update(api_job=app['id'], api_downloads_verified=True, recolor_without_inference_verified=True,
                  stl_records=records, offline_four_color_slicing_validated=True,
                  slicer_result=str((reference / 'slice-four-colors-positioned/result.json').relative_to(ROOT)),
                  slicer_test_profile='A1 / 0.4 mm / Generic PLA / 0.16 mm; not a confirmed user printer',
                  slicer_estimated_hours=round(plate['total_predication']/3600, 2),
                  slicer_filament_changes=plate['filament_change_times'],
                  slicer_total_filament_g=round(sum(p['total_used_g'] for p in plate['filaments']), 2),
                  tests_passed=28, frontend_build_passed=True, frontend_lint_passed=True,
                  browser_ui_qa_performed=False)
write_json(data / 'single-photo-validation.json', validation)
runtime = read(data / 'xpu-runtime-check.json')
runtime.update(inference_validated=True, inference_report=str((job / 'inference.json').relative_to(ROOT)))
write_json(data / 'xpu-runtime-check.json', runtime)
manifest_path = ROOT / 'models/hunyuan3d-2mini/download-manifest-turbo.json'
manifest = read(manifest_path)
manifest.pop('cat_quality_validated', None)
manifest.update(cat_geometry_visually_reviewed=True, print_pipeline_validated=True,
                physical_print_validated=False, user_likeness_acceptance='not_assessed',
                validation_record='data/single-photo-validation.json')
write_json(manifest_path, manifest)
app.pop('verification_resumed_after_server_restart', None)
app['verification_resumed_after_interruption'] = True
write_json(data / 'cat-app-validation.json', app)
print(json.dumps({'api_job': app['id'], 'stl_parts': len(records), 'slicer_success': True,
                  'recolor_success': True, 'tests_passed': 28}, indent=2))
