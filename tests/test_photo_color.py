import numpy as np
from PIL import Image, ImageDraw
import trimesh
import zipfile
import xml.etree.ElementTree as ET
import pytest
from backend.geometry import export_3mf
from backend.photo_color import prepare_colored_bust, repair_tiny_holes


def test_photo_projection_does_not_wrap_dark_marks_onto_back(tmp_path):
    sphere = trimesh.creation.icosphere(subdivisions=4, radius=1)
    raw = tmp_path / 'raw.ply'; sphere.export(raw)
    picture = Image.new('RGBA', (256, 256), (190, 135, 75, 255))
    draw = ImageDraw.Draw(picture)
    draw.rectangle((80, 80, 175, 135), fill=(8, 8, 8, 255))
    foreground = tmp_path / 'foreground.png'; picture.save(foreground)
    target = tmp_path / 'source.ply'
    report = prepare_colored_bust(raw, foreground, target)
    mesh = trimesh.load(target, process=True)
    assert mesh.is_watertight and mesh.volume > 0
    # +Z model front maps to -Y in printer coordinates.
    back = mesh.vertices[:, 1] > .65
    front = mesh.vertices[:, 1] < -.65
    rgb = np.asarray(mesh.visual.vertex_colors[:, :3])
    assert np.all(rgb[back, 0] > 150)
    assert np.any(rgb[front, 0] < 70)
    assert report['base_trim_volume_fraction'] < .05
    bottom = mesh.vertices[:, 2] <= mesh.bounds[0, 2] + 1.e-6
    assert np.count_nonzero(bottom) > 3


def test_3mf_bambu_slots_match_core_material_parts(tmp_path):
    parts = [(trimesh.creation.box(), [230, 170, 65]),
             (trimesh.creation.box(), [45, 40, 35])]
    path = tmp_path / 'colors.3mf'
    export_3mf(parts, path)
    with zipfile.ZipFile(path) as archive:
        config = ET.fromstring(archive.read('Metadata/model_settings.config'))
        model = ET.fromstring(archive.read('3D/3dmodel.model'))
    ns = {'m': 'http://schemas.microsoft.com/3dmanufacturing/core/2015/02'}
    assembly = config.find('object')
    assert assembly.get('id') == model.find('m:build/m:item', ns).get('objectid')
    for index, part in enumerate(assembly.findall('part')):
        assert part.get('id') == str(index+2)
        assert part.find("metadata[@key='extruder']").get('value') == str(index+1)
        assert '#' in part.find("metadata[@key='name']").get('value')


def test_paper_thin_generation_is_not_exported_as_a_bust(tmp_path):
    mesh = trimesh.creation.box(extents=[1, 1, .01])
    path = tmp_path / 'raw.ply'; mesh.export(path)
    with pytest.raises(ValueError, match='过薄'):
        prepare_colored_bust(path, tmp_path / 'unused.png', tmp_path / 'source.ply')
    assert not (tmp_path / 'source.ply').exists()


def test_zero_area_mc_triangle_is_cleaned_but_open_surface_is_rejected(tmp_path):
    mesh = trimesh.creation.icosphere(subdivisions=2)
    mesh.faces = np.vstack((mesh.faces, [0, 0, 1]))
    raw = tmp_path / 'raw.ply'; mesh.export(raw)
    foreground = tmp_path / 'foreground.png'
    Image.new('RGBA', (256, 256), (190, 135, 75, 255)).save(foreground)
    target = tmp_path / 'source.ply'
    report = prepare_colored_bust(raw, foreground, target)
    assert report['removed_degenerate_or_duplicate_faces'] == 1
    assert trimesh.load(target).is_watertight
    opened = trimesh.creation.icosphere(subdivisions=2)
    opened.update_faces(np.arange(len(opened.faces)) != 0)
    opened.export(raw)
    with pytest.raises(ValueError, match='开放边界'):
        prepare_colored_bust(raw, foreground, tmp_path / 'bad.ply')


def test_small_hole_repair_preserves_surface_and_rejects_large_missing_face():
    opened = trimesh.creation.icosphere(subdivisions=4)
    opened.update_faces(np.arange(len(opened.faces)) != 0)
    repaired, report = repair_tiny_holes(opened, 10)
    assert repaired.is_watertight and repaired.is_winding_consistent
    assert np.array_equal(repaired.vertices, opened.vertices)
    assert report['added_faces'] == 1 and report['max_patch_edge_mm'] <= .8
    assert not opened.is_watertight, 'Original input was changed'
    with pytest.raises(ValueError, match='0.8 mm'):
        repair_tiny_holes(opened, 95)
