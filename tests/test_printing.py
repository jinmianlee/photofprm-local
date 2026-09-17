import itertools
import json
import struct
import zipfile
import xml.etree.ElementTree as ET
import numpy as np
import pytest
import trimesh
from backend.geometry import process_mesh, to_manifold, prepare_mesh, make_demo, load_mesh


@pytest.fixture(scope='module')
def export(tmp_path_factory):
    root = tmp_path_factory.mktemp('colored')
    source = root / 'source.ply'
    make_demo(source)
    options = dict(size_mm=40, colors=3, pitch_mm=1, min_feature_mm=.8)
    report = process_mesh(source, root / 'output', options, lambda *a: None)
    return root, report


def test_stls_are_manifold_and_form_a_nonoverlapping_partition(export):
    root, report = export
    meshes = [trimesh.load(root/'output'/p['file']) for p in report['parts']]
    assert len(meshes) == 3
    assert all(m.is_watertight and m.is_winding_consistent and m.volume > 0 for m in meshes)
    source = trimesh.load(root/'output'/'combined.stl')
    solids = [to_manifold(m) for m in meshes]
    for a, b in itertools.combinations(solids, 2):
        assert abs((a ^ b).volume()) < 1e-4
    union = solids[0]
    for s in solids[1:]:
        union = union + s
    original = to_manifold(source)
    assert abs((original-union).volume()) < .01
    assert abs((union-original).volume()) < .01
    assert np.max(source.extents) == pytest.approx(40, abs=1e-4)
    assert min(source.vertices[:,2]) == pytest.approx(0)


def test_3mf_has_colors_units_and_single_assembly(export):
    root, report = export
    ns = {'m':'http://schemas.microsoft.com/3dmanufacturing/core/2015/02'}
    with zipfile.ZipFile(root/'output'/'colored.3mf') as z:
        assert '[Content_Types].xml' in z.namelist()
        model = ET.fromstring(z.read('3D/3dmodel.model'))
    assert model.get('unit') == 'millimeter'
    colors = model.findall('m:resources/m:basematerials/m:base',ns)
    assert [c.get('displaycolor')[:7].lower() for c in colors] == [p['color'] for p in report['parts']]
    assert len(model.findall('m:build/m:item',ns)) == 1
    assert len(model.findall('m:resources/m:object/m:components/m:component',ns)) == len(report['parts'])
    for mesh in model.findall('m:resources/m:object/m:mesh',ns):
        vertex_count = len(mesh.findall('m:vertices/m:vertex',ns))
        for triangle in mesh.findall('m:triangles/m:triangle',ns):
            assert all(0 <= int(triangle.get(k)) < vertex_count for k in ('v1','v2','v3'))


def test_zip_contains_every_part_and_report(export):
    root, report = export
    with zipfile.ZipFile(root/'output'/'print_bundle.zip') as z:
        assert {'colored.3mf','combined.stl','report.json','READ_ME.txt'} <= set(z.namelist())
        assert all(p['file'] in z.namelist() for p in report['parts'])
        saved = json.loads(z.read('report.json'))
        assert saved['checks']['wall_thickness_verified'] is False


def test_preview_linear_materials_match_printable_srgb_palette(export):
    root, report = export
    raw = (root/'output'/'preview.glb').read_bytes()
    size, kind = struct.unpack_from('<II', raw, 12)
    assert kind == 0x4e4f534a
    tree = json.loads(raw[20:20+size])
    palette = {p['name']: p['color'] for p in report['parts']}
    for material in tree['materials']:
        linear = np.array(material['pbrMetallicRoughness']['baseColorFactor'][:3])
        srgb = np.where(linear <= .0031308, 12.92*linear, 1.055*linear**(1/2.4)-.055)
        expected = np.array([int(palette[material['name']][i:i+2],16) for i in (1,3,5)])
        assert np.allclose(srgb*255, expected, atol=.01)
    assert all('COLOR_0' not in primitive['attributes'] for mesh in tree['meshes'] for primitive in mesh['primitives'])


def test_cut_material_preview_keeps_exterior_normals_and_print_geometry(tmp_path):
    from backend.geometry import export_preview,from_manifold
    from scipy.spatial import cKDTree
    exterior=trimesh.creation.icosphere(subdivisions=3)
    piece=from_manifold(to_manifold(exterior).trim_by_plane((1,0,0),0))
    destination=tmp_path/'preview.glb'
    export_preview([(piece,np.array([140,75,30]))],destination,exterior)
    loaded=trimesh.load(destination,force='mesh',process=False)
    raw=destination.read_bytes();size=struct.unpack_from('<I',raw,12)[0]
    tree=json.loads(raw[20:20+size]);binary=20+size+8
    attrs=tree['meshes'][0]['primitives'][0]['attributes']
    def attribute(name):
        accessor=tree['accessors'][attrs[name]]
        view=tree['bufferViews'][accessor['bufferView']]
        assert accessor['componentType']==5126 and accessor['type']=='VEC3'
        return np.frombuffer(raw,dtype='<f4',count=accessor['count']*3,
            offset=binary+view.get('byteOffset',0)+accessor.get('byteOffset',0)).reshape(-1,3)
    # Scene-to-mesh import recalculates normals. Inspect the actual glTF
    # attributes consumed by browser renderers instead of that import cache.
    vertices,normals=attribute('POSITION'),attribute('NORMAL')
    distance,index=cKDTree(exterior.vertices).query(vertices)
    shared=distance<1.e-5
    assert shared.sum()>100
    assert np.allclose(normals[shared],exterior.vertex_normals[index[shared]],atol=1.e-5)
    assert loaded.is_watertight and abs(loaded.volume-piece.volume)<1.e-5


def test_open_surface_is_rejected(tmp_path):
    mesh = trimesh.creation.box()
    mesh.update_faces(np.arange(6))
    mesh.export(tmp_path/'open.ply')
    with pytest.raises(ValueError, match='开放边界'):
        prepare_mesh(tmp_path/'open.ply', 40)


def test_splat_or_point_cloud_is_not_treated_as_stl(tmp_path):
    cloud = trimesh.points.PointCloud(np.random.default_rng(1).random((40,3)))
    cloud.export(tmp_path/'cloud.ply')
    with pytest.raises(ValueError, match='三角面'):
        load_mesh(tmp_path/'cloud.ply')


def test_excessive_resolution_is_rejected_instead_of_silently_reduced(tmp_path):
    from backend.geometry import partition
    mesh = trimesh.creation.icosphere(radius=200)
    colors = np.tile([255,0,0,255],(len(mesh.vertices),1))
    colors[mesh.vertices[:,2]>0] = [0,0,255,255]
    mesh.visual.vertex_colors = colors
    with pytest.raises(ValueError, match='不会自动降低精度'):
        partition(mesh,2,.15,lambda *a:None)


def test_single_color_export_is_one_solid(tmp_path):
    mesh = trimesh.creation.box([10,15,20])
    mesh.export(tmp_path/'box.stl')
    report = process_mesh(tmp_path/'box.stl',tmp_path/'out',dict(size_mm=40,colors=8,pitch_mm=1,min_feature_mm=.8),lambda *a:None)
    assert len(report['parts']) == 1
    assert report['parts'][0]['watertight']


def test_external_glb_resource_is_rejected_before_loading(tmp_path):
    document = json.dumps({'asset':{'version':'2.0'},'buffers':[{'uri':'../../private.txt','byteLength':1}]}).encode()
    document += b' ' * (-len(document)%4)
    raw = struct.pack('<4sIIII',b'glTF',2,20+len(document),len(document),0x4E4F534A)+document
    (tmp_path/'external.glb').write_bytes(raw)
    with pytest.raises(ValueError,match='内嵌纹理'):
        load_mesh(tmp_path/'external.glb')


def test_inner_cavity_is_preserved(tmp_path):
    from backend.geometry import from_manifold
    outer = to_manifold(trimesh.creation.box([20,20,20]))
    inner = to_manifold(trimesh.creation.box([10,10,10]))
    shell = from_manifold(outer-inner)
    shell.export(tmp_path/'hollow.stl')
    prepared,_ = prepare_mesh(tmp_path/'hollow.stl',20)
    assert abs(to_manifold(prepared).volume()) == pytest.approx(7000,abs=.01)


def test_adjacent_minor_colors_leave_a_closed_remainder(tmp_path):
    # Adjacent upper colors share a curved boundary; the lower color must not
    # acquire a zero-thickness sheet when the upper regions are subtracted.
    mesh = trimesh.creation.icosphere(subdivisions=3)
    rgb = np.tile([170, 120, 60, 255], (len(mesh.vertices), 1))
    upper = mesh.vertices[:, 2] > .2
    rgb[upper & (mesh.vertices[:, 0] < 0)] = [20, 20, 20, 255]
    rgb[upper & (mesh.vertices[:, 0] >= 0)] = [220, 210, 180, 255]
    mesh.visual.vertex_colors = rgb
    source = tmp_path / 'adjacent.ply'; mesh.export(source)
    report = process_mesh(source, tmp_path / 'output',
                          dict(size_mm=30, colors=3, pitch_mm=.8, min_feature_mm=.8), lambda *a: None)
    assert len(report['parts']) == 3
    parts = [trimesh.load(tmp_path / 'output' / part['file']) for part in report['parts']]
    assert all(p.is_watertight and p.is_winding_consistent for p in parts)
    original = trimesh.load(tmp_path / 'output/combined.stl')
    assert sum(p.volume for p in parts) == pytest.approx(original.volume, rel=1e-5)


def test_embedded_texture_is_sampled_inside_faces(tmp_path):
    from PIL import Image
    from backend.geometry import palette_and_samples
    image=np.zeros((64,64,3),dtype=np.uint8)
    image[:,:32]=[255,0,0]
    image[:,32:]=[0,0,255]
    mesh=trimesh.Trimesh(vertices=[[0,0,0],[1,0,0],[0,1,0]],faces=[[0,1,2]])
    mesh.visual=trimesh.visual.texture.TextureVisuals(uv=[[0,0],[1,0],[0,1]],image=Image.fromarray(image))
    _,labels,colors=palette_and_samples(mesh,2)
    assert len(colors)==2
    assert len(np.unique(labels))==2
    assert any(c[0]>250 and c[2]<5 for c in colors)
    assert any(c[2]>250 and c[0]<5 for c in colors)


def test_tiny_colored_fragments_merge_without_changing_the_print_solid(tmp_path):
    # The blue body and red disconnected tiny islands form one source import.
    # Material cleanup must merge the tiny islands, preserving all geometry.
    body=trimesh.creation.icosphere(subdivisions=2,radius=5)
    body.visual.face_colors=[170,120,70,255]
    dots=[]
    for x in [-1,1]:
        dot=trimesh.creation.icosphere(subdivisions=2,radius=.2)
        dot.apply_translation([x,0,5.4]);dot.visual.face_colors=[15,15,15,255];dots.append(dot)
    source=tmp_path/'source.ply';trimesh.util.concatenate([body,*dots]).export(source)
    report=process_mesh(source,tmp_path/'out',dict(size_mm=10,colors=2,pitch_mm=.25,min_feature_mm=.8,merge_small_islands=True),lambda *a:None)
    assert report['sampling']['merged_small_material_islands']>=1
    assert report['sampling']['partition_volume_relative_error']<1e-5
    assert all(p['watertight'] and p['winding_consistent'] for p in report['parts'])
