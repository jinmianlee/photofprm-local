"""Solid colour partitioning. Booleans retain the original exterior triangles.

The sampling pitch controls *material boundaries*, not reconstruction accuracy.
Regions partition the whole bounding box, then intersect a manifold source.
Sequential differences make parts disjoint; the last material receives the
remainder, so their union equals the source within boolean numeric tolerance.
"""
from pathlib import Path
import json
import struct
import zipfile
import xml.etree.ElementTree as ET
import numpy as np
import trimesh
import manifold3d as m3d
from scipy.cluster.vq import kmeans2
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter, distance_transform_edt, maximum_filter, minimum_filter
from skimage.color import rgb2lab, lab2rgb
from skimage.measure import marching_cubes
from .settings import write_json

MAX_GRID_POINTS = 8_000_000


def to_manifold(mesh):
    result = m3d.Manifold(m3d.Mesh(
        vert_properties=np.asarray(mesh.vertices, dtype=np.float32),
        tri_verts=np.asarray(mesh.faces, dtype=np.uint32)))
    if result.status() != m3d.Error.NoError:
        raise ValueError('网格不是有效的封闭实体。请补齐缺失视角，或修复网格后重新导入。')
    return result


def from_manifold(solid):
    raw = solid.to_mesh()
    mesh = trimesh.Trimesh(vertices=np.asarray(raw.vert_properties)[:, :3],
                           faces=np.asarray(raw.tri_verts), process=True)
    mesh.update_faces(mesh.nondegenerate_faces() & mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    return mesh


def load_mesh(path):
    # GLB may legally reference external resources. This local uploader accepts
    # only embedded assets, so an uploaded model cannot read arbitrary files.
    if Path(path).suffix.lower() == '.glb':
        raw = Path(path).read_bytes()
        if len(raw) < 20 or raw[:4] != b'glTF':
            raise ValueError('GLB 文件头无效。')
        size, chunk_type = struct.unpack_from('<II', raw, 12)
        if chunk_type != 0x4E4F534A or 20 + size > len(raw):
            raise ValueError('GLB 结构无效。')
        document = json.loads(raw[20:20+size])
        for collection in ('buffers', 'images'):
            if any('uri' in entry and not entry['uri'].startswith('data:') for entry in document.get(collection, [])):
                raise ValueError('只支持内嵌纹理 GLB，不能引用本机路径或外部网址。')
    obj = trimesh.load(path, process=False)
    if isinstance(obj, trimesh.Scene):
        obj = obj.to_mesh()
    if not isinstance(obj, trimesh.Trimesh) or not len(obj.faces):
        raise ValueError('需要含三角面的网格。3DGS 的高斯 PLY/点云必须先提取表面，不能直接改名成 STL。')
    if not np.isfinite(obj.vertices).all():
        raise ValueError('模型含无效坐标。')
    if len(obj.faces) > 3_000_000:
        raise ValueError('模型超过 300 万面，请先在重建工具中适度简化。')
    obj.process(validate=True)
    return obj


def mesh_report(mesh):
    return {'vertices': int(len(mesh.vertices)), 'faces': int(len(mesh.faces)),
            'watertight': bool(mesh.is_watertight),
            'winding_consistent': bool(mesh.is_winding_consistent),
            'volume_mm3': round(float(abs(mesh.volume)), 3),
            'dimensions_mm': np.round(mesh.extents, 3).tolist(),
            'components': int(len(mesh.split(only_watertight=False)))}


def prepare_mesh(source, width, repair=False):
    mesh = load_mesh(source)
    if float(mesh.extents.max()) < 1e-9:
        raise ValueError('模型尺寸为零。')
    mesh.apply_translation(-mesh.bounds.mean(axis=0))
    mesh.apply_scale(width / mesh.extents.max())
    mesh.apply_translation([0, 0, -mesh.bounds[0, 2]])
    before = mesh_report(mesh)
    if repair:
        # Only fill triangles/quads. No Poisson reconstruction or silent remeshing.
        trimesh.repair.fill_holes(mesh)
    # Preserve intentionally inverted inner shells (hollow models).
    trimesh.repair.fix_normals(mesh, multibody=False)
    if not mesh.is_watertight:
        raise ValueError('重建网格存在开放边界，已保留原始彩色模型供检查。请补拍底部/遮挡处，或使用网格修复工具补洞后重新导入。')
    return mesh, before


def palette_and_samples(mesh, count, balanced=False, fixed_palette=None):
    # Area-weighted surface samples avoid over-representing tiny triangles.
    points, face_ids = trimesh.sample.sample_surface(mesh, min(120_000, max(20_000, len(mesh.faces))), seed=17)
    if mesh.visual.kind == 'texture':
        bary = trimesh.triangles.points_to_barycentric(mesh.triangles[face_ids], points)
        uv = np.einsum('ni,nij->nj', bary, mesh.visual.uv[mesh.faces[face_ids]])
        rgb = np.asarray(mesh.visual.material.to_color(uv))[:, :3] / 255.0
    elif mesh.visual.kind == 'face':
        rgb = np.asarray(mesh.visual.face_colors)[face_ids, :3] / 255.0
    else:
        vertex_rgb = np.asarray(mesh.visual.vertex_colors)[:, :3] / 255.0
        bary = trimesh.triangles.points_to_barycentric(mesh.triangles[face_ids], points)
        rgb = np.einsum('ni,nij->nj', bary, vertex_rgb[mesh.faces[face_ids]])
    rgb = np.clip(rgb, 0, 1)
    labs = rgb2lab(rgb.reshape(-1, 1, 3)).reshape(-1, 3)
    if fixed_palette is not None:
        colors = np.asarray(fixed_palette, dtype=np.uint8)
        centers = rgb2lab((colors/255.0).reshape(-1, 1, 3)).reshape(-1, 3)
        labels = cKDTree(centers).query(labs)[1]
        return points, labels, colors
    unique = np.unique(np.round(labs, 0), axis=0)
    count = min(count, len(unique))
    if count == 1:
        centers = labs.mean(axis=0, keepdims=True)
        labels = np.zeros(len(labs), dtype=int)
    else:
        # Photo busts have a large inferred back with one color. Balance coarse
        # Lab bins so this area cannot consume nearly the whole palette and
        # erase smaller light/dark facial colors. Generic imported meshes keep
        # the original area-weighted strategy.
        training = np.unique(np.round(labs/4)*4, axis=0) if balanced else labs[::max(1, len(labs)//20000)]
        centers, _ = kmeans2(training, min(count, len(training)),
                            iter=30, minit='++', seed=17)
        labels = cKDTree(centers).query(labs)[1]
        active = np.unique(labels)
        centers = centers[active]
        labels = cKDTree(centers).query(labs)[1]
    colors = np.rint(np.clip(lab2rgb(centers.reshape(-1, 1, 3)).reshape(-1, 3), 0, 1)*255).astype(np.uint8)
    return points, labels, colors


def partition(mesh, count, pitch, update, balanced=False, fixed_palette=None, min_island_mm=0):
    source = to_manifold(mesh)
    points, labels, colors = palette_and_samples(mesh, count, balanced=balanced, fixed_palette=fixed_palette)
    if len(colors) == 1:
        return [(mesh, colors[0])], {'grid_points': 0, 'pitch_mm': pitch}
    origin = mesh.bounds[0] - pitch*3
    shape = np.ceil((mesh.extents+pitch*6)/pitch).astype(int)+1
    size = int(np.prod(shape))
    if size > MAX_GRID_POINTS:
        suggested = pitch * (size/MAX_GRID_POINTS)**(1/3) * 1.03
        raise ValueError(f'该尺寸和分色间距需要 {size:,} 个采样点，超出本机任务上限。请将分色间距调至至少 {suggested:.2f} mm，或缩小模型。不会自动降低精度。')
    tree = cKDTree(points)
    update(58, '建立表面材料种子，计算空间距离场')
    cell=np.clip(np.rint((points-origin)/pitch).astype(int),0,shape-1)
    seed_ids=np.ravel_multi_index(cell.T,tuple(shape))
    displacement=np.sum((points-(cell*pitch+origin))**2,axis=1)
    order=np.lexsort((displacement,seed_ids))
    sorted_ids=seed_ids[order]
    first=np.r_[True,sorted_ids[1:]!=sorted_ids[:-1]]
    selected=order[first]
    seed_field=np.full(size,255,dtype=np.uint8)
    seed_field[seed_ids[selected]]=labels[selected]
    seed_field=seed_field.reshape(tuple(shape))
    nearest=distance_transform_edt(seed_field==255,return_distances=False,return_indices=True)
    field=seed_field[tuple(nearest)]
    del nearest,seed_field
    # EDT gives a bounded-grid approximation quickly. Query the original
    # continuous samples throughout a two-cell band at every material change
    # so small visible markings are not limited by snapped surface seeds.
    changes=maximum_filter(field,size=3)!=minimum_filter(field,size=3)
    boundary=maximum_filter(changes,size=3)
    indices=np.flatnonzero(boundary)
    del changes,boundary
    flat=field.reshape(-1)
    for start in range(0, len(indices), 100_000):
        idx = indices[start:start+100_000]
        coords = np.column_stack(np.unravel_index(idx, tuple(shape))) * pitch + origin
        flat[idx] = labels[tree.query(coords, workers=2)[1]]
        if start==0 or start+100_000>=len(indices) or start%500_000==0:
            update(60, f'精确细化材料边界 · {min(start+100_000,len(indices)):,}/{len(indices):,} 点')
    remainder = source
    parts = []
    merged_count=0
    merged_volume=0.
    # Dominant color gets the remainder. Small colors carved first.
    order = sorted(range(len(colors)), key=lambda k: np.count_nonzero(labels == k))
    for position, color_id in enumerate(order):
        update(65 + int(25*position/len(order)), f'生成第 {position+1}/{len(order)} 个封闭分色实体')
        if position == len(order)-1:
            piece = remainder
        else:
            # Smooth sub-grid checkerboard transitions before extracting a
            # material region. Exact binary saddle points can create touching
            # edges that a slicer's vertex welding makes non-manifold. Only
            # internal/color boundaries change; the source exterior is kept.
            binary = gaussian_filter(np.pad((field == color_id).astype(np.float32), 2), sigma=.6)
            # A slight inward offset separates complementary color regions.
            # At exactly .5, their shared surface can leave zero-thickness
            # sheets in the final remainder after sequential subtraction.
            if binary.max() <= .505:
                continue
            vertices, faces, _, _ = marching_cubes(binary, level=0.505, spacing=(pitch,)*3)
            region = trimesh.Trimesh(vertices=vertices + origin - pitch*2, faces=faces, process=True)
            trimesh.repair.fix_normals(region, multibody=False)
            cutter = to_manifold(region)
            piece = remainder ^ cutter
            remainder = remainder - cutter
        if piece.is_empty():
            continue
        if min_island_mm>0:
            components=piece.decompose()
            small=[c for c in components if 0<c.volume()<min_island_mm**3]
            large=[c for c in components if not 0<c.volume()<min_island_mm**3]
            if small and position<len(order)-1:
                # Return sub-printable islands to the unassigned core. This
                # changes material ownership only, preserving the exterior.
                remainder=remainder+m3d.Manifold.batch_boolean(small,m3d.OpType.Add)
                piece=m3d.Manifold.batch_boolean(large,m3d.OpType.Add)
                merged_count+=len(small);merged_volume+=sum(c.volume() for c in small)
            elif small and parts:
                # A core sliver surrounded by a colored region is reassigned
                # to its nearest adjacent material, joining its shared faces.
                trees=[cKDTree(p.vertices) for p,_ in parts]
                for island in small:
                    island_mesh=from_manifold(island)
                    distances=[tree.query(island_mesh.vertices)[0].min() for tree in trees]
                    target=int(np.argmin(distances))
                    combined=to_manifold(parts[target][0])+island
                    candidate=from_manifold(combined)
                    if combined.status()!=m3d.Error.NoError or not candidate.is_watertight or not candidate.is_winding_consistent:
                        large.append(island);continue
                    parts[target]=(candidate,parts[target][1])
                    merged_count+=1;merged_volume+=island.volume()
                piece=m3d.Manifold.batch_boolean(large,m3d.OpType.Add)
            if piece.is_empty():
                continue
        if piece.status() != m3d.Error.NoError:
            raise ValueError('分色布尔运算未通过实体检查，请增大分色间距后重试。')
        part = from_manifold(piece)
        if not part.is_watertight or not part.is_winding_consistent:
            raise ValueError('分色结果存在非封闭部件，已阻止导出，请调整分色间距。')
        parts.append((part, colors[color_id]))
    actual_volume = sum(abs(p.volume) for p, _ in parts)
    relative_error = abs(actual_volume-abs(mesh.volume))/max(abs(mesh.volume), 1e-9)
    if relative_error > 0.001:
        raise ValueError('分体总容积与原模型不一致，未生成打印文件。')
    return parts, {'grid_points': size, 'pitch_mm': pitch,
                   'material_boundary_smoothing_sigma_mm': pitch*.6,
                   'material_region_isovalue': .505,
                   'material_field_method': 'surface_seed_edt_with_exact_boundary_queries',
                   'exact_boundary_queries': len(indices),
                   'surface_seed_max_snap_mm': pitch*np.sqrt(3)/2,
                   'merged_small_material_islands': merged_count,
                   'merged_material_volume_mm3': round(merged_volume,5),
                   'island_volume_threshold_mm3': min_island_mm**3,
                   'palette_method': 'flat_photo_palette' if fixed_palette is not None else 'balanced_lab_bins' if balanced else 'area_weighted_lab',
                   'partition_volume_relative_error': relative_error}


def export_3mf(parts, target):
    ns = 'http://schemas.microsoft.com/3dmanufacturing/core/2015/02'
    ET.register_namespace('', ns)
    def tag(name): return '{' + ns + '}' + name
    model = ET.Element(tag('model'), {'unit': 'millimeter', '{http://www.w3.org/XML/1998/namespace}lang': 'zh-CN'})
    ET.SubElement(model, tag('metadata'), {'name': 'Application'}).text = 'PhotoForm Local'
    resources = ET.SubElement(model, tag('resources'))
    materials = ET.SubElement(resources, tag('basematerials'), {'id': '1'})
    for i, (_, rgb) in enumerate(parts):
        ET.SubElement(materials, tag('base'), {'name': f'Color {i+1}', 'displaycolor': '#' + ''.join(f'{int(c):02X}' for c in rgb) + 'FF'})
    for i, (mesh, _) in enumerate(parts):
        obj = ET.SubElement(resources, tag('object'), {'id': str(i+2), 'type': 'model', 'pid': '1', 'pindex': str(i), 'name': f'Part {i+1}'})
        node = ET.SubElement(obj, tag('mesh'))
        verts = ET.SubElement(node, tag('vertices'))
        for v in mesh.vertices:
            ET.SubElement(verts, tag('vertex'), dict(zip(('x','y','z'), (f'{float(c):.7g}' for c in v))))
        faces = ET.SubElement(node, tag('triangles'))
        for f in mesh.faces:
            ET.SubElement(faces, tag('triangle'), dict(zip(('v1','v2','v3'), (str(int(c)) for c in f))))
    # One assembly ensures all parts preserve alignment in standard 3MF readers.
    assembly_id = str(len(parts)+2)
    assembly = ET.SubElement(resources, tag('object'), {'id': assembly_id, 'type': 'model', 'name': 'PhotoForm assembly'})
    components = ET.SubElement(assembly, tag('components'))
    for i in range(len(parts)):
        ET.SubElement(components, tag('component'), {'objectid': str(i+2)})
    build = ET.SubElement(model, tag('build'))
    ET.SubElement(build, tag('item'), {'objectid': assembly_id})
    # Bambu Studio preserves the assembly but ignores core material defaults
    # when assigning extruders. Its optional per-part metadata retains slots.
    config = ET.Element('config')
    config_object = ET.SubElement(config, 'object', {'id': assembly_id})
    ET.SubElement(config_object, 'metadata', {'key': 'name', 'value': 'PhotoForm assembly'})
    for i, (_, rgb) in enumerate(parts):
        color = '#' + ''.join(f'{int(c):02X}' for c in rgb)
        part = ET.SubElement(config_object, 'part', {'id': str(i+2), 'subtype': 'normal_part'})
        ET.SubElement(part, 'metadata', {'key': 'name', 'value': f'Part {i+1} {color}'})
        ET.SubElement(part, 'metadata', {'key': 'extruder', 'value': str(i+1)})
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('[Content_Types].xml', '<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/><Default Extension="config" ContentType="application/xml"/></Types>')
        z.writestr('_rels/.rels', '<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/></Relationships>')
        z.writestr('3D/3dmodel.model', ET.tostring(model, encoding='utf-8', xml_declaration=True))
        z.writestr('Metadata/model_settings.config', ET.tostring(config, encoding='utf-8', xml_declaration=True))


def make_demo(path):
    mesh = trimesh.creation.icosphere(subdivisions=4, radius=40)
    mesh.vertices[:, 2] *= 1.25
    face_centers = mesh.triangles_center
    colors = np.tile([24, 158, 133, 255], (len(mesh.faces), 1))
    colors[face_centers[:, 2] > 18] = [242, 176, 71, 255]
    colors[face_centers[:, 2] < -18] = [49, 95, 190, 255]
    mesh.visual.face_colors = colors.astype(np.uint8)
    mesh.export(path)


def export_preview(parts, path, exterior=None):
    """glTF material factors are linear RGB; printable palette/3MF are sRGB."""
    scene = trimesh.Scene()
    factors = {}
    surface_tree = cKDTree(exterior.triangles_center) if exterior is not None else None
    source_normals = exterior.vertex_normals if exterior is not None else None
    for i, (part, color) in enumerate(parts):
        name = f'part_{i+1:02d}_' + ''.join(f'{int(c):02x}' for c in color)
        rgb = np.asarray(color, dtype=float) / 255.
        linear = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
        factors[name] = [*linear.tolist(), 1.]
        preview = part.copy()
        if surface_tree is not None:
            # Internal material walls distort automatically averaged normals
            # along shared exterior edges. Interpolate the uncut source's
            # normals only at vertices verified to lie on that surface.
            # This changes display normals, never positions or print geometry.
            normals = preview.vertex_normals.copy()
            tolerance = max(exterior.extents)*1.e-6
            k = min(8, len(exterior.faces))
            for start in range(0,len(preview.vertices),10000):
                points = preview.vertices[start:start+10000]
                ids = surface_tree.query(points,k=k)[1].reshape(len(points),k)
                triangles = exterior.triangles[ids]
                closest = trimesh.triangles.closest_point(triangles.reshape(-1,3,3),np.repeat(points,k,axis=0)).reshape(len(points),k,3)
                distances = np.sum((closest-points[:,None])**2,axis=2)
                best = distances.argmin(axis=1); rows = np.arange(len(points))
                mask = distances[rows,best] <= tolerance**2
                if mask.any():
                    chosen = ids[rows,best][mask]
                    weights = trimesh.triangles.points_to_barycentric(exterior.triangles[chosen],closest[rows,best][mask])
                    smooth = np.sum(source_normals[exterior.faces[chosen]]*weights[:,:,None],axis=1)
                    smooth /= np.maximum(np.linalg.norm(smooth,axis=1,keepdims=True),1.e-12)
                    normals[start+np.flatnonzero(mask)] = smooth
        preview.visual = trimesh.visual.texture.TextureVisuals(
            material=trimesh.visual.material.PBRMaterial(
                name=name, baseColorFactor=[255,255,255,255],
                roughnessFactor=.8, metallicFactor=0.))
        if surface_tree is not None:
            preview.vertex_normals = normals
        scene.add_geometry(preview, node_name=name, geom_name=name)

    def correct_materials(tree):
        for material in tree.get('materials', []):
            material['pbrMetallicRoughness']['baseColorFactor'] = factors[material['name']]

    Path(path).write_bytes(scene.export(file_type='glb', tree_postprocessor=correct_materials))


def process_mesh(source, output, options, update, provenance=None):
    output = Path(output)
    output.mkdir(exist_ok=True)
    update(52, '检查网格、统一毫米尺寸')
    mesh, before = prepare_mesh(source, options['size_mm'], options.get('repair_small_holes', False))
    mesh.export(output / 'source_scaled.ply')
    update(57, '提取照片颜色，计算实体分区')
    parts, sampling = partition(mesh, options['colors'], options['pitch_mm'], update,
                                balanced=bool(provenance and provenance.get('color_method')),
                                fixed_palette=(provenance or {}).get('flat_palette_rgb'),
                                min_island_mm=options['min_feature_mm'] if options.get('merge_small_islands',True) else 0)
    update(92, '检查分体并写入 STL / 3MF')
    manifest = []
    for i, (part, color) in enumerate(parts):
        name = f'part_{i+1:02d}_' + ''.join(f'{int(c):02x}' for c in color)
        part.export(output / (name + '.stl'))
        entry = mesh_report(part)
        entry.update(name=name, color='#'+''.join(f'{int(c):02x}' for c in color), file=name+'.stl')
        # Component volume is a practical island indicator, not a wall-thickness guarantee.
        components = part.split(only_watertight=False)
        entry['small_islands'] = int(sum(abs(c.volume) < options['min_feature_mm']**3 for c in components))
        manifest.append(entry)
    export_preview(parts, output/'preview.glb',mesh)
    mesh.export(output / 'combined.stl')
    export_3mf(parts, output / 'colored.3mf')
    warnings = ['照片重建无绝对尺寸：这里以模型最长边缩放为设定的毫米数，请用实物尺寸校准。',
                '颜色分体按空间颜色区域划分，不是按头、手、衣服等语义分件；没有装配间隙或连接榫。',
                '已检查封闭性、法线与容积；未完成薄壁厚度、悬垂、支撑和打印机可制造性验证，请在切片预览中检查。',
                '3MF 存储部件和显示颜色；耗材槽位需在切片软件内确认和分配。']
    if any(p['small_islands'] for p in manifest):
        warnings.append('检测到小于所选最小特征体积的孤岛；请减少颜色数、调整参数，或在切片软件中检查。')
    if provenance:
        warnings.extend(provenance.get('warnings', []))
    report = {'source': before, 'final': mesh_report(mesh), 'parts': manifest,
              'generation': provenance,
              'sampling': sampling, 'options': options, 'warnings': warnings,
              'checks': {'all_parts_watertight': all(p['watertight'] for p in manifest),
                         'nonoverlap_by_construction': True, 'wall_thickness_verified': False,
                         'physical_accuracy_verified': False}}
    write_json(output / 'report.json', report)
    (output / 'READ_ME.txt').write_text('PhotoForm 本地彩色模型\n单位：毫米\n\n优先导入 colored.3mf，保留同一装配体中的颜色部件。若导入 STL，请同时选择全部 part_*.stl，并在切片软件中选择“作为一个对象的多个部件导入”；不要分别自动居中。\ncombined.stl 是单色完整模型。\n\n'+'\n'.join(warnings), 'utf-8')
    with zipfile.ZipFile(output / 'print_bundle.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for path in sorted(output.iterdir()):
            if path.suffix in ('.stl', '.3mf', '.json', '.txt'):
                z.write(path, path.name)
    return report
