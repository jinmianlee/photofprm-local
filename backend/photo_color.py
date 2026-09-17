"""Prepare a Y-up, +Z-facing generated bust and project its observed photo colors."""
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates, maximum_filter
import trimesh
from .geometry import to_manifold, from_manifold
from .settings import write_json


def repair_tiny_holes(mesh, size_mm):
    """Opt-in triangle/quad patches, bounded in final print units; no remeshing."""
    candidate = mesh.copy()
    before = len(mesh.faces)
    trimesh.repair.fill_holes(candidate)
    count = len(candidate.faces)-before
    if not 0 < count <= 32 or not candidate.is_watertight:
        raise ValueError('小孔修复未得到封闭表面，不能自动填补这处缺失。')
    if not np.array_equal(candidate.vertices, mesh.vertices) or not np.array_equal(candidate.faces[:before], mesh.faces):
        raise ValueError('小孔修复改变了原有表面，已停止。')
    patch = candidate.triangles[before:]
    spans = np.linalg.norm(patch-np.roll(patch, 1, axis=1), axis=-1)
    max_span_mm = float(spans.max()/mesh.extents.max()*size_mm)
    if max_span_mm > .8 or candidate.area_faces[before:].sum() > mesh.area*.001:
        raise ValueError('缺口超过微小孔修复范围（跨度 0.8 mm、总表面积 0.1%），请使用建模工具检查。')
    return candidate, {'added_faces':count, 'max_patch_edge_mm':round(max_span_mm, 4),
                       'existing_vertices_unchanged':True}


def prepare_colored_bust(raw_path, foreground_path, destination, photo_pitch_deg=25, photo_yaw_deg=7,
                         color_style='photo', colors=4, photo_auto_align=True, palette_override=None,
                         repair_small_holes=False, size_mm=95, photo_landmarks=None,
                         model_name='Hunyuan3D-2mini-Turbo'):
    if not (-45 <= photo_pitch_deg <= 60 and -60 <= photo_yaw_deg <= 60):
        raise ValueError('照片投影角度超出范围。')
    mesh = trimesh.load(raw_path, force='mesh', process=True)
    if not np.isfinite(mesh.vertices).all():
        raise ValueError('生成表面存在无效坐标，已保留原始网格供检查。')
    # Marching cubes can emit zero-area triangles where a surface crosses a
    # grid vertex exactly. Remove those and duplicate faces without filling
    # holes or changing the exterior positions.
    faces_before = len(mesh.faces)
    mesh.update_faces(mesh.nondegenerate_faces() & mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    removed_faces = faces_before - len(mesh.faces)
    hole_repair = None
    if not mesh.is_watertight and repair_small_holes:
        mesh, hole_repair = repair_tiny_holes(mesh, size_mm)
    if not mesh.is_watertight:
        raise ValueError('生成表面存在开放边界，已保留原始网格。可在“更多打印检查参数”开启小孔修复后恢复任务；大范围缺失仍会被阻止。')
    if mesh.extents.min() / max(mesh.extents.max(), 1.e-9) < .10:
        raise ValueError('生成结果过薄，没有形成可靠的立体头胸像；已阻止打印导出并保留原始网格。')
    trimesh.repair.fix_normals(mesh, multibody=True)
    components = sorted(mesh.split(), key=lambda part: abs(part.volume), reverse=True)
    if not components:
        raise ValueError('生成结果没有封闭实体。')
    total_volume = sum(abs(part.volume) for part in components)
    if abs(components[0].volume) < .65 * total_volume:
        raise ValueError('生成结果过于破碎，没有形成明确主体；已阻止打印导出。')
    # Keep substantial separate components; remove only numerical specks.
    cutoff = abs(components[0].volume) * 0.0001
    kept = [part for part in components if abs(part.volume) >= cutoff]
    mesh = trimesh.util.concatenate(kept)
    original_volume = abs(mesh.volume)
    cut_y = float(mesh.bounds[0, 1] + mesh.extents[1] * 0.015)
    solid = to_manifold(mesh).trim_by_plane((0, 1, 0), cut_y)
    mesh = from_manifold(solid)
    if not len(mesh.faces) or not mesh.is_watertight:
        raise ValueError('底面处理未得到封闭实体。')
    trimmed_fraction = 1 - abs(mesh.volume) / original_volume
    if not -0.001 <= trimmed_fraction < 0.05:
        raise ValueError('底面裁切影响范围过大，请检查原始模型。')

    photo = np.asarray(Image.open(foreground_path).convert('RGBA'))
    if color_style == 'flat':
        from .print_colors import project_flat_colors
        provenance = project_flat_colors(mesh, photo, colors, photo_pitch_deg, photo_yaw_deg, photo_auto_align, palette_override, photo_landmarks)
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0]))
        mesh.export(destination)
        provenance.update(model=model_name, local=True,
            small_hole_repair=hole_repair,
            removed_numerical_components=len(components)-len(kept),
            removed_degenerate_or_duplicate_faces=removed_faces, base_trim_volume_fraction=round(trimmed_fraction, 6),
            warnings=['纯色模式减弱明暗参与分色，不是头、手、衣服的语义识别；真实深色花纹也可能被合并。',
                      '未见表面使用主体材料色，避免眼鼻色块绕到背面；背面花纹仍需要额外照片。',
                      '单张照片不能保证实物相似度或打印精度；底面已小幅裁平。'])
        write_json(destination.parent / 'generation.json', provenance)
        return provenance
    if color_style != 'photo':
        raise ValueError('未知上色模式。')
    alpha = photo[..., 3] >= 128
    yy, xx = np.where(alpha)
    if not len(xx):
        raise ValueError('照片前景为空。')
    rgb = photo[..., :3].astype(float)
    base = np.median(rgb[alpha], axis=0)
    # Orthographic alignment is approximate. Hidden surfaces use the dominant
    # foreground color; dark eyes and furniture must not wrap around the back.
    # Defaults were calibrated against the supplied cat photo. Expose both
    # angles for other photos; this is not an automatic camera-pose estimate.
    projection = (trimesh.transformations.rotation_matrix(np.deg2rad(photo_pitch_deg), [1, 0, 0])[:3, :3]
                  @ trimesh.transformations.rotation_matrix(np.deg2rad(photo_yaw_deg), [0, 1, 0])[:3, :3])
    projected = mesh.vertices @ projection.T
    bounds = np.array([projected.min(0), projected.max(0)])
    extent = np.ptp(projected, axis=0)
    px = (projected[:, 0] - bounds[0, 0]) / extent[0] * (xx.max()-xx.min()) + xx.min()
    py = (bounds[1, 1] - projected[:, 1]) / extent[1] * (yy.max()-yy.min()) + yy.min()
    ix = np.clip(np.rint(px).astype(int), 0, photo.shape[1]-1)
    iy = np.clip(np.rint(py).astype(int), 0, photo.shape[0]-1)
    depth = np.full(alpha.shape, -np.inf)
    np.maximum.at(depth, (iy, ix), projected[:, 2])
    depth = maximum_filter(depth, size=5)
    visible = (projected[:, 2] >= depth[iy, ix] - extent[2]*0.035) & alpha[iy, ix]
    frontness = (mesh.vertex_normals @ projection.T)[:, 2]
    blend = np.clip((frontness-.12)/.42, 0, 1) * visible
    sampled = np.column_stack([map_coordinates(rgb[..., channel], [py, px], order=1, mode='nearest')
                               for channel in range(3)])
    colors = base[None] * (1-blend[:, None]) + sampled * blend[:, None]
    mesh.visual.vertex_colors = np.column_stack((np.rint(colors).astype(np.uint8),
                                                 np.full(len(colors), 255, dtype=np.uint8)))
    # Convert the model's Y-up frame to the printer's Z-up frame.
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0]))
    mesh.export(destination)
    provenance = {
        'small_hole_repair': hole_repair,
        'model': model_name, 'local': True,
        'color_method': 'front_photo_projection_with_dominant_color_on_unseen_surfaces',
        'photo_pitch_deg': photo_pitch_deg,
        'photo_yaw_deg': photo_yaw_deg,
        'base_color_rgb': np.rint(base).astype(int).tolist(),
        'projected_vertex_fraction': round(float(np.mean(blend > 0)), 4),
        'removed_numerical_components': len(components)-len(kept),
        'removed_degenerate_or_duplicate_faces': removed_faces,
        'base_trim_volume_fraction': round(trimmed_fraction, 6),
        'warnings': [
            '单张照片的侧面、背面和遮挡结构由模型推断，不代表实物扫描精度。',
            '正面颜色来自照片投影；看不见的表面使用前景主色。请检查眼鼻位置和颜色边界。',
            '底部已小幅裁平；极小的数值碎片已移除。照片中的细毛和胡须不保证形成可打印几何。',
        ],
    }
    write_json(destination.parent / 'generation.json', provenance)
    return provenance
