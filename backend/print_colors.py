"""Small, flat photo palettes for filament printing, with reduced shading weight.

This is color segmentation, not a semantic or intrinsic-image neural model.
It retains a luminance term to distinguish black/white marks from colored areas.
"""
import numpy as np
from PIL import Image
from scipy.cluster.vq import kmeans2
from scipy.ndimage import binary_closing, binary_dilation, median_filter, label
from scipy.spatial import cKDTree
from skimage.color import rgb2lab, lab2rgb
import trimesh


def flat_palette(photo, count):
    image = Image.fromarray(photo).copy()
    image.thumbnail((768, 768))
    pixels = np.asarray(image)
    alpha = pixels[..., 3] >= 128
    rgb = pixels[..., :3].astype(float) / 255
    # Suppress pixel noise without allowing background RGB into the palette.
    smooth = np.stack([median_filter(rgb[..., c], size=3) for c in range(3)], axis=-1)
    labs = rgb2lab(smooth)
    # Multiplicative light/shadow changes RGB magnitudes much more than their
    # ratios. Normalize chroma before clustering, with a bounded gain so nearly
    # black sensor noise is not amplified into a strong material color.
    gain = np.minimum(4., .75/np.maximum(smooth.max(axis=-1), .001))
    features = rgb2lab(np.clip(smooth*gain[..., None], 0, 1))
    # Most broad brightness changes should stay in the same material, while
    # the color axes retain their full separation (skin versus grey clothing).
    features[..., 0] = labs[..., 0] * .28
    training = features[alpha]
    if not len(training):
        raise ValueError('照片前景为空。')
    bins, frequencies = np.unique(np.round(training/2)*2, axis=0, return_counts=True)
    samples = bins[frequencies >= 4]
    if not len(samples):
        samples = training[::max(1, len(training)//30000)]
    count = min(count, len(np.unique(np.round(samples, 1), axis=0)))
    # Rare dark markings (pupils, nose tips, lettering) must not compete with
    # dozens of fur/shadow chroma bins for the same material slot. Do not
    # normalize them towards an illuminated brown before quantization.
    dark = alpha & (labs[..., 0] < 24) & (smooth.max(axis=-1) < .27)
    preserve_dark = count >= 2 and dark.sum() >= max(16, alpha.sum()*.001) and np.any(alpha & ~dark)
    if preserve_dark:
        regular = np.unique(np.round(features[alpha & ~dark]/2)*2, axis=0)
        count = min(count, 1+len(regular))
    if count == 1:
        centers = samples.mean(axis=0, keepdims=True)
    elif preserve_dark:
        dark_center = np.median(features[dark], axis=0, keepdims=True)
        if count == 2:
            centers = np.vstack((dark_center, np.median(regular, axis=0)))
        else:
            others, _ = kmeans2(regular, count-1, minit='++', iter=40, seed=17)
            centers = np.vstack((dark_center, others))
    else:
        centers, _ = kmeans2(samples, count, minit='++', iter=40, seed=17)
    labels = cKDTree(centers).query(features.reshape(-1, 3))[1].reshape(alpha.shape)
    labels = median_filter(labels.astype(np.uint8), size=3)
    if preserve_dark:
        labels[dark] = 0
    # Merge small medium-brightness islands such as glare on hair or a shadow
    # patch on a shirt. Very dark marks and true bright white details survive.
    # This is bounded cleanup of color regions; it does not smooth the geometry.
    # A percentage as large as 1.2% erased forehead stripes and eyelid marks
    # in a portrait. Remove sensor specks, not recognizable small markings.
    min_area = max(12, int(alpha.sum()*.0002))
    cleaned = labels.copy()
    for index in range(len(centers)):
        regions, number = label((labels == index) & alpha)
        areas = np.bincount(regions.ravel()); areas[0] = 0
        largest = int(np.argmax(areas))
        for region_id in range(1, number+1):
            if region_id == largest or areas[region_id] >= min_area:
                continue
            region = regions == region_id
            brightness = float(np.median(labs[..., 0][region]))
            if brightness < 25 or brightness > 87:
                continue
            boundary = binary_dilation(region, iterations=2) & ~region & alpha
            alternatives = labels[boundary & (labels != index)]
            if len(alternatives):
                cleaned[region] = np.argmax(np.bincount(alternatives, minlength=len(centers)))
    labels = cleaned
    palette = []
    for index in range(len(centers)):
        selected = labs[alpha & (labels == index)]
        if not len(selected):
            selected = labs[alpha]
        center = np.median(selected, axis=0)
        # Prefer an illuminated representative of each material, not its shadow.
        center[0] = np.median(selected[:, 0]) if preserve_dark and index == 0 else np.percentile(selected[:, 0], 85)
        palette.append(np.rint(np.clip(lab2rgb(center[None, None])[0, 0], 0, 1)*255).astype(np.uint8))
    labels = np.asarray(Image.fromarray(labels).resize((photo.shape[1], photo.shape[0]), Image.Resampling.NEAREST))
    return labels, np.asarray(palette)


def projection_matrix(pitch, yaw):
    return (trimesh.transformations.rotation_matrix(np.deg2rad(pitch), [1, 0, 0])[:3, :3]
            @ trimesh.transformations.rotation_matrix(np.deg2rad(yaw), [0, 1, 0])[:3, :3])


def estimate_alignment(mesh, alpha):
    """Approximate silhouette fit; does not claim to recognize facial landmarks."""
    yy, xx = np.where(alpha)
    crop = Image.fromarray(alpha[yy.min():yy.max()+1, xx.min():xx.max()+1])
    target = np.asarray(crop.resize((192, 192), Image.Resampling.NEAREST))
    aspect = (xx.max()-xx.min()+1)/(yy.max()-yy.min()+1)
    vertices = mesh.vertices[::max(1, len(mesh.vertices)//120000)]
    def score(pitch, yaw):
        points = vertices @ projection_matrix(pitch, yaw).T
        lo, hi = points.min(0), points.max(0)
        extent = hi-lo
        px = np.clip(np.rint((points[:, 0]-lo[0])/extent[0]*191).astype(int), 0, 191)
        py = np.clip(np.rint((hi[1]-points[:, 1])/extent[1]*191).astype(int), 0, 191)
        mask = np.zeros((192, 192), dtype=bool)
        mask[py, px] = True
        mask = binary_closing(binary_dilation(mask), border_value=1)
        iou = np.count_nonzero(mask & target)/max(1, np.count_nonzero(mask | target))
        return iou - .08*abs(np.log((extent[0]/extent[1])/aspect))
    candidates = [(score(p, y), p, y) for p in (-15, 0, 15, 30) for y in (-20, 0, 20)]
    _, pitch, yaw = max(candidates)
    candidates += [(score(p, y), p, y) for p in (pitch-7.5, pitch, pitch+7.5)
                   for y in (yaw-10, yaw, yaw+10)]
    quality, pitch, yaw = max(candidates)
    return pitch, yaw, round(quality, 4)


def project_flat_colors(mesh, photo, count, pitch, yaw, automatic=True, palette_override=None, landmarks=None):
    labels, palette = flat_palette(photo, count)
    automatic_palette = palette.copy()
    if palette_override:
        import re
        if len(palette_override) != len(palette) or any(not re.fullmatch(r'#[0-9a-fA-F]{6}', value) for value in palette_override):
            raise ValueError('自选材料色数量须与生成的色区数量一致，每个颜色使用 #RRGGBB。')
        palette = np.array([[int(value[i:i+2], 16) for i in (1, 3, 5)] for value in palette_override], dtype=np.uint8)
    alpha = photo[..., 3] >= 128
    fit = None
    if automatic and not landmarks:
        pitch, yaw, fit = estimate_alignment(mesh, alpha)
    rotation = projection_matrix(pitch, yaw)
    projected = mesh.vertices @ rotation.T
    lo, hi = projected.min(0), projected.max(0)
    extent = hi-lo
    yy, xx = np.where(alpha)
    affine = None
    if landmarks:
        source = np.array([v['model'] for v in landmarks]) @ rotation.T
        uv = np.column_stack(((source[:, 0]-lo[0])/extent[0], (hi[1]-source[:, 1])/extent[1], np.ones(len(source))))
        target = np.array([v['photo'] for v in landmarks]) * np.array([photo.shape[1]-1, photo.shape[0]-1])
        if np.linalg.matrix_rank(uv) < 3 or np.linalg.cond(uv) > 200:
            raise ValueError('校准点过于接近或几乎共线，请重新标记两眼与鼻尖。')
        affine = np.linalg.lstsq(uv, target, rcond=None)[0]
        if np.linalg.det(affine[:2]) <= 0:
            raise ValueError('照片与模型的左右对应顺序不一致，请重新标记校准点。')
    def coordinates(points):
        if affine is not None:
            uv = np.column_stack(((points[:, 0]-lo[0])/extent[0], (hi[1]-points[:, 1])/extent[1], np.ones(len(points))))
            px, py = (uv @ affine).T
        else:
            px = (points[:, 0]-lo[0])/extent[0]*(xx.max()-xx.min())+xx.min()
            py = (hi[1]-points[:, 1])/extent[1]*(yy.max()-yy.min())+yy.min()
        return np.clip(np.rint(px).astype(int), 0, photo.shape[1]-1), np.clip(np.rint(py).astype(int), 0, photo.shape[0]-1)
    # Surface-center depth prevents hidden folds receiving colors through the body.
    centers = mesh.triangles_center @ rotation.T
    ix, iy = coordinates(centers)
    depth = np.full(alpha.shape, -np.inf)
    np.maximum.at(depth, (iy, ix), centers[:, 2])
    from scipy.ndimage import maximum_filter
    depth = maximum_filter(depth, size=3)
    normals = mesh.face_normals @ rotation.T
    visible = (centers[:, 2] >= depth[iy, ix]-extent[2]*.025) & alpha[iy, ix] & (normals[:, 2] > .12)
    assigned = labels[iy, ix].copy()
    if not visible.any():
        raise ValueError('无法对齐照片颜色，请切换手动视角。')
    # Unseen sides and back use the dominant material. A single photograph
    # provides no evidence for continuing eye, nose or stripe markings there.
    dominant = int(np.argmax(np.bincount(labels[alpha], minlength=len(palette))))
    assigned[~visible] = dominant
    mesh.visual.face_colors = np.column_stack((palette[assigned], np.full(len(assigned), 255, dtype=np.uint8)))
    return {'color_method': 'flat_photo_palette_shading_suppressed',
            'flat_palette_rgb': palette.tolist(), 'automatic_palette_rgb': automatic_palette.tolist(),
            'photo_pitch_deg': pitch, 'photo_yaw_deg': yaw,
            'alignment_method': 'paired_landmarks_affine' if landmarks else 'approximate_silhouette' if automatic else 'manual', 'alignment_score': fit,
            'landmark_count': len(landmarks or []),
            'projected_face_fraction': round(float(np.mean(visible)), 4),
            'luminance_clustering_weight': .28, 'chroma_normalization': 'bounded_rgb_ratio'}
