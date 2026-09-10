"""Render actual mesh geometry on CPU for front/side/back visual inspection."""
import argparse
from pathlib import Path
import numpy as np
from numba import njit
from PIL import Image, ImageDraw
import trimesh


@njit(cache=True)
def rasterize(vertices, faces, normals, rgb, width, unlit=False):
    canvas = np.full((width, width, 3), 239, dtype=np.uint8)
    depth = np.full((width, width), -1.e20, dtype=np.float32)
    light = np.array([-0.35, 0.65, 0.68])
    for ids in faces:
        a, b, c = vertices[ids[0]], vertices[ids[1]], vertices[ids[2]]
        min_x = max(0, int(np.floor(min(a[0], b[0], c[0]))))
        max_x = min(width - 1, int(np.ceil(max(a[0], b[0], c[0]))))
        min_y = max(0, int(np.floor(min(a[1], b[1], c[1]))))
        max_y = min(width - 1, int(np.ceil(max(a[1], b[1], c[1]))))
        denominator = (b[1]-c[1])*(a[0]-c[0]) + (c[0]-b[0])*(a[1]-c[1])
        if abs(denominator) < 1.e-8:
            continue
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                u = ((b[1]-c[1])*(x+.5-c[0]) + (c[0]-b[0])*(y+.5-c[1])) / denominator
                v = ((c[1]-a[1])*(x+.5-c[0]) + (a[0]-c[0])*(y+.5-c[1])) / denominator
                w = 1 - u - v
                if min(u, v, w) < 0:
                    continue
                z = u*a[2] + v*b[2] + w*c[2]
                if z <= depth[y, x]:
                    continue
                depth[y, x] = z
                normal = u*normals[ids[0]] + v*normals[ids[1]] + w*normals[ids[2]]
                normal /= max(np.linalg.norm(normal), 1.e-9)
                diffuse = max(0., np.dot(normal, light))
                shade = 1. if unlit else .36 + .56*diffuse + .08*max(0., normal[2])
                color = u*rgb[ids[0]] + v*rgb[ids[1]] + w*rgb[ids[2]]
                canvas[y, x] = np.minimum(color * shade, 255).astype(np.uint8)
    return canvas


def render(mesh, camera, width=512, color=False, unlit=False, frame_bounds=None):
    forward = np.array(camera, dtype=float)
    forward /= np.linalg.norm(forward)
    right = np.cross([0, 1, 0], forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    basis = np.column_stack((right, up, forward))
    bounds = mesh.bounds if frame_bounds is None else np.asarray(frame_bounds)
    vertices = (mesh.vertices - bounds.mean(0)) @ basis
    vertices[:, :2] *= .84 * width / max(bounds[1]-bounds[0])
    vertices[:, 0] += width/2
    vertices[:, 1] = width/2 - vertices[:, 1]
    normals = np.asarray(mesh.vertex_normals) @ basis
    colors = np.asarray(mesh.visual.vertex_colors[:, :3], dtype=float) if color else np.full_like(mesh.vertices, 218.)
    return Image.fromarray(rasterize(vertices, np.asarray(mesh.faces), normals, colors, width, unlit))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mesh')
    parser.add_argument('output')
    parser.add_argument('--color', action='store_true')
    parser.add_argument('--unlit', action='store_true')
    parser.add_argument('--up-axis', choices=['y', 'z'], default='y')
    args = parser.parse_args()
    mesh = trimesh.load(args.mesh, force='mesh', process=True)
    if args.up_axis == 'z':
        mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi/2, [1, 0, 0]))
    trimesh.repair.fix_normals(mesh)
    views = [('Front +Z', [0, 0, 1]), ('Front quarter', [1, .12, 1.8]),
             ('Right +X', [1, 0, 0]), ('Back -Z', [0, 0, -1])]
    sheet = Image.new('RGB', (1024, 1112), '#efefef')
    draw = ImageDraw.Draw(sheet)
    for index, (name, camera) in enumerate(views):
        x, y = index % 2 * 512, index // 2 * 556
        sheet.paste(render(mesh, camera, color=args.color, unlit=args.unlit), (x, y+36))
        draw.text((x+18, y+12), name, fill='#222222')
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination)
    print(destination, flush=True)


if __name__ == '__main__':
    main()
