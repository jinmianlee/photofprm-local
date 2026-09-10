"""Compare actual generated geometry; common framing, lighting and no texture."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import trimesh
from render_mesh_views import render


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline')
    parser.add_argument('candidate')
    parser.add_argument('output')
    parser.add_argument('--label', default='局部放大编码 · 强度 0.35')
    args = parser.parse_args()
    meshes = [trimesh.load(path, force='mesh', process=True)
              for path in (args.baseline, args.candidate)]
    whole_bounds = np.stack([np.minimum(meshes[0].bounds[0], meshes[1].bounds[0]),
                             np.maximum(meshes[0].bounds[1], meshes[1].bounds[1])])
    threshold = meshes[0].bounds[0, 1]+meshes[0].extents[1]*.65
    heads = [mesh.submesh([np.flatnonzero(mesh.triangles_center[:, 1] > threshold)], append=True)
             for mesh in meshes]
    head_bounds = np.stack([np.minimum(heads[0].bounds[0], heads[1].bounds[0]),
                           np.maximum(heads[0].bounds[1], heads[1].bounds[1])])
    width, row_height = 640, 690
    sheet = Image.new('RGB', (width*2, row_height*3+65), '#efefef')
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 22)
    for column, title in enumerate(['原版混元 Mini Turbo', args.label]):
        draw.text((column*width+20, 15), title, fill='#202020', font=font)
    for row, (label, group, bounds, camera) in enumerate([
        ('整体 · 正面', meshes, whole_bounds, [0, 0, 1]),
        ('头部 · 正面', heads, head_bounds, [0, 0, 1]),
        ('头部 · 侧前方', heads, head_bounds, [1, .12, 1.8]),
    ]):
        for column, mesh in enumerate(group):
            y = 65+row*row_height
            draw.text((column*width+20, y), label, fill='#202020', font=font)
            sheet.paste(render(mesh, camera, width=width, frame_bounds=bounds), (column*width, y+40))
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination)
    records = []
    for mesh, path in zip(meshes, [args.baseline, args.candidate]):
        records.append({'path': str(Path(path).resolve()), 'faces': len(mesh.faces),
                        'extents': mesh.extents.tolist(), 'watertight_raw': bool(mesh.is_watertight),
                        'winding_consistent': bool(mesh.is_winding_consistent),
                        'finite_vertices': bool(np.isfinite(mesh.vertices).all())})
    destination.with_suffix('.json').write_text(json.dumps({
        'comparison': 'same 255 extraction resolution, seed and steps; common camera bounds; no texture',
        'geometry_counts_are_not_quality_scores': True,
        'meshes': records,
    }, indent=2), 'utf-8')
    print(destination, flush=True)


if __name__ == '__main__':
    main()
