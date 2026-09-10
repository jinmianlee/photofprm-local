"""Render actual before/after print meshes with identical cameras and lighting."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import trimesh
from render_mesh_views import render


def load(path):
    mesh = trimesh.load(path, force='mesh', process=True)
    mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi/2, [1, 0, 0]))
    trimesh.repair.fix_normals(mesh)
    return mesh


def main():
    root = Path(__file__).resolve().parents[1]
    record = json.loads((root/'data/flat-print-validation.json').read_text('utf-8'))
    before = root/'data'/record['original_job']/'output'
    after = root/'data'/record['job_id']/'output'
    sheet = Image.new('RGB', (1440, 1600), '#efefef')
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 26)
    captions = [('原版：照片明暗分成 4 色', before), ('新版：3 种示例耗材色', after)]
    for column, (caption, directory) in enumerate(captions):
        colored = load(directory/'preview.glb')
        sheet.paste(render(colored, [0, 0, 1], width=720, color=True, unlit=True), (column*720, 55))
        draw.text((column*720+25, 15), caption, font=font, fill='#252525')
        geometry = load(directory/'combined.stl')
        threshold = geometry.bounds[0, 1]+geometry.extents[1]*.65
        face_ids = np.flatnonzero(geometry.triangles_center[:, 1] > threshold)
        head = geometry.submesh([face_ids], append=True)
        sheet.paste(render(head, [0, .03, 1], width=720, color=False), (column*720, 850))
        draw.text((column*720+25, 802), '几何局部 · '+('标准 255' if column==0 else '精细 383'), font=font, fill='#252525')
    target = root/'data/flat-print-comparison.png'
    sheet.save(target)
    print(target)


if __name__ == '__main__':
    main()
