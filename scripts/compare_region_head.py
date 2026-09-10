"""Show the head from whole-image inference next to independently generated head."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import trimesh
from render_mesh_views import render

ROOT = Path(__file__).resolve().parents[1]
original = trimesh.load(ROOT/'data/2412cfadbbde4c04a068dfd582f05a6a/generated_raw.ply', force='mesh')
threshold = original.bounds[0,1]+original.extents[1]*.65
original = original.submesh([np.flatnonzero(original.triangles_center[:,1]>threshold)], append=True)
local = trimesh.load(ROOT/'data/detail-head-only/generated_raw.ply', force='mesh')
# Display both head/neck regions at the same height, without deforming either mesh.
for mesh in (original, local):
    mesh.apply_translation(-mesh.bounds.mean(0))
    mesh.apply_scale(1/mesh.extents[1])
frame = np.array([[-.64,-.57,-.64],[.64,.57,.64]])
sheet = Image.new('RGB', (1400,1510), '#efefef')
draw = ImageDraw.Draw(sheet)
font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 24)
for col,(mesh,title) in enumerate([(original,'整图模型中的头部 · 放大显示'),(local,'头部单独生成 · 独立模型')]):
    draw.text((col*700+20,18),title,font=font,fill='#222222')
    for row,camera in enumerate([[0,0,1],[1,.12,1.8]]):
        sheet.paste(render(mesh,camera,width=700,frame_bounds=frame),(col*700,65+row*710))
draw.text((20,1480),'相同视角与灯光；两侧独立居中、等比缩放显示；未拼接身体。',font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18),fill='#333333')
sheet.save(ROOT/'data/detail-head-only/head-comparison.png')
print(ROOT/'data/detail-head-only/head-comparison.png')
