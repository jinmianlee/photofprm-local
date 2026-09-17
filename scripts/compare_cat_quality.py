"""Render real exported geometry/materials beside the original cat photograph.

Private example data is not included in the repository. No image enhancement,
synthetic details or learned score: assess front, side and back directly.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image, ImageDraw
import trimesh

sys.path.insert(0,str(Path(__file__).resolve().parent))
from render_mesh_views import render


def load_print(folder):
    folder=Path(folder)
    report=json.loads((folder/'report.json').read_text('utf-8'))
    pieces=[]
    for part in report['parts']:
        mesh=trimesh.load(folder/part['file'],force='mesh')
        mesh.visual.vertex_colors=[*[int(part['color'][i:i+2],16) for i in (1,3,5)],255]
        pieces.append(mesh)
    model=trimesh.util.concatenate(pieces)
    model.apply_transform(trimesh.transformations.rotation_matrix(-np.pi/2,[1,0,0]))
    return model


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--photo',required=True)
    p.add_argument('--baseline',required=True,help='Baseline output folder')
    p.add_argument('--colored',required=True,help='Current actual printable output folder')
    p.add_argument('--shape21',help='Optional actual 2.1 raw PLY')
    p.add_argument('--output',required=True)
    a=p.parse_args()
    columns=[('Original photo',None),('Before: 3 materials',load_print(a.baseline)),
             ('After: 5 materials, calibrated',load_print(a.colored))]
    if a.shape21:
        columns.append(('2.1: geometry, no material yet',trimesh.load(a.shape21,force='mesh')))
    w=440; sheet=Image.new('RGB',(w*len(columns),w*2+100),'#efefef');draw=ImageDraw.Draw(sheet)
    pitch,yaw=np.deg2rad([25,7]);camera=[-np.sin(yaw)*np.cos(pitch),np.sin(pitch),np.cos(yaw)*np.cos(pitch)]
    for i,(title,mesh) in enumerate(columns):
        draw.text((i*w+15,12),title,fill='#222222')
        if mesh is None:
            photo=Image.open(a.photo).convert('RGBA');photo.thumbnail((w-40,w-40))
            background=Image.new('RGB',photo.size,'#efefef');background.paste(photo,mask=photo.getchannel('A'))
            sheet.paste(background,(i*w+(w-photo.width)//2,50+(w-photo.height)//2))
            draw.text((i*w+15,w+90),'Single photo: back is unknown',fill='#555555')
        else:
            mesh.apply_translation(-mesh.bounds.mean(0));mesh.apply_scale(1/max(mesh.extents))
            trimesh.repair.fix_normals(mesh)
            colored=i in [1,2]
            sheet.paste(render(mesh,camera,width=w,color=colored,unlit=colored),(i*w,50))
            if colored:
                # Separate closed material pieces include internal faces;
                # their display normals can create artificial color seams.
                # Inspect the actual combined exterior for shape comparison.
                folder=a.baseline if i==1 else a.colored
                geometry=trimesh.load(Path(folder)/'combined.stl',force='mesh')
                geometry.apply_transform(trimesh.transformations.rotation_matrix(-np.pi/2,[1,0,0]))
                geometry.apply_translation(-geometry.bounds.mean(0));geometry.apply_scale(1/max(geometry.extents))
            else:geometry=mesh
            sheet.paste(render(geometry,[1,.1,1.8],width=w,color=False),(i*w,w+70))
            draw.text((i*w+15,w+52),'Geometry: identical light/view',fill='#555555')
    output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True);sheet.save(output)
    print(output,flush=True)


if __name__=='__main__':main()
