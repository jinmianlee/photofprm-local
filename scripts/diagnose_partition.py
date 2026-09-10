"""Inspect precision at the manifold/float STL boundary for the real cat."""
from pathlib import Path
import sys
import io
import numpy as np
import trimesh
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend import geometry

def inspect(solid):
    raw = solid.to_mesh64()
    bare = trimesh.Trimesh(vertices=np.asarray(raw.vert_properties)[:, :3], faces=np.asarray(raw.tri_verts), process=False)
    print('BARE', bare.is_watertight, len(bare.vertices), 'merges',len(raw.merge_from_vert),
          'degenerate', sum(~bare.nondegenerate_faces()), 'components',len(solid.decompose()), flush=True)
    if not bare.is_watertight:
        print('MERGE ATTRS', [n for n in dir(raw) if 'merge' in n], flush=True)
    else:
        weld = bare.copy(); weld.merge_vertices()
        if not weld.is_watertight:
            np.savez(ROOT / 'data/cat-color-angle25-yaw7/touching-part.npz', vertices=bare.vertices,faces=bare.faces)
            print('SAVED touching part', flush=True)
    for tolerance in (0, .00001, .0001, .001):
        candidate = solid if not tolerance else solid.simplify(tolerance)
        for precision in (32, 64):
            raw = candidate.to_mesh() if precision == 32 else candidate.to_mesh64()
            mesh = trimesh.Trimesh(vertices=np.asarray(raw.vert_properties)[:, :3],
                                   faces=np.asarray(raw.tri_verts), process=True)
            degenerate = int(sum(~mesh.nondegenerate_faces()))
            mesh.update_faces(mesh.nondegenerate_faces() & mesh.unique_faces())
            mesh.remove_unreferenced_vertices()
            counts = np.bincount(mesh.edges_unique_inverse)
            reloaded = trimesh.load(io.BytesIO(mesh.export(file_type='stl')), file_type='stl', process=True)
            reloaded.update_faces(reloaded.nondegenerate_faces() & reloaded.unique_faces())
            print({'tolerance': tolerance, 'precision': precision, 'faces': len(mesh.faces),
                   'removed_degenerate': degenerate, 'boundary': int(sum(counts == 1)),
                   'nonmanifold_edges': int(sum(counts > 2)), 'watertight': mesh.is_watertight,
                   'stl_watertight_after_cleanup': reloaded.is_watertight}, flush=True)
            if mesh.is_watertight and reloaded.is_watertight:
                return mesh
    raise ValueError('No validated conversion')

geometry.from_manifold = inspect
source = ROOT / 'data/cat-color-angle25-yaw7/source.ply'
mesh, _ = geometry.prepare_mesh(source, 95)
parts, report = geometry.partition(mesh, 4, .8, lambda *x: print(x, flush=True), balanced=True)
print(report)
