import argparse
import json
import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

parser = argparse.ArgumentParser()
parser.add_argument('mesh')
args = parser.parse_args()
mesh = trimesh.load(args.mesh, force='mesh', process=True)
before = len(mesh.faces)
mesh.update_faces(mesh.nondegenerate_faces() & mesh.unique_faces())
mesh.remove_unreferenced_vertices()
counts = np.bincount(mesh.edges_unique_inverse)
edges = mesh.edges_unique[counts == 1]
lengths = np.linalg.norm(mesh.vertices[edges[:, 1]]-mesh.vertices[edges[:, 0]], axis=1)
graph = coo_matrix((np.ones(len(edges)), (edges[:,0], edges[:,1])), shape=(len(mesh.vertices),)*2)
_, labels = connected_components(graph, directed=False)
loops=[]
for group in np.unique(labels[np.unique(edges)]):
    ids = np.unique(edges[labels[edges[:,0]]==group])
    coords = mesh.vertices[ids]
    loops.append({'vertices':len(ids),'diameter_bbox':float(np.linalg.norm(np.ptp(coords,axis=0))),
                  'bounds':[coords.min(0).tolist(), coords.max(0).tolist()],
                  'coordinates':coords.tolist() if len(ids) <= 8 else None})
print(json.dumps({'removed_faces':before-len(mesh.faces),'boundary_edges':len(edges),
    'nonmanifold_edges':int((counts>2).sum()), 'edge_length_range':[float(lengths.min()),float(lengths.max())] if len(edges) else [],
    'boundary_components':loops},indent=2))
