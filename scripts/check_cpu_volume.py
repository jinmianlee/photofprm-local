"""Check spatial sampling against a known off-centre sphere, without AI weights."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from skimage.measure import marching_cubes
from backend.cpu_volume import CPUVolumeDecoder

torch.set_num_threads(4)
center = torch.tensor([0.19, -0.12, 0.23])
radius = 0.48
def sphere(queries, latents):
    return (40 * (radius - torch.linalg.vector_norm(queries - center, dim=-1)))[..., None]

grid = CPUVolumeDecoder()(torch.zeros(1, 1, 1), sphere,
                         octree_resolution=63, min_resolution=15, num_chunks=4096)
assert torch.isfinite(grid).all()
vertices, faces, _, _ = marching_cubes(grid[0].numpy(), 0)
vertices = vertices / 63 * 2.02 - 1.01
radial_error = np.abs(np.linalg.norm(vertices - center.numpy(), axis=1) - radius)
assert radial_error.max() < 0.003, radial_error.max()
assert np.max(np.abs((vertices.min(0) + vertices.max(0))/2 - center.numpy())) < 0.003
print(f'CPU_VOLUME_OK: {len(faces)} faces, max sphere error {radial_error.max():.6f}')
