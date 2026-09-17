"""Reuse finished geometry only when its image, settings and bytes match."""
import hashlib
import json
from pathlib import Path
import struct
if __package__:
    from .file_io import write_json
else:
    from file_io import write_json

WEIGHT_SHA = 'bdbcef30dd0149a281e17d5b5b1fdad1122c904e098a42f3100e04e03c247bc4'


def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def generation_signature(input_path, steps=5, seed=12345):
    return {'input_sha256': file_sha(input_path), 'weight_sha256': WEIGHT_SHA,
            'steps': steps, 'seed': seed, 'guidance': 5.0, 'dtype': 'float16',
            'num_latents': 3072, 'scheduler': 'ConsistencyFlowMatchEulerDiscreteScheduler'}


def record_shape(job, signature, resolution, mesh):
    write_json(job / 'shape_result.json', {'signature': signature, 'resolution': resolution,
               'raw_sha256': file_sha(job / 'generated_raw.ply'),
               'faces': len(mesh.faces), 'vertices': len(mesh.vertices), 'dimensions': mesh.extents.tolist()})


def completed_shape(job, input_path, resolution=255, steps=5, seed=12345):
    """Pure file check; never import torch or start a model process on a hit."""
    raw = job / 'generated_raw.ply'
    if not raw.is_file() or not 100 < raw.stat().st_size <= 200*1024**2:
        return False
    signature = generation_signature(input_path, steps, seed)
    manifest = job / 'shape_result.json'
    try:
        if manifest.is_file():
            saved = json.loads(manifest.read_text('utf-8'))
            return (saved['signature'] == signature and saved['resolution'] == resolution
                    and saved['raw_sha256'] == file_sha(raw))
        # Migration for earlier completed jobs, whose final progress record
        # and image-bound safetensors checkpoint predate shape_result.json.
        result = json.loads((job / 'inference.json').read_text('utf-8'))
        if (result.get('state') != 'shape_generated' or result.get('resolution') != resolution
                or result.get('model') != 'Hunyuan3D-2mini-Turbo'):
            return False
        checkpoint = job / 'diffusion_checkpoint.safetensors'
        if not checkpoint.is_file() or checkpoint.stat().st_size > 1024**2:
            return False
        with checkpoint.open('rb') as stream:
            size = struct.unpack('<Q', stream.read(8))[0]
            if not 2 <= size <= 65536 or size+8 > checkpoint.stat().st_size:
                return False
            header = json.loads(stream.read(size))
        meta = header.get('__metadata__', {})
        tensor = header.get('latents', {})
        if (json.loads(meta.get('signature', '{}')) != signature or meta.get('completed') != str(steps)
                or tensor.get('shape') != [1, 3072, 64] or tensor.get('dtype') != 'F16'):
            return False
        import numpy as np
        import trimesh
        mesh = trimesh.load(raw, force='mesh', process=False)
        if (len(mesh.faces) != result.get('faces') or len(mesh.vertices) != result.get('vertices')
                or not np.isfinite(mesh.vertices).all()
                or not np.allclose(mesh.extents, result.get('dimensions', []), atol=1e-6)):
            return False
        record_shape(job, signature, resolution, mesh)
        return True
    except (OSError, ValueError, KeyError, TypeError, struct.error):
        return False
