"""CPU surface sampling with floating coordinates and finite signed grids.

Coarse samples locate the surface, then each level evaluates a wider band around
its zero crossings. Interpolated signed values fill unqueried space. This avoids
NaN triangles while retaining real neural evaluations throughout the surface.
"""
import torch
import torch.nn.functional as F


class CPUVolumeDecoder:
    def __init__(self, progress=None):
        self.progress = progress or (lambda *args, **kwargs: None)

    @torch.no_grad()
    def __call__(self, latents, geo_decoder, bounds=1.01, num_chunks=2048,
                 octree_resolution=255, mc_level=0.0, min_resolution=63, **kwargs):
        if latents.shape[0] != 1:
            raise ValueError('CPU decoder accepts one object at a time.')
        if isinstance(bounds, (int, float)):
            bounds = [-bounds] * 3 + [bounds] * 3
        low = torch.tensor(bounds[:3], dtype=torch.float32, device=latents.device)
        span = torch.tensor(bounds[3:], dtype=torch.float32, device=latents.device) - low
        levels = [int(octree_resolution)]
        while levels[-1] // 2 >= min_resolution:
            levels.append(levels[-1] // 2)
        grid = None
        for cells in reversed(levels):
            width = cells + 1
            if grid is None:
                indices = torch.cartesian_prod(*[
                    torch.arange(width, device=latents.device)] * 3)
                flat = torch.empty(width ** 3, device=latents.device)
                flat_ids = torch.arange(width ** 3, device=latents.device)
            else:
                fine = F.interpolate(grid[None, None], size=(width,) * 3,
                                     mode='trilinear', align_corners=True)[0, 0]
                positive = (fine > mc_level).float()[None, None]
                maximum = F.max_pool3d(positive, 3, stride=1, padding=1)
                minimum = -F.max_pool3d(-positive, 3, stride=1, padding=1)
                crossing = maximum != minimum
                # Two extra fine cells accommodate displacement from interpolation.
                band = F.max_pool3d(crossing.float(), 5, stride=1, padding=2)[0, 0] > 0
                # Include uncertain logits as in the upstream hierarchical sampler.
                band |= (fine - mc_level).abs() < 0.95
                indices = band.nonzero()
                flat_ids = indices[:, 0] * width ** 2 + indices[:, 1] * width + indices[:, 2]
                flat = fine.reshape(-1)
            total = len(indices)
            if not total:
                raise ValueError('No surface band found in neural field.')
            self.progress('细化三维表面', grid_resolution=cells, samples_total=total, samples_done=0)
            for start in range(0, total, num_chunks):
                end = min(start + num_chunks, total)
                # Indices are integers. Cast BEFORE multiplying by a subunit step.
                points = indices[start:end].to(torch.float32) / cells * span + low
                values = geo_decoder(queries=points[None].to(latents.dtype), latents=latents)
                flat[flat_ids[start:end]] = values.reshape(-1).float()
                if start == 0 or end == total or (start // num_chunks) % 8 == 0:
                    self.progress('细化三维表面', grid_resolution=cells,
                                  samples_total=total, samples_done=end)
            grid = flat.reshape((width,) * 3)
            if not torch.isfinite(grid).all():
                raise ValueError('Neural field contains nonfinite values.')
        return grid[None]
