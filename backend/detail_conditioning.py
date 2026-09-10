"""Experimental multiscale DINO conditioning; no learned weights are changed.

A selected region is encoded at the full image-encoder size. Its spatial tokens
are mapped back into the whole-image token grid with a soft boundary. The global
class token and all tokens outside the ROI remain unchanged. This is an ablation
candidate, not a validated universal quality improvement.
"""
import math
import torch
import torch.nn.functional as F


def enrich_conditioning(pipeline, image, condition, roi, strength=.35):
    if not 0 <= strength <= .6:
        raise ValueError('Experimental detail strength must be between 0 and 0.6.')
    if len(roi) != 4 or not all(math.isfinite(v) for v in roi):
        raise ValueError('Detail region requires four finite normalized coordinates.')
    x0, y0, x1, y1 = roi
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1 and min(x1-x0, y1-y0) >= .03):
        raise ValueError('Detail region is outside the prepared image or too small.')
    if strength == 0:
        return condition, {'strength': 0, 'changed_tokens': 0}
    height, width = image.shape[-2:]
    left, right = round(x0*width), round(x1*width)
    top, bottom = round(y0*height), round(y1*height)
    crop = image[..., top:bottom, left:right]
    ch, cw = crop.shape[-2:]
    side = max(ch, cw)
    pad_left, pad_top = (side-cw)//2, (side-ch)//2
    crop = F.pad(crop, (pad_left, side-cw-pad_left, pad_top, side-ch-pad_top), value=1.)
    crop = F.interpolate(crop, (height, width), mode='bilinear', align_corners=False, antialias=True)
    detailed = pipeline.encode_cond(image=crop, additional_cond_inputs={},
                                    do_classifier_free_guidance=False, dual_guidance=False)['main']
    original = condition['main']
    size = math.isqrt(original.shape[1]-1)
    if size**2+1 != original.shape[1] or detailed.shape != original.shape:
        raise ValueError('Unexpected DINO grid; refuse to guess token coordinates.')
    # The small interpolation runs on CPU to avoid a new GPU operator dependency.
    features = detailed[:, 1:].float().cpu().transpose(1, 2).reshape(1, -1, size, size)
    ys = (torch.arange(size, dtype=torch.float32)+.5)/size*height
    xs = (torch.arange(size, dtype=torch.float32)+.5)/size*width
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    ux, uy = (xx-left+pad_left)/side, (yy-top+pad_top)/side
    grid = torch.stack((ux*2-1, uy*2-1), dim=-1)[None]
    sampled = F.grid_sample(features, grid, mode='bilinear', padding_mode='border', align_corners=False)
    sampled = sampled.flatten(2).transpose(1, 2).to(device=original.device, dtype=original.dtype)
    wx = torch.minimum((xx-left)/max(cw, 1), (right-xx)/max(cw, 1))
    wy = torch.minimum((yy-top)/max(ch, 1), (bottom-yy)/max(ch, 1))
    blend = (torch.minimum(wx, wy)*5).clamp(0, 1).reshape(1, -1, 1).to(original.device, original.dtype)*strength
    original_patches = original[:, 1:]
    # Match per-token RMS so magnification does not merely amplify conditioning.
    original_rms = original_patches.float().square().mean(-1, keepdim=True).sqrt()
    sampled_rms = sampled.float().square().mean(-1, keepdim=True).sqrt().clamp_min(1.e-6)
    sampled = (sampled.float()*original_rms/sampled_rms).to(original.dtype)
    patches = original_patches+(sampled-original_patches)*blend
    mixed = torch.cat((original[:, :1], patches), dim=1)
    return {**condition, 'main': mixed}, {
        'method': 'multiscale_dino_roi_v1', 'roi_prepared_image': list(roi), 'strength': strength,
        'changed_tokens': int((blend > 0).sum()), 'total_patch_tokens': size**2,
        'global_class_token_preserved': torch.equal(mixed[:, :1], original[:, :1]),
        'relative_condition_delta': float((mixed-original).float().norm()/original.float().norm()),
        'crop_pixels_before_encoding': [cw, ch], 'crop_encoder_size': width,
    }
