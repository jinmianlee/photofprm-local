"""Exact query-chunked attention for Intel GPUs with small allocation limits.

Every query still attends to every key. This is not token pruning or an
approximate attention algorithm. Apply only within the isolated inference worker.
"""
import torch.nn.functional as F

NATIVE_SDPA = F.scaled_dot_product_attention


def bounded_sdpa(query, key, value, attn_mask=None, dropout_p=0.0,
                 is_causal=False, scale=None, enable_gqa=False):
    kwargs = dict(attn_mask=attn_mask, dropout_p=dropout_p,
                  is_causal=is_causal, scale=scale, enable_gqa=enable_gqa)
    # Nontrivial masks and causal offsets require their own chunk alignment.
    # They are not used by these image and shape inference models.
    if query.shape[-2] <= 1024 or attn_mask is not None or is_causal or dropout_p:
        return NATIVE_SDPA(query, key, value, **kwargs)
    import torch
    return torch.cat([NATIVE_SDPA(chunk, key, value, **kwargs)
                      for chunk in query.split(1024, dim=-2)], dim=-2)


def install():
    from hy3dgen.shapegen.models.denoisers import hunyuan3ddit
    from hy3dgen.shapegen.models.autoencoders import attention_blocks, attention_processors
    F.scaled_dot_product_attention = bounded_sdpa
    hunyuan3ddit.scaled_dot_product_attention = bounded_sdpa
    attention_blocks.scaled_dot_product_attention = bounded_sdpa
    attention_processors.scaled_dot_product_attention = bounded_sdpa
