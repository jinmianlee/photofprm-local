"""Load verified local inference tensors without initializing disposable weights.

Callers must verify the pinned config and safetensors hashes before using this
helper. Buffers are created normally; only parameters start on the meta device.
"""
from accelerate import init_empty_weights
from safetensors import safe_open
from hy3dgen.shapegen.pipelines import instantiate_from_config


def load_component(name, config, weight, device, dtype):
    with init_empty_weights(include_buffers=False):
        module = instantiate_from_config(config[name])
    if name == 'vae':
        # Neither training-only encoder is in the inference checkpoint or used
        # by shape decoding. Requiring all remaining keys prevents silent gaps.
        del module.encoder
        del module.pre_kl
    prefix = name + '.'
    with safe_open(weight, framework='pt', device='cpu') as tensors:
        state = {key[len(prefix):]: tensors.get_tensor(key) for key in tensors.keys()
                 if key.startswith(prefix)
                 and not (name == 'vae' and key.startswith(('vae.encoder.', 'vae.pre_kl.')))}
    module.load_state_dict(state, strict=True, assign=True)
    module.requires_grad_(False)
    module.to(device=device, dtype=dtype)
    return module


def load_pipeline(config, weight, device, dtype):
    from hy3dgen.shapegen.pipelines import Hunyuan3DDiTFlowMatchingPipeline
    return Hunyuan3DDiTFlowMatchingPipeline(
        model=load_component('model', config, weight, device, dtype),
        vae=load_component('vae', config, weight, device, dtype),
        conditioner=load_component('conditioner', config, weight, device, dtype),
        scheduler=instantiate_from_config(config['scheduler']),
        image_processor=instantiate_from_config(config['image_processor']),
        device=device, dtype=dtype)
