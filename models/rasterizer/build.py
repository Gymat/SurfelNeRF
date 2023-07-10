from fvcore.common.registry import Registry  # for backward compatibility.
import pdb

RASTERIZER_REGISTRY = Registry("Surfel Rasterizer")  # noqa F401 isort:skip
RASTERIZER_REGISTRY.__doc__ = """
Registry for meta-solver, i.e. the whole training pipeline and the model.

The registered object will be called with `obj(cfg)`
and expected to return a `nn.Module` object.
"""

# GENERATOR_REGISTRY = Registry("GENERATOR")  # noqa F401 isort:skip
# GENERATOR_REGISTRY.__doc__ = """
# Registry for meta-solver, i.e. the whole training pipeline and the model.
#
# The registered object will be called with `obj(cfg)`
# and expected to return a `nn.Module` object.
# """

__all__ = ['RASTERIZER_REGISTRY', 'build_rasterizer']

def build_rasterizer(cfg, **kwargs):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    """
    meta_name = cfg.MODEL.RASTERIZER.NAME
    solver = RASTERIZER_REGISTRY.get(meta_name)(cfg, **kwargs)
    # model.to(torch.device(cfg.MODEL.DEVICE))

    return solver
