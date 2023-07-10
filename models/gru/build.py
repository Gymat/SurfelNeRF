from fvcore.common.registry import Registry  # for backward compatibility.
import pdb

FUSION_REGISTRY = Registry("FUSION")  # noqa F401 isort:skip
FUSION_REGISTRY.__doc__ = """
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

__all__ = ['FUSION_REGISTRY', 'build_fusion_net']

def build_fusion_net(cfg):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    """
    meta_name = cfg.MODEL.FUSION.NAME
    solver = FUSION_REGISTRY.get(meta_name)(cfg,)
    # model.to(torch.device(cfg.MODEL.DEVICE))

    return solver
