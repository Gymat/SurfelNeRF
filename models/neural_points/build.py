from fvcore.common.registry import Registry  # for backward compatibility.
import pdb

POINTS_REGISTRY = Registry("Neural Points")  # noqa F401 isort:skip
POINTS_REGISTRY.__doc__ = """
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

__all__ = ['POINTS_REGISTRY', 'build_neural_points']

def build_neural_points(cfg, **kwargs):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    """
    meta_name = cfg.MODEL.NEURAL_POINTS.NAME
    solver = POINTS_REGISTRY.get(meta_name)(cfg, **kwargs)
    # model.to(torch.device(cfg.MODEL.DEVICE))

    return solver
