from fvcore.common.registry import Registry  # for backward compatibility.

MODEL_REGISTRY = Registry("MODEL")  # noqa F401 isort:skip
MODEL_REGISTRY.__doc__ = """
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

__all__ = ['MODEL_REGISTRY', 'build_model']

def build_model(cfg, **kwargs):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    """
    meta_solver = cfg.MODEL.NAME
    solver = MODEL_REGISTRY.get(meta_solver)(cfg, **kwargs)
    # model.to(torch.device(cfg.MODEL.DEVICE))

    return solver