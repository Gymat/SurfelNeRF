import torch
from torch.utils.data import DataLoader
from fvcore.common.registry import Registry  # for backward compatibility.

from data.ddp_sampler import DistributedSampler_gru

DATASET_REGISTRY = Registry("DATASET")  # noqa F401 isort:skip
DATASET_REGISTRY.__doc__ = """
Registry for meta-solver, i.e. the whole training pipeline and the model.

The registered object will be called with `obj(cfg)`
and expected to return a `nn.Module` object.
"""

def build_dataloader(cfg, **kwargs):
    if 'test_dataset' in kwargs:
        dataset = build_test_dataset(cfg, **kwargs)
    else:
        dataset = build_dataset(cfg, **kwargs)
    if cfg.IS_TRAIN:
        shuffle_data = cfg.TRAIN.SHUFFLE_DATA
    else:
        shuffle_data = False
    if 'distributed' in kwargs:
        distributed = kwargs['distributed']
    else:
        distributed = False
    if distributed:
        if cfg.MODEL.FUSION.BUILD_FUSION and cfg.MODEL.FUSION.USE_GRU_DDP:
            sampler_gru = DistributedSampler_gru(dataset, shuffle=False, num_replicas=torch.cuda.device_count())
            sampler_item = sampler_gru
        else:
            sampler_item = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle_data,
                                                                       num_replicas=torch.cuda.device_count(),
                                                                       )
        dataloader = DataLoader(dataset=dataset,
                                batch_size=cfg.TRAIN.BATCH_SIZE, pin_memory=cfg.TRAIN.PIN_MEMORY,
                                num_workers=cfg.TRAIN.NUM_WORKERS, drop_last=True, sampler=sampler_item)
    else:
        dataloader = DataLoader(dataset=dataset,
                                batch_size=cfg.TRAIN.BATCH_SIZE, pin_memory=cfg.TRAIN.PIN_MEMORY,
                                shuffle=shuffle_data, num_workers=cfg.TRAIN.NUM_WORKERS, drop_last=True,)

    return dataloader


def build_dataset(cfg, **kwargs):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    """
    meta_dataset = cfg.DATA.DATASET_NAME
    dataset = DATASET_REGISTRY.get(meta_dataset)(cfg, **kwargs)
    # model.to(torch.device(cfg.MODEL.DEVICE))

    return dataset


def build_test_dataset(cfg, **kwargs):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    """
    meta_dataset = cfg.DATA.EVAL_DATASET_NAME
    dataset = DATASET_REGISTRY.get(meta_dataset)(cfg, **kwargs)
    # model.to(torch.device(cfg.MODEL.DEVICE))

    return dataset


if __name__ == '__main__':
    from config.defaults import get_config
    cfg = get_config()
    dataloader = build_dataloader(cfg)
