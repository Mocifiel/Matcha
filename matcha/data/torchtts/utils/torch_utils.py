import torch.nn as nn
from torchtts.utils.dict_utils import map_nested


def recursive_detach(data_structure, to_cpu=False):
    """Detach all tensors in `in_dict`.

    May operate recursively if some of the values in `in_dict` are dictionaries
    which contain instances of `torch.Tensor`. Other types in `in_dict` are
    not affected by this utility function.

    Args:
        data_structure:
        to_cpu: Whether to move tensor to cpu
    Return:
        out_dict:
    """

    def detach_fn(v):
        if callable(getattr(v, "detach", None)):
            # detach
            v = v.detach()
            if to_cpu:
                v = v.cpu()
        return v

    return map_nested(detach_fn, data_structure)


def check_ddp_wrapped(model: nn.Module) -> bool:
    """Checks whether model is wrapped with DataParallel/DistributedDataParallel."""
    parallel_wrappers = nn.DataParallel, nn.parallel.DistributedDataParallel
    return isinstance(model, parallel_wrappers)


def unwrap_ddp(model: nn.Module) -> nn.Module:
    if check_ddp_wrapped(model):
        model = model.module
    return model
