from collections import OrderedDict
import torch.distributed as dist

from torchtts.hooks.base_hook import Hook
from torchtts.hooks.base_hook import HookNode
from torchtts.utils.import_utils import _HOROVOD_AVAILABLE

if _HOROVOD_AVAILABLE:
    import horovod.torch as hvd


def sort_hooks_by_order(hooks):
    if hooks is None:
        return OrderedDict()
    elif isinstance(hooks, (dict, OrderedDict)):
        output = [(k, v) for k, v in hooks.items()]
        output = sorted(output, key=lambda x: x[1].order)
        output = OrderedDict(output)
    elif isinstance(hooks, list):
        output = sorted(hooks, key=lambda x: x.order)
        output = OrderedDict([(i, value) for i, value in enumerate(output)])
    else:
        raise TypeError(f"Hooks must be either Dict/OrderedDict or list, got {type(hooks)}")

    return output


def check_hooks_type(hooks):
    hooks = list(hooks or [])
    for hook in hooks:
        if not isinstance(hook, Hook):
            raise TypeError("Hooks must be a Hook, given: {}".format(hook))
    return hooks


def select_hooks_by_node(hooks):
    # distributed run setting
    output = hooks.copy()
    if dist.is_initialized():
        rank = dist.get_rank()
    elif _HOROVOD_AVAILABLE and hvd.is_initialized():
        rank = hvd.rank()
    else:
        rank = -1
    if rank == 0:  # master node
        # remove worker-only hooks on master node
        for k in list(filter(lambda c: output[c].node == HookNode.WORKER, output)):
            del output[k]
    elif rank > 0:  # worker node
        # remove master-only hooks on worker nodes
        for k in list(filter(lambda c: output[c].node == HookNode.MASTER, output)):
            del output[k]
    return output
