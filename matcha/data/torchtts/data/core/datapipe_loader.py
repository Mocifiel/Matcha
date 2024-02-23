import functools
import logging

import numpy as np
import random
import torch
import torch.distributed as dist

from torchtts.data.core.datapipes import IterDataPipe

logger = logging.getLogger(__name__)


def _get_distributed_settings():
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(), dist.get_rank()
    else:
        return 1, 0


def _sharding_worker_init_fn(worker_init_fn, world_size, rank_id, worker_id):
    global_worker_id = worker_id
    info = torch.utils.data.get_worker_info()
    total_workers = info.num_workers
    datapipe = info.dataset
    # To distribute elements across distributed process evenly, we should shard data on distributed
    # processes first then shard on worker processes
    total_workers *= world_size
    global_worker_id = global_worker_id * world_size + rank_id
    datapipe.apply_sharding(total_workers, global_worker_id)
    if worker_init_fn is not None:
        worker_init_fn(worker_id)


class DataPipeLoader:
    """Data pipeline loader which extends the PyTorch DataLoader.

    The loader is customized for use with data pipeline datasets. Especially, we
    customize the random seed across gpu ranks and workers. The majority of
    arguments in the basic PyTorch DataLoader are supported.

    Notes:
        The following are not supported because their functions are duplicated in
        data pipeline: *batch_size* *shuffle*, *sampler*, *batch_sampler*,
        *worker_init_fn*
    """

    def __init__(self, **kwargs):
        self._worker_size = 0
        self.datapipe = kwargs["dataset"]

        if not isinstance(self.datapipe, IterDataPipe):
            raise ValueError("Expect dataset to be a DataPipe")

        if "num_workers" in kwargs:
            self._worker_size = kwargs["num_workers"]

        self._worker_init = _WorkerInit(self._worker_size)

        if "seed" in kwargs:
            random_seed = [kwargs["seed"] + i for i in range(self._worker_size)]
            np_seed = [kwargs["seed"] + i for i in range(self._worker_size)]
            torch_seed = [kwargs["seed"] + i for i in range(self._worker_size)]
            self._worker_init.set_seeds(random_seed, np_seed, torch_seed)
            del kwargs["seed"]

        worker_init_fn = self._worker_init.init_worker

        ws, rank = _get_distributed_settings()
        if self._worker_size > 0:
            worker_init_fn = functools.partial(_sharding_worker_init_fn, worker_init_fn, ws, rank)
        else:
            self.datapipe.apply_sharding(ws, rank)

        new_kwargs = kwargs.copy()
        for arg in ("batch_size", "shuffle", "sampler", "batch_sampler", "worker_init_fn"):
            if arg in kwargs:
                logger.warning('Your argument "%s" will be ignored!', arg)
                del new_kwargs[arg]

        # Batching has already handled by data pipeline
        new_kwargs["batch_size"] = None
        new_kwargs["worker_init_fn"] = worker_init_fn
        self._loader = torch.utils.data.DataLoader(**new_kwargs)

    def __iter__(self):
        for item in self._loader:
            yield item


class _WorkerInit:
    """Simple wrapper object for managing worker initialization state"""

    def __init__(self, worker_size):
        self._worker_size = worker_size
        self._random_seed = None
        self._numpy_seeds = None
        self._torch_seeds = None
        self._worker_rank = 0

    def set_seeds(self, random_seed, numpy_seeds, torch_seeds):
        """Set initialization seeds"""
        self._random_seed = random_seed
        self._numpy_seeds = numpy_seeds
        self._torch_seeds = torch_seeds

    def init_worker(self, rank):
        """Callback for data loader worker initialization."""
        self._worker_rank = rank

        if self._random_seed is not None:
            random.seed(self._random_seed[self._worker_rank])
            np.random.seed(self._numpy_seeds[self._worker_rank])
            torch.manual_seed(self._torch_seeds[self._worker_rank])
        else:
            # Notes: By default, the seeds of PyTorch and random module have been
            # set to base_seed + worker_id. However, seeds for other libraries
            # like numpy may not been set yet. Here, we set it with the same seed
            # as PyTorch.
            worker_info = torch.utils.data.get_worker_info()
            torch_seed = worker_info.seed % (2**32 - 1)
            random.seed(torch_seed)
            np.random.seed(torch_seed)

        logger.debug("Worker %d of %d initialized", (self._worker_rank + 1), self._worker_size)
