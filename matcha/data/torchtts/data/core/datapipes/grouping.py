import re
import warnings
import logging
from collections import defaultdict
from typing import Any, Callable, DefaultDict, TypeVar, Tuple, Optional, Iterable, Iterator, List, Sized

import numpy as np

from torchtts.data.core.datapipes.iter_datapipe import functional_datapipe
from torchtts.data.core.datapipes.iter_datapipe import IterDataPipe

logger = logging.getLogger(__name__)
T_co = TypeVar("T_co", covariant=True)


@functional_datapipe("sharding_filter")
class ShardingFilterIterDataPipe(IterDataPipe):
    r"""
    Wrapper that allows DataPipe to be sharded (functional name: ``sharding_filter``). After ``apply_sharding`` is
    called, each instance of the DataPipe (on different workers) will have every `n`-th element of the
    original DataPipe, where `n` equals to the number of instances.
    Args:
        datapipe: Iterable DataPipe that will be sharded
    """

    def __init__(self, datapipe: IterDataPipe):
        self.datapipe = datapipe
        self.num_of_instances = 1
        self.instance_id = 0

    def apply_sharding(self, num_of_instances, instance_id):
        self.num_of_instances = num_of_instances
        self.instance_id = instance_id
        super().apply_sharding(num_of_instances, instance_id)

    def __iter__(self):
        for i, item in enumerate(self.datapipe):
            if i % self.num_of_instances == self.instance_id:
                yield item

    def __len__(self):
        if isinstance(self.datapipe, Sized):
            return len(self.datapipe) // self.num_of_instances + (
                1 if (self.instance_id < len(self.datapipe) % self.num_of_instances) else 0
            )
        raise TypeError("{} instance doesn't have valid length".format(type(self).__name__))


@functional_datapipe("batch")
class BatchIterDataPipe(IterDataPipe[List[T_co]]):
    r""":class:`BatchIterDataPipe`.
    Iterable DataPipe to create mini-batches of data. An outer dimension will be added as
    `batch_size` if `drop_last` is set to `True`, or `length % batch_size` for the
    last batch if `drop_last` is set to `False`.
    Args:
        datapipe: Iterable DataPipe being batched
        batch_size: The size of each batch
        drop_last: Option to drop the last batch if it's not full
    """
    datapipe: IterDataPipe[T_co]
    batch_size: int
    drop_last: bool
    length: Optional[int]

    def __init__(
        self,
        datapipe: IterDataPipe[T_co],
        batch_size: int,
        drop_last: bool = False,
    ) -> None:
        assert batch_size > 0, "Batch size is required to be larger than 0!"
        super().__init__()
        self.datapipe = datapipe
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.length = None

    def __iter__(self) -> Iterator[List[T_co]]:
        batch: List[T_co] = []
        for x in self.datapipe:
            batch.append(x)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0:
            if not self.drop_last:
                yield batch

    def __len__(self) -> int:
        if self.length is not None:
            return self.length
        if isinstance(self.datapipe, Sized) and len(self.datapipe) >= 0:
            if self.drop_last:
                self.length = len(self.datapipe) // self.batch_size
            else:
                self.length = (len(self.datapipe) + self.batch_size - 1) // self.batch_size
            return self.length
        raise NotImplementedError


@functional_datapipe("dynamic_batch")
class DynamicBatchIterDataPipe(IterDataPipe[List[T_co]]):
    r""":class:`DynamicBatchIterDataPipe`.
    Iterable DataPipe to create dynamic mini-batches of data with buckets, which are
    used to group items with the keys generated from group_key_func.
     Args:
        datapipe: Iterable DataPipe being batched
        group_key_fn: Function, from element in DataBlock to int, determines
          the bucket it goes into.
        bucket_boundaries: List of integer, upper length boundaries of buckets.
        batch_sizes: List of integer, batch size per bucket. Should be
          len(bucket_boundaries) + 1.
    """
    datapipe: IterDataPipe[T_co]
    group_key_fn: Callable
    bucket_boundaries: List[int]
    batch_sizes: List[int]

    def __init__(
        self,
        datapipe,
        group_key_fn,
        bucket_boundaries,
        batch_sizes,
    ) -> None:
        self.datapipe = datapipe
        if hasattr(group_key_fn, "__name__") and group_key_fn.__name__ == "<lambda>":
            warnings.warn(
                "Lambda function is not supported for pickle, please use "
                "regular python function or functools.partial instead."
            )
        self.group_key_fn = group_key_fn
        self.batch_sizes = batch_sizes

        self.buckets_min = np.array([np.iinfo(np.int32).min] + bucket_boundaries)
        self.buckets_max = np.array(bucket_boundaries + [np.iinfo(np.int32).max])

    def __iter__(self):
        buckets: DefaultDict[int, List[T_co]] = defaultdict(list)
        for x in self.datapipe:
            key = self.group_key_fn(x)
            # Calculate the bucket id
            condition = np.logical_and(np.less_equal(self.buckets_min, key), np.less(key, self.buckets_max))
            bucket_id = np.min(np.where(condition))
            # Insert to the corresponding bucket
            bucket = buckets[bucket_id]
            bucket.append(x)
            if len(bucket) == self.batch_sizes[bucket_id]:
                yield buckets.pop(bucket_id)
        while buckets:
            yield buckets.popitem()[-1]

    def __len__(self) -> int:
        raise NotImplementedError


def base_plus_ext(path):
    """Helper method that splits off all extension."""
    match = re.match(r"^((?:.*/|).+)[.]([^/]*)$", path)
    if not match:
        return None, None
    return match.group(1), match.group(2)


class GroupByKeyIterDataPipe(IterDataPipe):
    r""":class:`GroupByKeyIterDataPipe`.
    Iterable datapipe to group data from input iterable by keys which are generated from `group_key_fn`,
    yields a list with `group_size` items in it, each item in the list is a tuple of key and data
    args:
        datapipe: Iterable datapipe that provides data. (typically str key (eg. pathname) and data stream in tuples)
        group_size: the size of group
        max_buffer_size: the max size of stream buffer which is used to store not yet grouped but iterated data
        group_key_fn: a function which is used to generate group key from the data in the input datapipe
        length: a nominal length of the datapipe
    """

    def __init__(self, datapipe: Iterable[Tuple[str, Any]]):
        super().__init__()

        self.datapipe: Iterable[Tuple[str, Any]] = datapipe
        self.key_set = set()

    def __iter__(self) -> Iterator[list]:
        current_sample, prev_key = None, None
        for item in self.datapipe:
            key, suffix = base_plus_ext(item[0])
            if key in self.key_set:
                logger.warning(
                    f"found duplicated key {key} when grouping items, we have "
                    f"already processed and yielded a group with key {key}."
                    "Tips: this often means you don't put data in shards in order, "
                    "please adjust the order of data to make sure the same keys "
                    "all adjacent."
                )
                continue
            if key != prev_key:
                if prev_key is not None:
                    yield current_sample
                current_sample = {'__key__': key}
                prev_key = key
            if suffix in current_sample:
                logger.warning(
                    f"{item[0]} may be duplicated feature in (tar/zip/chunk) "
                    f"since current sample already contains {current_sample.keys()}"
                )
            current_sample[suffix] = item[1]
        yield current_sample

    def __len__(self):
        if self.length == -1:
            raise NotImplementedError
        return self.length
