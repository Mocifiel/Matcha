import warnings
import numpy as np
from typing import TypeVar, Callable, Iterator, Sized, Optional, Tuple, Dict

from torch.utils.data import _utils

from torchtts.data.core.datapipes.iter_datapipe import functional_datapipe
from torchtts.data.core.datapipes.iter_datapipe import IterDataPipe

T_co = TypeVar("T_co", covariant=True)


# Default function to return each item directly
# In order to keep datapipe picklable, eliminates the usage
# of python lambda function
def default_fn(data):
    return data


@functional_datapipe("map")
class MapIterDataPipe(IterDataPipe[T_co]):
    r""":class:`MapIterDataPipe`.
    Iterable DataPipe to run a function over each item from the source DataPipe.
    Args:
        datapipe: Source Iterable DataPipe
        fn: Function called over each item
    """
    datapipe: IterDataPipe
    fn: Callable

    def __init__(
        self,
        datapipe: IterDataPipe,
        fn: Callable = default_fn,
        fn_args: Optional[Tuple] = None,
        fn_kwargs: Optional[Dict] = None,
    ) -> None:
        super().__init__()
        self.datapipe = datapipe
        # Partial object has no attribute '__name__', but can be pickled
        if hasattr(fn, "__name__") and fn.__name__ == "<lambda>":
            warnings.warn(
                "Lambda function is not supported for pickle, please use "
                "regular python function or functools.partial instead."
            )
        self.fn = fn  # type: ignore
        self.args = () if fn_args is None else fn_args
        self.kwargs = {} if fn_kwargs is None else fn_kwargs

    def __iter__(self) -> Iterator[T_co]:
        for data in self.datapipe:
            yield self.fn(data, *self.args, **self.kwargs)

    def __len__(self) -> int:
        if isinstance(self.datapipe, Sized) and len(self.datapipe) >= 0:
            return len(self.datapipe)
        raise NotImplementedError


@functional_datapipe("filter_map")
class FilterMapIterDataPipe(IterDataPipe[T_co]):
    r""":class:`FilterMapIterDataPipe`.
    Creates an iterator that both filters and maps. The returned iterator yields only the values for
    which the supplied fn returns is not None.
    Args:
        datapipe: Source Iterable DataPipe
        fn: Function called over each item (but may return None)
    """
    datapipe: IterDataPipe
    fn: Callable

    def __init__(
        self,
        datapipe: IterDataPipe,
        fn: Callable = default_fn,
        fn_args: Optional[Tuple] = None,
        fn_kwargs: Optional[Dict] = None,
    ) -> None:
        super().__init__()
        self.datapipe = datapipe
        # Partial object has no attribute '__name__', but can be pickled
        if hasattr(fn, "__name__") and fn.__name__ == "<lambda>":
            warnings.warn(
                "Lambda function is not supported for pickle, please use "
                "regular python function or functools.partial instead."
            )
        self.fn = fn  # type: ignore
        self.args = () if fn_args is None else fn_args
        self.kwargs = {} if fn_kwargs is None else fn_kwargs

    def __iter__(self) -> Iterator[T_co]:
        for data in self.datapipe:
            data = self.fn(data, *self.args, **self.kwargs)
            if data is not None:
                yield data

    def __len__(self) -> int:
        if isinstance(self.datapipe, Sized) and len(self.datapipe) >= 0:
            return len(self.datapipe)
        raise NotImplementedError


@functional_datapipe("repeat")
class RepeatEpochDataPipe(IterDataPipe[T_co]):
    r""":class:`ExpandEpochDataPipe`.
    Iterable DataPipe to repeat one epoch data.
    This is very helpfull if the training dataset is very small
    Args:
        datapipe: Source Iterable DataPipe
        mul: The times that one epoch data will be repeated
    """

    def __init__(
        self,
        datapipe: IterDataPipe,
        mul=10,
    ) -> None:
        super().__init__()
        self.datapipe = datapipe
        self.mul = mul
        self.external_epoch = 0

    def __iter__(self) -> Iterator[T_co]:
        for i in range(self.mul):
            self.datapipe.set_epoch(self.external_epoch * self.mul + i)
            for data in self.datapipe:
                yield data

    def set_epoch(self, epoch):
        self.external_epoch = epoch
        super().set_epoch(epoch)

    def __len__(self) -> int:
        if isinstance(self.datapipe, Sized) and len(self.datapipe) >= 0:
            return len(self.datapipe) * self.mul
        raise NotImplementedError


def _pad_batch(batch, padding_axis, padding_value, max_length):
    padded_shape = list(batch[0].shape)
    padded_shape[padding_axis] = max_length
    # Restore the actual length of padding axis
    length = []
    padded_batch = []
    for item in batch:
        actual_length = item.shape[padding_axis]
        padded_length = max_length - actual_length
        pad_width = [(0, 0) for _ in range(len(item.shape))]
        pad_width[padding_axis] = (0, padded_length)
        padded_item = np.pad(item, pad_width, mode="constant", constant_values=padding_value)
        padded_batch.append(padded_item)
        length.append(np.asarray(actual_length, dtype=np.int64))

    return padded_batch, length


def default_collate_fn(dict_batch, padding_axes=None, padding_values=None, blacklist=None):
    padding_axes = {} if padding_axes is None else padding_axes
    padding_values = {} if padding_values is None else padding_values
    blacklist = [] if blacklist is None else blacklist

    batch_dict = {k: [dic[k] for dic in dict_batch] for k in dict_batch[0]}
    collated_dict = {}
    for key, batch in batch_dict.items():
        if key == "__key__" or key in blacklist:
            collated_dict[key] = batch
        elif key in padding_axes:
            padding_axis = padding_axes[key]
            max_length = max(item.shape[padding_axis] for item in batch)
            padding_value = padding_values.get(key, 0)
            padded_batch, actual_length = _pad_batch(batch, padding_axis, padding_value, max_length)
            collated_dict[key] = _utils.collate.default_collate(padded_batch)
        else:
            collated_dict[key] = _utils.collate.default_collate(batch)
    return collated_dict


@functional_datapipe("collate")
class CollateIterDataPipe(MapIterDataPipe):
    r""":class:`CollateIterDataPipe`.
    Iterable DataPipe to collate samples from datapipe to Tensor(s) by `util_.collate.default_collate`,
    or customized Data Structure by collate_fn.
    Args:
        datapipe: Iterable DataPipe being collated
        collate_fn: Customized collate function to collect and combine data or a batch of data.
                    Default function collates to Tensor(s) based on data type.
    Example: Convert integer data to float Tensor
        >>> class MyIterDataPipe(torch.utils.data.IterDataBlock):
        ...     def __init__(self, start, end):
        ...         super(MyIterDataPipe).__init__()
        ...         assert end > start, "this example code only works with end >= start"
        ...         self.start = start
        ...         self.end = end
        ...
        ...     def __iter__(self):
        ...         return iter(range(self.start, self.end))
        ...
        ...     def __len__(self):
        ...         return self.end - self.start
        ...
        >>> ds = MyIterDataPipe(start=3, end=7)
        >>> print(list(ds))
        [3, 4, 5, 6]
        >>> def collate_fn(batch):
        ...     return torch.tensor(batch, dtype=torch.float)
        ...
        >>> collated_ds = CollateIterDataPipe(ds, collate_fn=collate_fn)
        >>> print(list(collated_ds))
        [tensor(3.), tensor(4.), tensor(5.), tensor(6.)]
    """

    def __init__(
        self,
        datapipe: IterDataPipe,
        collate_fn: Callable = default_collate_fn,
        fn_args: Optional[Tuple] = None,
        fn_kwargs: Optional[Dict] = None,
    ) -> None:
        super().__init__(datapipe, fn=collate_fn, fn_args=fn_args, fn_kwargs=fn_kwargs)
