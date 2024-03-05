import random
from typing import List, Sized, TypeVar, Iterator

from torchtts.data.core.datapipes.iter_datapipe import functional_datapipe
from torchtts.data.core.datapipes.iter_datapipe import IterDataPipe

T_co = TypeVar("T_co", covariant=True)


@functional_datapipe("shuffle")
class ShuffleIterDataPipe(IterDataPipe[T_co]):
    r""":class:`ShuffleIterDataPipe`
    Iterable DataPipe to shuffle the input DataPipe with a buffer. The buffer
    with `buffer_size` is filled with elements from the datapipe first. Then,
    each item will be yielded from the buffer by reservoir sampling via iterator.
    `buffer_size` is required to be larger than 0. For `buffer_size == 1`, the
    datapipe is not shuffled. In order to fully shuffle all elements from datapipe,
    `buffer_size` is required to be greater than or equal to the size of datapipe.
    When it is used with :class:`~torch.utils.data.DataLoader`, the methods to
    set up random seed are different based on :attr:`num_workers`.
    For single-process mode (:attr:`num_workers == 0`), the random seed is set before
    the :class:`~torch.utils.data.DataLoader` in the main process. For multi-process
    mode (:attr:`num_worker > 0`), `worker_init_fn` is used to set up a random seed
    for each worker process.
    args:
        datapipe: The IterDataPipe being shuffled
        buffer_size: The buffer size for shuffling
    """
    datapipe: IterDataPipe[T_co]
    buffer_size: int
    _buffer: List[T_co]

    def __init__(self, datapipe: IterDataPipe[T_co], *, buffer_size: int) -> None:
        super().__init__()
        assert buffer_size > 0, "buffer_size should be larger than 0"
        self.datapipe = datapipe
        self.buffer_size = buffer_size
        self._buffer = []
        self._rng = random.Random()

    def __iter__(self) -> Iterator[T_co]:
        for x in self.datapipe:
            if len(self._buffer) == self.buffer_size:
                idx = self._rng.randint(0, self.buffer_size - 1)
                yield self._buffer[idx]
                self._buffer[idx] = x
            else:
                self._buffer.append(x)
        self._rng.shuffle(self._buffer)
        while self._buffer:
            yield self._buffer.pop()

    def set_epoch(self, epoch):
        self._rng.seed(epoch)
        super().set_epoch(epoch)

    def __len__(self) -> int:
        if isinstance(self.datapipe, Sized) and len(self.datapipe) >= 0:
            return len(self.datapipe)
        raise NotImplementedError
