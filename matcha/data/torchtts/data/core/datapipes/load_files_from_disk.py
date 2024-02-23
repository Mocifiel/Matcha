import warnings
from io import BufferedIOBase
from typing import Iterable, Iterator, Tuple

from torchtts.data.core.datapipes.iter_datapipe import IterDataPipe
from torchtts.data.core.datapipes.utils import StreamWrapper


class LoadFilesFromDiskIterDataPipe(IterDataPipe):
    r""":class:`LoadFilesFromDiskIterDataPipe`.
    Iterable Datapipe to load file binary streams from given path names,
    yield pathname and binary stream in a tuple.
    args:
        datapipe: Iterable datapipe that provides path names
        length: a nominal length of the datapipe
    """

    def __init__(self, datapipe: Iterable[str], length: int = -1):
        super().__init__()
        self.datapipe: Iterable = datapipe
        self.length: int = length

    def __iter__(self) -> Iterator[Tuple[str, BufferedIOBase]]:
        yield from get_file_binaries_from_pathnames(self.datapipe)

    def __len__(self):
        if self.length == -1:
            raise NotImplementedError
        return self.length


def get_file_binaries_from_pathnames(pathnames: Iterable):
    if not isinstance(pathnames, Iterable):
        warnings.warn("get_file_binaries_from_pathnames needs the input be an Iterable")
        raise TypeError

    for pathname in pathnames:
        if not isinstance(pathname, str):
            warnings.warn("file pathname must be string type, but got {}".format(type(pathname)))
            raise TypeError

        yield pathname, StreamWrapper(open(pathname, "rb"))
