import warnings
import struct
from io import BufferedIOBase
from typing import Iterable, Iterator, Tuple

from torchtts.data.core.datapipes.iter_datapipe import IterDataPipe
from torchtts.data.core.datapipes.utils import validate_pathname_binary_tuple


class ReadFilesFromChunkIterDataPipe(IterDataPipe):
    r""":class:`ReadFilesFromChunkIterDataPipe`.
    Iterable datapipe to read chunk binary streams from input iterable which contains tuples of
    pathname and tar binary stream, yields pathname and binary data in a tuple.
    args:
        datapipe: Iterable datapipe that provides pathname and binary data in tuples
        length: a nominal length of the datapipe
    """

    def __init__(self, datapipe: Iterable[Tuple[str, BufferedIOBase]], length: int = -1):
        super().__init__()
        self.datapipe: Iterable[Tuple[str, BufferedIOBase]] = datapipe
        self.length: int = length

    def __iter__(self) -> Iterator[Tuple[str, BufferedIOBase]]:
        if not isinstance(self.datapipe, Iterable):
            raise TypeError("datapipe must be Iterable type but got {}".format(type(self.datapipe)))
        for data in self.datapipe:
            validate_pathname_binary_tuple(data)
            pathname, data_stream = data
            try:
                while True:
                    bytes_read = data_stream.read(4)
                    if bytes_read == b"":
                        break
                    (inner_key_size,) = struct.unpack("<I", bytes_read)
                    inner_key = data_stream.read(inner_key_size).decode("utf-8")
                    (value_size,) = struct.unpack("<I", data_stream.read(4))
                    value = data_stream.read(value_size)
                    yield inner_key, value
            except Exception as e:
                warnings.warn(
                    "Unable to extract files from corrupted chunk stream {} due to: {}, abort!".format(pathname, e)
                )
                raise e

    def __len__(self):
        if self.length == -1:
            raise NotImplementedError
        return self.length
