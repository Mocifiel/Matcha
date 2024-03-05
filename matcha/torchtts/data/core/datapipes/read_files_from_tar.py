import os
import tarfile
import warnings
from io import BufferedIOBase
from typing import Iterable, Iterator, Tuple, Optional, IO, cast

from torchtts.data.core.datapipes.iter_datapipe import IterDataPipe
from torchtts.data.core.datapipes.utils import validate_pathname_binary_tuple
from torchtts.data.core.datapipes.utils import StreamWrapper


class ReadFilesFromTarIterDataPipe(IterDataPipe):
    r""":class:`ReadFilesFromTarIDP`.
    Iterable datapipe to extract tar binary streams from input iterable which contains tuples of
    pathname and tar binary stream, yields pathname and extracted binary stream in a tuple.
    args:
        datapipe: Iterable datapipe that provides pathname and tar binary stream in tuples
        length: a nominal length of the datapipe
    """

    def __init__(self, datapipe: Iterable[Tuple[str, BufferedIOBase]], mode: str = "r:*", length: int = -1):
        super().__init__()
        self.datapipe: Iterable[Tuple[str, BufferedIOBase]] = datapipe
        self.mode = mode
        self.length: int = length

    def __iter__(self) -> Iterator[Tuple[str, BufferedIOBase]]:
        if not isinstance(self.datapipe, Iterable):
            raise TypeError("datapipe must be Iterable type but got {}".format(type(self.datapipe)))
        for data in self.datapipe:
            validate_pathname_binary_tuple(data)
            pathname, data_stream = data
            try:
                reading_mode = (
                    self.mode
                    if hasattr(data_stream, "seekable") and data_stream.seekable()
                    else self.mode.replace(":", "|")
                )
                # typing.cast is used here to silence mypy's type checker
                tar = tarfile.open(fileobj=cast(Optional[IO[bytes]], data_stream), mode=reading_mode)
                for tarinfo in tar:
                    if not tarinfo.isfile():
                        continue
                    extracted_fobj = tar.extractfile(tarinfo)
                    if extracted_fobj is None:
                        warnings.warn("failed to extract file {} from source tarfile {}".format(tarinfo.name, pathname))
                        raise tarfile.ExtractError
                    inner_pathname = os.path.normpath(os.path.join(pathname, tarinfo.name))
                    yield inner_pathname, StreamWrapper(extracted_fobj, data_stream, name=inner_pathname)
            except Exception as e:
                warnings.warn(
                    "Unable to extract files from corrupted tarfile stream {} due to: {}, abort!".format(pathname, e)
                )
                raise e
            finally:
                if isinstance(data_stream, StreamWrapper):
                    data_stream.autoclose()

    def __len__(self):
        if self.length == -1:
            raise NotImplementedError
        return self.length
