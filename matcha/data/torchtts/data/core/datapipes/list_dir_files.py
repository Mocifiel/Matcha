import fnmatch
import os
import warnings
from typing import List, Union, Iterable, Iterator

from torchtts.data.core.datapipes.iter_datapipe import IterDataPipe


class ListDirFilesIterDataPipe(IterDataPipe):
    r""":class:`ListDirFilesIterDataPipe`
    Iterable DataPipe to load file pathname(s) (path + filename), yield pathname from given disk root dir.
    args:
        root : root dir
        mask : a unix style filter string or string list for filtering file name(s)
        abspath : whether to return relative pathname or absolute pathname
    """

    def __init__(
        self, root: str = ".", masks: Union[str, List[str]] = "", *, recursive: bool = False, abspath: bool = False
    ):
        super().__init__()
        self.root: str = root
        self.masks: Union[str, List[str]] = masks
        self.recursive: bool = recursive
        self.abspath: bool = abspath

    def __iter__(self) -> Iterator[str]:
        yield from get_file_pathnames_from_root(self.root, self.masks, self.recursive, self.abspath)

    # We do not provide __len__ here because this may prevent triggering some
    # fallback behavior. E.g., the built-in `list(X)` tries to call `len(X)`
    # first, and executes a different code path if the method is not found or
    # `NotImplemented` is returned, while raising an `NotImplementedError`
    # will propagate and and make the call fail where it could have use
    # __iter__` to complete the call.


def match_masks(name: str, masks: Union[str, List[str]]) -> bool:
    # empty mask matches any input name
    if not masks:
        return True

    if isinstance(masks, str):
        return fnmatch.fnmatch(name, masks)

    for mask in masks:
        if fnmatch.fnmatch(name, mask):
            return True
    return False


def get_file_pathnames_from_root(
    root: str, masks: Union[str, List[str]], recursive: bool = False, abspath: bool = False
) -> Iterable[str]:
    # print out an error message and raise the error out
    def onerror(err: OSError):
        warnings.warn(err.filename + " : " + err.strerror)
        raise err

    for path, _, files in os.walk(root, onerror=onerror):
        if abspath:
            path = os.path.abspath(path)
        for f in files:
            if match_masks(f, masks):
                yield os.path.join(path, f)
        if not recursive:
            break
