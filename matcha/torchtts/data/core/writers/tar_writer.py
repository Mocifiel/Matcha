import io
import logging
import tarfile

from torchtts.data.core.writers.base_writer import Writer

logger = logging.getLogger(__name__)


class TarWriter(Writer):
    """A class for writing dictionaries to tar files.
    :param fileobj: file name for tar file (.tgz/.tar) or open file descriptor
    :param encoder: sample encoding (Default value = True)
    :param compress:  (Default value = None)
    `True` will use an encoder that behaves similar to the automatic
    decoder for `Dataset`. `False` disables encoding and expects byte strings
    (except for metadata, which must be strings). The `encoder` argument can
    also be a `callable`, or a dictionary mapping extensions to encoders.
    The following code will add two file to the tar archive: `a/b.png` and
    `a/b.output.png`.

    Examples:
    ```Python
        tar_writer = TarWriter(stream)
        image = imread("b.jpg")
        image2 = imread("b.out.jpg")
        sample = {"__key__": "a/b", "png": image, "output.png": image2}
        tar_writer.write(sample)
    ```
    """

    def __init__(self, filename, user="bigdata", group="bigdata", mode=0o0444, compress=False):
        tar_mode = "w|gz" if compress else "w|"
        self.tar_stream = tarfile.open(filename, mode=tar_mode)

        self.user = user
        self.group = group
        self.mode = mode
        self.compress = compress

    def close(self):
        """Close the tar file."""
        self.tar_stream.members = []
        self.tar_stream.close()

    def write(self, obj):
        """Write a dictionary to the tar file."""
        total = 0

        if "__key__" not in obj:
            raise ValueError("object must contain a __key__")

        # Check encoded types
        for k, v in obj.items():
            if k[0] == "_":
                continue
            if not isinstance(v, (bytes, bytearray, memoryview)):
                raise ValueError(f"{k} doesn't map to a bytes after encoding ({type(v)})")

        key = obj["__key__"]
        for k in sorted(obj.keys()):
            if k == "__key__":
                continue
            v = obj[k]
            tarinfo = tarfile.TarInfo(key + "." + k)
            tarinfo.size = len(v)
            tarinfo.mode = self.mode
            stream = io.BytesIO(v)
            self.tar_stream.addfile(tarinfo, stream)
            total += tarinfo.size
        return total
