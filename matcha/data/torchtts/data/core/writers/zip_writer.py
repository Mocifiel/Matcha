import io
import logging
import zipfile

from torchtts.data.core.writers.base_writer import Writer

logger = logging.getLogger(__name__)


class ZipWriter(Writer):
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
        zip_writer = ZipWriter(stream)
        image = imread("b.jpg")
        image2 = imread("b.out.jpg")
        sample = {"__key__": "a/b", "png": image, "output.png": image2}
        zip_writer.write(sample)
    ```
    """

    def __init__(self, filename, mode=0o0444, compress=False):
        if compress:
            compression = zipfile.ZIP_DEFLATED
        else:
            compression = zipfile.ZIP_STORED

        self.mode = mode
        self.compress = compress

        self.zip_stream = zipfile.ZipFile(filename, mode="w", compression=compression)

    def close(self):
        """Close the tar file."""
        self.zip_stream.close()

    def write(self, obj):
        """Write a dictionary to the zip file."""
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
            zipinfo = zipfile.ZipInfo(key + "." + k)
            zipinfo.external_attr |= self.mode << 16
            stream = io.BytesIO(v)
            self.zip_stream.writestr(zipinfo, stream.getvalue())
            total += len(v)
        return total
