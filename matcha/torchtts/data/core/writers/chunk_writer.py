import io
import logging
import struct

from torchtts.data.core.writers.base_writer import Writer

logger = logging.getLogger(__name__)


class ChunkWriter(Writer):
    def __init__(self, filename):
        self.chunk_stream = open(filename, "wb")

    def close(self):
        """Close the tar file."""
        self.chunk_stream.close()

    def write(self, obj):
        """Write a dictionary to the chunk file."""
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
            inner_key = key + "." + k
            # Write key size and data
            key_size = len(inner_key)
            self.chunk_stream.write(struct.pack("<I", key_size))
            self.chunk_stream.write(inner_key.encode("utf-8"))
            # Write value size and data
            value_size = len(v)
            stream = io.BytesIO(v)
            self.chunk_stream.write(struct.pack("<I", value_size))
            self.chunk_stream.write(stream.getvalue())
            total += value_size
        return total
