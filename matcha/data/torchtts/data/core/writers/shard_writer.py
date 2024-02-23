import logging

from torchtts.data.core.writers.base_writer import Writer

logger = logging.getLogger(__name__)


class ShardWriter(Writer):
    """Writer wrapper to split data into multiple shards."""

    def __init__(self, pattern, writer_cls, max_count=1000, max_size=3e9, **kwargs):
        self.pattern = pattern
        self.writer_cls = writer_cls
        self.max_count = max_count
        self.max_size = max_size
        self.kwargs = kwargs

        self.writer_stream = None
        self.shard = 0
        self.count = 0
        self.size = 0
        self.total_count = 0
        self.fname = None
        self.next_stream()

    def next_stream(self):
        self.finish()
        self.fname = self.pattern % self.shard
        self.shard += 1
        self.writer_stream = self.writer_cls(self.fname, **self.kwargs)
        self.count = 0
        self.size = 0

    def write(self, obj):
        if self.writer_stream is None or self.count >= self.max_count or self.size > self.max_size:
            self.next_stream()
        size = self.writer_stream.write(obj)
        self.count += 1
        self.size += size
        self.total_count += 1

    def finish(self):
        if self.writer_stream is not None:
            self.writer_stream.close()
            assert self.fname is not None
            self.writer_stream = None

    def close(self):
        logger.info(f"{self.total_count} examples have been written to {self.shard} shards")
        self.finish()
