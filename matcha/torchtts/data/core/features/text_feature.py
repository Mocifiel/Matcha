from torchtts.data.core.features import tensor_feature
from torchtts.data.core.datapipes.utils import is_stream_handle


class Text(tensor_feature.Tensor):
    def __init__(self):
        super().__init__(shape=(), dtype=str)

    def encode_example(self, example):
        return example.encode("utf-8")

    def decode_example(self, example):
        if is_stream_handle(example):
            ds = example
            example = example.read()
            ds.close()
        return example.decode("utf-8")
