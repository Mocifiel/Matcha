import numpy as np
import pickle

from torchtts.data.core.datapipes.utils import is_stream_handle
from torchtts.data.core.features.feature_connector import FeatureConnector
from torchtts.data.core.features.tensor_info import TensorInfo
from torchtts.data.core.features.tensor_shape import assert_shape_match

DEFAULT_PICKLE_VERSION = 4


class Tensor(FeatureConnector):
    """Generic data of arbitrary shape and type."""

    def __init__(self, *, shape, dtype, serialized_dtype=None, serialized_shape=None):
        self._shape = tuple(shape)
        self._dtype = dtype
        self._serialized_dtype = serialized_dtype or self._dtype
        self._serialized_shape = self._shape if serialized_shape is None else serialized_shape

    def get_tensor_info(self):
        return TensorInfo(shape=self._shape, dtype=self._dtype)

    def encode_example(self, example):
        np_dtype = self._serialized_dtype
        if not isinstance(example, np.ndarray):
            example = np.array(example, dtype=np_dtype)
        # Ensure the shape and dtype match
        if example.dtype != np_dtype:
            raise ValueError("Dtype {} do not match {}".format(example.dtype, self.dtype))
        assert_shape_match(example.shape, self._serialized_shape)
        return pickle.dumps(example, protocol=DEFAULT_PICKLE_VERSION)

    def decode_example(self, example):
        if is_stream_handle(example):
            ds = example
            example = example.read()
            ds.close()
        return pickle.loads(example)
