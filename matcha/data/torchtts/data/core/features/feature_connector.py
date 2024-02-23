from abc import ABC
from abc import abstractmethod
import collections

from torchtts.data.core.features.tensor_info import TensorInfo
from torchtts.utils.dict_utils import map_nested


class FeatureConnector(ABC):
    """Abstract base class for feature types.

    This class provides an interface between the way the information is stored
    on disk, and the way it is presented to the user.
    Here is a diagram on how FeatureConnector methods fit into the data
    generation/reading:

    ```
    generator => encode_example() => Features => decode_example() => data dict
    ```

    The connector can either get raw or dictionary values as input, depending on
    the connector type.
    """

    # Keep track of all sub-classes
    _registered_features = {}

    def __init_subclass__(cls):
        """Registers subclasses features"""
        cls._registered_features["{cls.__module__}.{cls.__name__}"] = cls

    @abstractmethod
    def get_tensor_info(self):
        raise NotImplementedError("Must be implemented in subclasses.")

    @property
    def shape(self):
        return map_nested(lambda t: t.shape, self.get_tensor_info())

    @property
    def dtype(self):
        return map_nested(lambda t: t.dtype, self.get_tensor_info())

    def get_serialized_info(self):
        return self.get_tensor_info()

    @abstractmethod
    def encode_example(self, example):
        raise NotImplementedError("Must be implemented in subclasses.")

    def decode_example(self, example):
        return example

    def _additional_repr_info(self):
        return {}

    def __repr__(self):
        tensor_info = self.get_tensor_info()
        if not isinstance(tensor_info, TensorInfo):
            return "{}({})".format(type(self).__name__, tensor_info)

        # Ensure ordering of keys by adding them one-by-one
        repr_info = collections.OrderedDict()
        repr_info["shape"] = tensor_info.shape
        repr_info["dtype"] = repr(tensor_info.dtype)
        additional_info = self._additional_repr_info()
        for k, v in additional_info.items():
            repr_info[k] = v

        info_str = ", ".join(["%s=%s" % (k, v) for k, v in repr_info.items()])
        return "{}({})".format(
            type(self).__name__,
            info_str,
        )
