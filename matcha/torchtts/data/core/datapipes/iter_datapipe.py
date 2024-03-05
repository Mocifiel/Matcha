import functools

from torch.utils.data.dataset import IterableDataset
from typing import TypeVar, Dict, Callable

T_co = TypeVar("T_co", covariant=True)
T = TypeVar("T")


class functional_datapipe(object):  # noqa: N801
    def __init__(self, name):
        self.name = name

    def __call__(self, cls):
        if not issubclass(cls, IterDataPipe):
            raise Exception("Can only decorate IterDataPipe")
        IterDataPipe.register_datapipe_as_function(self.name, cls)
        return cls


class IterDataPipe(IterableDataset[T_co]):
    """The base class for all iterable data pipes."""

    functions: Dict[str, Callable] = {}

    def __getattr__(self, attribute_name):
        if attribute_name in IterDataPipe.functions:
            function = functools.partial(IterDataPipe.functions[attribute_name], self)
            return function
        else:
            raise AttributeError

    @classmethod
    def register_function(cls, function_name, function):
        IterDataPipe.functions[function_name] = function

    @classmethod
    def register_datapipe_as_function(cls, function_name, cls_to_register):
        if function_name in IterDataPipe.functions:
            raise Exception("Unable to add DataPipe function name {} as it is already taken".format(function_name))

        def class_function(dp_cls, source_dp, *args, **kwargs):
            return dp_cls(source_dp, *args, **kwargs)

        function = functools.partial(class_function, cls_to_register)
        IterDataPipe.functions[function_name] = function

    def set_epoch(self, epoch):
        if hasattr(self, "datapipe") and hasattr(self.datapipe, "set_epoch"):
            self.datapipe.set_epoch(epoch)

    def apply_sharding(self, num_of_instances, instance_id):
        if hasattr(self, "datapipe") and hasattr(self.datapipe, "apply_sharding"):
            self.datapipe.apply_sharding(num_of_instances, instance_id)
