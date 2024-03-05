from abc import ABC
from abc import abstractmethod


class Writer(ABC):
    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.close()

    @abstractmethod
    def write(self, obj):
        raise NotImplementedError

    @abstractmethod
    def close(self):
        raise NotImplementedError
