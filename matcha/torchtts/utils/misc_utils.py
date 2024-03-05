import backoff
import contextlib


class memoized_property(property):  # noqa: N801
    """Descriptor that mimics @property but caches output in member variable."""

    def __get__(self, obj, obj_type=None):
        # See https://docs.python.org/3/howto/descriptor.html#properties
        if obj is None:
            return self
        if self.fget is None:
            raise AttributeError("unreadable attribute")
        attr = "__cached_" + self.fget.__name__
        cached = getattr(obj, attr, None)
        if cached is None:
            cached = self.fget(obj)
            setattr(obj, attr, cached)
        return cached


class class_property(property):  # noqa: N801
    """Descriptor to be used as decorator for @classmethods."""

    def __get__(self, obj, obj_type=None):
        return self.fget.__get__(None, obj_type)()


@contextlib.contextmanager
def temporary_assignment(obj, attr, value):
    """Temporarily assign obj.attr to value."""
    original = getattr(obj, attr)
    setattr(obj, attr, value)
    try:
        yield
    finally:
        setattr(obj, attr, original)


@backoff.on_exception(backoff.expo, Exception, max_tries=5)
def call_with_retry(function, *args, **kwargs):
    """Calls a given function with retry."""
    return function(*args, **kwargs)
