import contextlib

from tqdm import auto as tqdm_lib

_active = True


def disable_progress_bar():
    global _active
    _active = False


class EmptyTqdm(object):
    """Dummy tqdm which doesn't do anything."""

    def __init__(self, *args, **kwargs):
        self._iterator = args[0] if args else None

    def __iter__(self):
        return iter(self._iterator)

    def __getattr__(self, _):
        """Return empty function."""

        def empty_fn(*args, **kwargs):
            return

        return empty_fn

    def __enter__(self):
        return self

    def __exit__(self, type_, value, traceback):
        return


def tqdm(*args, **kwargs):
    if _active:
        return tqdm_lib.tqdm(*args, **kwargs)
    else:
        return EmptyTqdm(*args, **kwargs)


def async_tqdm(*args, **kwargs):
    if _active:
        return _async_tqdm(*args, **kwargs)
    else:
        return EmptyTqdm(*args, **kwargs)


@contextlib.contextmanager
def _async_tqdm(*args, **kwargs):
    """Wrapper around Tqdm which can be updated in threads.
    Usage:
    ```
    with utils.async_tqdm(...) as progress_bar:
      # progress_bar can then be modified inside a thread
      # progress_bar.update_total(3)
      # progress_bar.update()
    ```
    Args:
      *args: args of tqdm
      **kwargs: kwargs of tqdm
    Yields:
      progress_bar: Async progress bar which can be shared between threads.
    """
    with tqdm_lib.tqdm(*args, **kwargs) as progress_bar:
        progress_bar = _TqdmProgressBarAsync(progress_bar)
        yield progress_bar
        progress_bar.clear()  # pop progress bar from the active list
        print()  # Avoid the next log to overlap with the bar


class _TqdmProgressBarAsync(object):
    """Wrapper around Tqdm progress bar which be shared between thread."""

    _tqdm_bars = []

    def __init__(self, progress_bar):
        self._lock = tqdm_lib.tqdm.get_lock()
        self._progress_bar = progress_bar
        self._tqdm_bars.append(progress_bar)

    def update_total(self, n=1):
        """Increment total progress bar value."""
        with self._lock:
            self._progress_bar.total += n
            self.refresh()

    def update(self, n=1):
        """Increment current value."""
        with self._lock:
            self._progress_bar.update(n)
            self.refresh()

    def refresh(self):
        """Refresh all."""
        for progress_bar in self._tqdm_bars:
            progress_bar.refresh()

    def clear(self):
        """Remove the tqdm progress bar from the update."""
        self._tqdm_bars.pop()
