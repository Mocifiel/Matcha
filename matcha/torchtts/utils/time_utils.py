from time import time


class TimeManager(object):
    """Simple time manager."""

    def __init__(self):
        self._starts = {}
        self.elapsed = {}

    def start(self, name: str):
        """Starts timer `name`.
        Args:
            name: name of a timer
        """
        self._starts[name] = time()

    def stop(self, name: str):
        """Stops timer `name`.
        Args:
            name: name of a timer
        """
        assert name in self._starts, f"Timer '{name}' wasn't started"

        self.elapsed[name] = time() - self._starts[name]
        del self._starts[name]

    def reset(self):
        """Reset all previous timers."""
        self.elapsed = {}
        self._starts = {}
