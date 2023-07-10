"""
Simple timer.
"""
from time import perf_counter


class TimeManager(object):
    def __init__(self):
        """Initialization"""
        self._starts = {}
        self.elapsed = {}

    def start(self, name: str) -> None:
        """Starts timer ``name``.

        Args:
            name: name of a timer
        """
        self._starts[name] = perf_counter()

    def stop(self, name: str) -> float:
        """Stops timer ``name``.

        Args:
            name: name of a timer
        """
        if name not in self._starts:
            raise KeyError(f"Timer '{name}' wasn't started. Current timers: {self._starts.keys()}")

        elapsed = perf_counter() - self._starts[name]
        self.elapsed[name] = elapsed
        del self._starts[name]
        return elapsed

    def reset(self) -> None:
        """Reset all previous timers."""
        self.elapsed = {}
        self._starts = {}


__all__ = ["TimeManager"]
