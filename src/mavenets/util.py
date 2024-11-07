"""Basic tools used elsewhere in the library."""
from typing import Iterator, Sequence, TypeVar, Any, Callable, Optional, Generic
from contextlib import contextmanager
from operator import lt

_T = TypeVar("_T")


@contextmanager
def null_contextmanager(*args, **kwargs) -> Iterator[None]:  # noqa: ARG001
    """Context mananger that does nothing.

    Useful as alternative in meta code selecting various possible managers. All
    arguments are ignored.
    """
    try:
        yield None
    finally:
        pass


def sequence_mean(s: Sequence[_T], start: Any = 0) -> _T:
    length = float(len(s))
    return sum(s, start=start) / length #type: ignore


class Patience(Generic[_T]):

    def __init__(self, comparison: Callable[[_T,_T],bool] = lt) -> None: #type: ignore
        self.comparison = comparison
        self.counter = 0
        self.saved: Optional[_T] = None

    def consider(self, new: _T) -> int:
        if self.saved is None or self.comparison(new, self.saved):
            self.saved = new
            self.counter = 0
        else:
            self.counter += 1
        return self.counter
