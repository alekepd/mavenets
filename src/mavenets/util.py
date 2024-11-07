"""Basic tools used elsewhere in the library."""
from typing import Iterator, Sequence, TypeVar, Any
from contextlib import contextmanager

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
