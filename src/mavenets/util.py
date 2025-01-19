"""Basic routines used elsewhere in the library.

This module should not import any other modules in the package.
"""
from typing import Iterator, Sequence, TypeVar, Any, Callable, Optional, Generic
from contextlib import contextmanager
from operator import lt
from torch import Tensor, sign, flatten

_T = TypeVar("_T")


# the noqa directive is for unused arguments, which are intended here.
@contextmanager
def null_contextmanager(*args, **kwargs) -> Iterator[None]:  # noqa: ARG001
    """Context manager that does nothing.

    Useful as alternative in meta code selecting various possible managers. All
    arguments are ignored.
    """
    try:
        yield None
    finally:
        pass


def sequence_mean(s: Sequence[_T], start: Any = 0) -> _T:
    """Calculate the mean of a sequence.

    Sequence is first summed, and then the result is divided by the length.

    This function cannot be easily type-hinted in a safe way when using generic types.

    Arguments:
    ---------
    s:
        Sequence to iterate over.
    start:
        Passed to sum.

    Returns:
    -------
    sum divided by length.

    """
    length = float(len(s))
    return sum(s, start=start) / length  # type: ignore


class Patience(Generic[_T]):
    """Repeatedly calculates whether a new element is better than a held element.

    If better, the new element is then held.

    This class is useful for patience-based stopping of training, where it can be
    used to report how long ago a better validation loss was observed.

    """

    # we ignore type errors as the type signature for lt is complex, and furthermore
    # does not apply to arbitrary types.
    def __init__(self, comparison: Callable[[_T, _T], bool] = lt) -> None:  # type: ignore
        """Store options.

        Arguments:
        ---------
        comparison:
            2-argument Callable used to compare new elements to previous elements.
            Should return True if the second argument is better than the first argument,
            False otherwise. Defaults to the function form of '<'

        """
        self.comparison = comparison
        self.counter = 0
        self.saved: Optional[_T] = None

    def consider(self, new: _T) -> int:
        """Consider a new element.

        If held item is better, we return the how long ago the held item was proposed.
        If the new item is better, we return 0 and make it the held item.

        Arguments:
        ---------
        new:
            item to consider.

        Returns:
        -------
        Integer describing how long ago the best item was found.

        """
        if self.saved is None or self.comparison(new, self.saved):
            self.saved = new
            self.counter = 0
        else:
            self.counter += 1
        return self.counter

    def __call__(self, new: _T) -> int:
        """Consider a new element.

        If held item is better, we return the how long ago the held item was proposed.
        If the new item is better, we return 0 and make it the held item.

        Arguments:
        ---------
        new:
            item to consider.

        Returns:
        -------
        Integer describing how long ago the best item was found.

        """
        return self.consider(new)


def num_changes(first: Tensor, second: Tensor, /, *, batch_axis: bool = True) -> Tensor:
    """Calculate the number of differing elements between two tensors.

    Arguments:
    ---------
    first:
        first and second are the two tensors we are comparing. They compatible in
        shape based on broadcasting using "-". If batch_axis is True, the result after
        broadcasting must be of rank at least 2.
    second:
        see first argument.
    batch_axis:
        If true, we assume that first and second contain `n` different cases, indexed
        by the first axis. We return the count for each case.

    Returns:
    -------
    Tensor containing the number of differences. If batch_axis, one-dimensional with
    length equal to that of the first axes of first and second. Else, zero-dimensional
    tensor.

    """
    flags = sign(first - second).abs()
    if batch_axis:
        # using flatten allows us to avoid python logic. Combine all axes past the 
        # first to make a two dimensional tensor.
        flat = flatten(flags, start_dim=1)
        return flat.sum(dim=1)
    else:
        return flags.sum()
