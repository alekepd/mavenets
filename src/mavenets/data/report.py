"""Tools for evaluating models and recording results."""
from typing import Callable, TypeVar, Iterable
from torch import Tensor
import pandas as pd  # type: ignore

_T = TypeVar("_T")


def evaluate_table(model: Callable[[_T], Tensor], source: Iterable[_T]) -> pd.DataFrame:
    pass
