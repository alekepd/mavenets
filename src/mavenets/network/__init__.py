"""Provides trainable networks."""

from typing import TYPE_CHECKING

# Eager imports for modules without heavy dependencies
from .transformer import SumTransformer  # noqa: F401
from .tune import (
    MHTuner,  # noqa: F401
    FFNTuner,  # noqa: F401
    SharedFanTuner,  # noqa: F401
    NullTuner,  # noqa: F401
    LinearTuner,  # noqa: F401
)
from .base import MLP  # noqa: F401
from .nontraditional import LRMLP  # noqa: F401

# Lazy imports for modules requiring torch_scatter/torch_geometric
_lazy_imports = {
    "GraphNet": ".mpn",
    "T5LoRAModel": ".t5lora",
}

if TYPE_CHECKING:
    from .mpn import GraphNet  # noqa: F401
    from .t5lora import T5LoRAModel  # noqa: F401


def __getattr__(name: str):
    if name in _lazy_imports:
        import importlib

        module = importlib.import_module(_lazy_imports[name], __package__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(globals().keys()) + list(_lazy_imports.keys())
