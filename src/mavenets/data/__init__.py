"""Provides interface to training, validation, and test data."""

from typing import TYPE_CHECKING

# Eager imports for modules without heavy dependencies
from .spec import (
    resolve_dataspec,  # noqa: F401
    DataSpec,  # noqa: F401
    MAX_DATASPEC_INDEX,  # noqa: F401
    CORE_DATA_SPECS,  # noqa: F401
    DATA_SPECS,  # noqa: F401
)
from .featurize import (
    get_default_int_encoder,  # noqa: F401
    IntEncoder,  # noqa: F401
    int_to_floatonehot,  # noqa: F401
    SARS_COV2_SEQ,  # noqa: F401
)

# Lazy imports for modules requiring torch_geometric/mdtraj
_lazy_imports = {
    "get_datasets": ".load",
    "SequenceDataset": ".load",
    "LegacyGraphDataReader": ".graph",
}

if TYPE_CHECKING:
    from .load import get_datasets, SequenceDataset  # noqa: F401
    from .graph import LegacyGraphDataReader  # noqa: F401


def __getattr__(name: str):
    if name in _lazy_imports:
        import importlib

        module = importlib.import_module(_lazy_imports[name], __package__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(globals().keys()) + list(_lazy_imports.keys())
