"""Provides interface to training, validation, and test data."""
from .load import get_datasets  # noqa: F401
from .spec import resolve_dataspec, DataSpec, MAX_DATASPEC_INDEX  # noqa: F401
from .graph import LegacyGraphDataReader  # noqa: F401
