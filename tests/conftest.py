"""Shared pytest fixtures and configuration for mavenets tests."""

from typing import Tuple
from unittest.mock import MagicMock
import sys

import pytest
import torch
from torch.utils.data import TensorDataset


# Mock torch_geometric if not installed, to allow importing mavenets modules
# that have torch_geometric imports at module level (tools.py, report.py, etc.)
# This must happen before any mavenets imports that depend on torch_geometric.
if "torch_geometric" not in sys.modules:
    try:
        import torch_geometric  # noqa: F401
    except ImportError:
        # torch_geometric not installed, create mock
        mock_pyg = MagicMock()
        mock_pyg_loader = MagicMock()
        mock_pyg_data = MagicMock()
        sys.modules["torch_geometric"] = mock_pyg
        sys.modules["torch_geometric.loader"] = mock_pyg_loader
        sys.modules["torch_geometric.data"] = mock_pyg_data
        mock_pyg.loader = mock_pyg_loader
        mock_pyg.data = mock_pyg_data


def create_synthetic_datasets(
    n_train: int = 200,
    n_val: int = 50,
    n_features: int = 100,
    n_heads: int = 3,
    device: str = "cpu",
    seed: int = 42,
) -> Tuple[TensorDataset, TensorDataset]:
    """Create synthetic train and validation datasets.

    The data is structured to mimic the MAVE experiment format:
    - Features: one-hot encoded sequences (flattened)
    - Signal: target values to predict
    - Experiment index: which tuning head to use

    Arguments:
    ---------
    n_train:
        Number of training samples.
    n_val:
        Number of validation samples.
    n_features:
        Size of input features (mimics flattened one-hot sequences).
    n_heads:
        Number of experiment heads.
    device:
        Torch device.
    seed:
        Random seed for reproducibility.

    Returns:
    -------
    Tuple of (train_dataset, valid_dataset).

    """
    torch.manual_seed(seed)

    # Create training data
    X_train = torch.randn(n_train, n_features, device=device)
    # Create targets with some structure (head-dependent linear combination)
    y_train = torch.zeros(n_train, device=device)
    exp_train = torch.randint(0, n_heads, (n_train,), device=device)
    for head in range(n_heads):
        mask = exp_train == head
        # Each head has a different relationship with features
        weights = torch.randn(n_features, device=device) * (head + 1) * 0.1
        y_train[mask] = (X_train[mask] @ weights) + torch.randn(mask.sum(), device=device) * 0.1

    train_dataset = TensorDataset(X_train, y_train, exp_train)

    # Create validation data with same structure
    X_val = torch.randn(n_val, n_features, device=device)
    y_val = torch.zeros(n_val, device=device)
    exp_val = torch.randint(0, n_heads, (n_val,), device=device)
    for head in range(n_heads):
        mask = exp_val == head
        weights = torch.randn(n_features, device=device) * (head + 1) * 0.1
        y_val[mask] = (X_val[mask] @ weights) + torch.randn(mask.sum(), device=device) * 0.1

    valid_dataset = TensorDataset(X_val, y_val, exp_val)

    return train_dataset, valid_dataset


@pytest.fixture
def device() -> str:
    """Return available device (cuda if available, else cpu)."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def cpu_device() -> str:
    """Force CPU device for deterministic tests."""
    return "cpu"


@pytest.fixture
def sample_sequence() -> str:
    """Return a short amino acid sequence for testing."""
    return "ACDEFGHIKLMNPQRSTVWY"


@pytest.fixture
def sample_alphabet() -> list[str]:
    """Return the standard amino acid alphabet."""
    return [
        "A", "C", "D", "E", "F", "G", "H", "I", "K", "L",
        "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y", "X",
    ]


@pytest.fixture
def small_batch_size() -> int:
    """Return a small batch size for fast tests."""
    return 4


@pytest.fixture
def random_seed() -> int:
    """Return a fixed random seed for reproducibility."""
    return 42


@pytest.fixture(autouse=True)
def set_random_seed(random_seed: int) -> None:
    """Set random seeds for reproducibility in all tests."""
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")
    config.addinivalue_line("markers", "integration: marks integration tests")


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip GPU tests if no GPU is available."""
    if not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(reason="GPU not available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)
