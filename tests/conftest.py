"""Shared pytest fixtures and configuration for mavenets tests."""

import pytest
import torch


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
