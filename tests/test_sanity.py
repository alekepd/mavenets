"""Basic sanity tests that don't require optional dependencies.

These tests verify the testing framework itself works correctly.
"""

import pytest
import torch


class TestPytestSetup:
    """Verify pytest is configured correctly."""

    def test_pytest_runs(self) -> None:
        """Basic test to verify pytest executes."""
        assert True

    def test_torch_available(self) -> None:
        """Verify PyTorch is importable."""
        x = torch.tensor([1, 2, 3])
        assert x.sum().item() == 6

    def test_fixtures_work(self, random_seed: int) -> None:
        """Verify fixtures from conftest.py are available."""
        assert random_seed == 42

    def test_device_fixture(self, cpu_device: str) -> None:
        """Verify device fixture works."""
        assert cpu_device == "cpu"


class TestMarkers:
    """Verify custom markers work correctly."""

    @pytest.mark.slow
    def test_slow_marker(self) -> None:
        """Test with slow marker (can be skipped with -m 'not slow')."""
        assert True

    @pytest.mark.integration
    def test_integration_marker(self) -> None:
        """Test with integration marker."""
        assert True
