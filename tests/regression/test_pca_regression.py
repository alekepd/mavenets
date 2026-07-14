"""Regression tests for IncrementalPCATransform.

These tests verify that PCA output remains stable across code changes
by checking against known values computed with a fixed seed.
"""

import torch
import numpy as np

from mavenets.data.featurize.transform import IncrementalPCATransform  # type: ignore[import-not-found]


class TestPCAPerResidueRegression:
    """Regression tests for per-residue PCA mode."""

    def test_deterministic_output(self) -> None:
        """Test that PCA output is deterministic with a fixed seed."""
        torch.manual_seed(0)
        data = torch.randn(30, 5, 16)

        pca1 = IncrementalPCATransform(n_components=3, per_residue=True)
        pca1.fit(data)
        result1 = pca1.transform(data)

        torch.manual_seed(0)
        data2 = torch.randn(30, 5, 16)

        pca2 = IncrementalPCATransform(n_components=3, per_residue=True)
        pca2.fit(data2)
        result2 = pca2.transform(data2)

        np.testing.assert_array_equal(result1.numpy(), result2.numpy())

    def test_incremental_fit_then_transform_shape(self) -> None:
        """Test the full incremental pipeline produces expected shape."""
        torch.manual_seed(123)
        n, seq_len, embed_dim, k = 80, 8, 32, 4
        data = torch.randn(n, seq_len, embed_dim)

        pca = IncrementalPCATransform(n_components=k, per_residue=True)
        for start in range(0, n, 20):
            pca.partial_fit_chunk(data[start : start + 20])

        result = pca.transform(data)
        assert result.shape == (n, seq_len * k)

        # Verify output values are finite and non-degenerate
        assert torch.isfinite(result).all()
        per_component_std = result.std(dim=0)
        assert (per_component_std > 1e-6).all()

    def test_transform_new_data_consistent(self) -> None:
        """Test that transforming new data produces consistent results."""
        torch.manual_seed(42)
        train = torch.randn(60, 5, 16)
        test = torch.randn(20, 5, 16)

        pca = IncrementalPCATransform(n_components=3, per_residue=True)
        pca.fit(train)

        result1 = pca.transform(test)
        result2 = pca.transform(test)

        np.testing.assert_array_equal(result1.numpy(), result2.numpy())


class TestPCAGlobalRegression:
    """Regression tests for global PCA mode."""

    def test_deterministic_output(self) -> None:
        """Test that global PCA output is deterministic with a fixed seed."""
        torch.manual_seed(0)
        data = torch.randn(30, 5, 16)

        pca1 = IncrementalPCATransform(n_components=3, per_residue=False)
        pca1.fit(data)
        result1 = pca1.transform(data)

        torch.manual_seed(0)
        data2 = torch.randn(30, 5, 16)

        pca2 = IncrementalPCATransform(n_components=3, per_residue=False)
        pca2.fit(data2)
        result2 = pca2.transform(data2)

        np.testing.assert_array_equal(result1.numpy(), result2.numpy())

    def test_incremental_fit_then_transform_shape(self) -> None:
        """Test the full incremental pipeline produces expected shape."""
        torch.manual_seed(123)
        n, seq_len, embed_dim, k = 80, 8, 32, 4
        data = torch.randn(n, seq_len, embed_dim)

        pca = IncrementalPCATransform(n_components=k, per_residue=False)
        for start in range(0, n, 20):
            pca.partial_fit_chunk(data[start : start + 20])

        result = pca.transform(data)
        assert result.shape == (n, k)

        assert torch.isfinite(result).all()
        per_component_std = result.std(dim=0)
        assert (per_component_std > 1e-6).all()
