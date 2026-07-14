"""Unit tests for IncrementalPCATransform."""

import pytest
import torch

from mavenets.data.featurize.transform import IncrementalPCATransform  # type: ignore[import-not-found]


class TestIncrementalPCATransformPerResidue:
    """Test IncrementalPCATransform in per-residue mode."""

    def test_fit_transform_output_shape(self) -> None:
        """Test that fit + transform produces the expected output shape."""
        n, seq_len, embed_dim, k = 50, 10, 32, 5
        data = torch.randn(n, seq_len, embed_dim)
        pca = IncrementalPCATransform(n_components=k, per_residue=True)
        pca.fit(data)
        result = pca.transform(data)
        assert result.shape == (n, seq_len * k)

    def test_partial_fit_produces_valid_transform(self) -> None:
        """Test that incremental fitting produces a valid dimensionality reduction."""
        torch.manual_seed(42)
        n, seq_len, embed_dim, k = 100, 10, 32, 5
        data = torch.randn(n, seq_len, embed_dim)

        pca = IncrementalPCATransform(n_components=k, per_residue=True)
        pca.partial_fit_chunk(data)
        result = pca.transform(data)

        assert result.shape == (n, seq_len * k)
        # Verify output is not degenerate (has non-trivial variance)
        assert result.std() > 0.1

    def test_partial_fit_chunks(self) -> None:
        """Test incremental fitting across multiple chunks."""
        torch.manual_seed(42)
        n, seq_len, embed_dim, k = 60, 10, 32, 3
        data = torch.randn(n, seq_len, embed_dim)

        pca = IncrementalPCATransform(n_components=k, per_residue=True)
        pca.partial_fit_chunk(data[:20])
        pca.partial_fit_chunk(data[20:40])
        pca.partial_fit_chunk(data[40:])

        result = pca.transform(data)
        assert result.shape == (n, seq_len * k)
        assert result.dtype == torch.float32

    def test_transform_before_fit_raises(self) -> None:
        """Test that transforming before fitting raises ValueError."""
        data = torch.randn(10, 5, 8)
        pca = IncrementalPCATransform(n_components=2, per_residue=True)
        with pytest.raises(ValueError, match="not yet fit"):
            pca.transform(data)

    def test_wrong_ndim_raises(self) -> None:
        """Test that 2D input raises ValueError."""
        data = torch.randn(10, 8)
        pca = IncrementalPCATransform(n_components=2, per_residue=True)
        with pytest.raises(ValueError, match="3D"):
            pca.fit(data)

    def test_seq_len_mismatch_raises(self) -> None:
        """Test that mismatched seq_len raises ValueError."""
        pca = IncrementalPCATransform(n_components=2, per_residue=True)
        pca.fit(torch.randn(10, 5, 8))
        with pytest.raises(ValueError, match="Sequence length mismatch"):
            pca.transform(torch.randn(10, 7, 8))

    def test_dimensionality_reduction(self) -> None:
        """Test that PCA actually reduces dimensionality meaningfully."""
        torch.manual_seed(42)
        n, seq_len, embed_dim, k = 100, 5, 32, 3
        data = torch.randn(n, seq_len, embed_dim)

        pca = IncrementalPCATransform(n_components=k, per_residue=True)
        pca.fit(data)
        result = pca.transform(data)

        # Output should have fewer total features than input
        assert result.shape[1] < seq_len * embed_dim
        assert result.shape[1] == seq_len * k

    def test_output_dtype(self) -> None:
        """Test that output is float32."""
        data = torch.randn(20, 5, 8)
        pca = IncrementalPCATransform(n_components=2, per_residue=True)
        pca.fit(data)
        result = pca.transform(data)
        assert result.dtype == torch.float32


class TestIncrementalPCATransformGlobal:
    """Test IncrementalPCATransform in global mode."""

    def test_fit_transform_output_shape(self) -> None:
        """Test that global PCA produces (n, k) output."""
        n, seq_len, embed_dim, k = 50, 10, 32, 5
        data = torch.randn(n, seq_len, embed_dim)
        pca = IncrementalPCATransform(n_components=k, per_residue=False)
        pca.fit(data)
        result = pca.transform(data)
        assert result.shape == (n, k)

    def test_partial_fit_chunks(self) -> None:
        """Test incremental fitting in global mode."""
        torch.manual_seed(42)
        n, seq_len, embed_dim, k = 60, 10, 32, 3
        data = torch.randn(n, seq_len, embed_dim)

        pca = IncrementalPCATransform(n_components=k, per_residue=False)
        pca.partial_fit_chunk(data[:20])
        pca.partial_fit_chunk(data[20:40])
        pca.partial_fit_chunk(data[40:])

        result = pca.transform(data)
        assert result.shape == (n, k)

    def test_transform_before_fit_raises(self) -> None:
        """Test that transforming before fitting raises ValueError."""
        data = torch.randn(10, 5, 8)
        pca = IncrementalPCATransform(n_components=2, per_residue=False)
        with pytest.raises(ValueError, match="not yet fit"):
            pca.transform(data)

    def test_global_differs_from_per_residue(self) -> None:
        """Test that global and per-residue modes produce different shapes."""
        torch.manual_seed(42)
        n, seq_len, embed_dim, k = 50, 10, 32, 3
        data = torch.randn(n, seq_len, embed_dim)

        pca_pr = IncrementalPCATransform(n_components=k, per_residue=True)
        pca_pr.fit(data)
        result_pr = pca_pr.transform(data)

        pca_gl = IncrementalPCATransform(n_components=k, per_residue=False)
        pca_gl.fit(data)
        result_gl = pca_gl.transform(data)

        assert result_pr.shape == (n, seq_len * k)
        assert result_gl.shape == (n, k)
