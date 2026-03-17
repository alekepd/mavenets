"""Unit tests for t5_pca_encode with mocked T5 model."""

from unittest.mock import patch, MagicMock

import pytest
import torch

from mavenets.data.featurize.transform import IncrementalPCATransform  # type: ignore[import-not-found]
from mavenets.data.featurize.core import get_default_int_encoder  # type: ignore[import-not-found]


def _make_mock_t5_wrapper(seq_len: int = 201, embed_dim: int = 1024):
    """Create a mock T5EncoderWrapper that returns random per-residue embeddings.

    The mock returns tensors of shape (batch, seq_len+1, embed_dim) from
    vectorized_encode, mimicking the T5 output with EOS token.
    """
    mock_wrapper = MagicMock()
    # seq_len + 1 to account for EOS token
    t5_seq_len = seq_len + 1

    def fake_vectorized_encode(int_encoded: torch.Tensor) -> torch.Tensor:
        batch = int_encoded.shape[0]
        torch.manual_seed(hash(int_encoded.data_ptr()) % (2**31))
        return torch.randn(batch, t5_seq_len, embed_dim)

    mock_wrapper.vectorized_encode = fake_vectorized_encode
    return mock_wrapper, t5_seq_len


class TestT5PcaEncodePerResidue:
    """Test t5_pca_encode in per-residue PCA mode."""

    @patch("mavenets.data.featurize.t5.T5EncoderWrapper")
    def test_output_shape_fit(self, mock_cls: MagicMock) -> None:
        """Test output shape when fitting and transforming."""
        n, seq_len, embed_dim, k = 40, 201, 1024, 5
        mock_wrapper, t5_seq_len = _make_mock_t5_wrapper(seq_len, embed_dim)
        mock_cls.return_value = mock_wrapper

        enc = get_default_int_encoder()
        int_encoded = torch.randint(0, len(enc.alphabet), (n, seq_len))

        pca = IncrementalPCATransform(n_components=k, per_residue=True)

        from mavenets.data.featurize.t5 import t5_pca_encode

        result = t5_pca_encode(
            int_encoded=int_encoded,
            pca_transform=pca,
            integer_encoder=enc,
            device="cpu",
            fit_pca=True,
            batch_size=16,
        )

        assert result.shape == (n, t5_seq_len * k)
        assert result.dtype == torch.float32
        assert pca.already_fit

    @patch("mavenets.data.featurize.t5.T5EncoderWrapper")
    def test_output_shape_transform_only(self, mock_cls: MagicMock) -> None:
        """Test output shape when using a pre-fitted PCA (val/test path)."""
        n_train, n_val, seq_len, embed_dim, k = 40, 20, 201, 1024, 5
        mock_wrapper, t5_seq_len = _make_mock_t5_wrapper(seq_len, embed_dim)
        mock_cls.return_value = mock_wrapper

        enc = get_default_int_encoder()
        train_data = torch.randint(0, len(enc.alphabet), (n_train, seq_len))
        val_data = torch.randint(0, len(enc.alphabet), (n_val, seq_len))

        pca = IncrementalPCATransform(n_components=k, per_residue=True)

        from mavenets.data.featurize.t5 import t5_pca_encode

        # Fit on train
        t5_pca_encode(
            int_encoded=train_data,
            pca_transform=pca,
            integer_encoder=enc,
            device="cpu",
            fit_pca=True,
            batch_size=16,
        )

        # Transform val
        result = t5_pca_encode(
            int_encoded=val_data,
            pca_transform=pca,
            integer_encoder=enc,
            device="cpu",
            fit_pca=False,
            batch_size=16,
        )

        assert result.shape == (n_val, t5_seq_len * k)

    @patch("mavenets.data.featurize.t5.T5EncoderWrapper")
    def test_transform_without_fit_raises(self, mock_cls: MagicMock) -> None:
        """Test that transform-only with unfitted PCA raises."""
        n, seq_len, embed_dim, k = 10, 201, 1024, 5
        mock_wrapper, _ = _make_mock_t5_wrapper(seq_len, embed_dim)
        mock_cls.return_value = mock_wrapper

        enc = get_default_int_encoder()
        int_encoded = torch.randint(0, len(enc.alphabet), (n, seq_len))

        pca = IncrementalPCATransform(n_components=k, per_residue=True)

        from mavenets.data.featurize.t5 import t5_pca_encode

        with pytest.raises(ValueError, match="not yet fit"):
            t5_pca_encode(
                int_encoded=int_encoded,
                pca_transform=pca,
                integer_encoder=enc,
                device="cpu",
                fit_pca=False,
                batch_size=16,
            )

    @patch("mavenets.data.featurize.t5.T5EncoderWrapper")
    def test_t5_wrapper_created_correctly(self, mock_cls: MagicMock) -> None:
        """Test that T5EncoderWrapper is created with per_protein=False, flatten=False."""
        mock_wrapper, _ = _make_mock_t5_wrapper()
        mock_cls.return_value = mock_wrapper

        enc = get_default_int_encoder()
        int_encoded = torch.randint(0, len(enc.alphabet), (10, 201))
        pca = IncrementalPCATransform(n_components=3, per_residue=True)

        from mavenets.data.featurize.t5 import t5_pca_encode

        t5_pca_encode(
            int_encoded=int_encoded,
            pca_transform=pca,
            integer_encoder=enc,
            device="cpu",
            fit_pca=True,
        )

        mock_cls.assert_called_once_with(
            integer_encoder=enc,
            device="cpu",
            per_protein=False,
            flatten=False,
            batch_size=32,
        )


class TestT5PcaEncodeGlobal:
    """Test t5_pca_encode in global PCA mode."""

    @patch("mavenets.data.featurize.t5.T5EncoderWrapper")
    def test_output_shape_fit(self, mock_cls: MagicMock) -> None:
        """Test output shape with global PCA."""
        n, seq_len, embed_dim, k = 40, 201, 1024, 5
        mock_wrapper, _ = _make_mock_t5_wrapper(seq_len, embed_dim)
        mock_cls.return_value = mock_wrapper

        enc = get_default_int_encoder()
        int_encoded = torch.randint(0, len(enc.alphabet), (n, seq_len))

        pca = IncrementalPCATransform(n_components=k, per_residue=False)

        from mavenets.data.featurize.t5 import t5_pca_encode

        result = t5_pca_encode(
            int_encoded=int_encoded,
            pca_transform=pca,
            integer_encoder=enc,
            device="cpu",
            fit_pca=True,
            batch_size=16,
        )

        assert result.shape == (n, k)
        assert result.dtype == torch.float32
