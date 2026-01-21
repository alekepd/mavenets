"""Tests for mavenets.data.featurize.core module."""

import pytest
import torch

pytest.importorskip("torch_geometric", reason="torch_geometric required for data module")

from mavenets.data.featurize.core import (
    IntEncoder,
    get_default_int_encoder,
    int_to_floatonehot,
    encoder_dict,
    decoder_dict,
    BASE_ALPHA,
)


class TestEncoderDecoder:
    """Tests for encoder_dict and decoder_dict functions."""

    def test_encoder_dict_creates_mapping(self, sample_alphabet: list[str]) -> None:
        """Encoder dict should map each letter to a unique integer."""
        enc = encoder_dict(sample_alphabet)
        assert len(enc) == len(sample_alphabet)
        assert set(enc.values()) == set(range(len(sample_alphabet)))

    def test_decoder_dict_creates_inverse_mapping(
        self, sample_alphabet: list[str]
    ) -> None:
        """Decoder dict should map integers back to letters."""
        dec = decoder_dict(sample_alphabet)
        assert len(dec) == len(sample_alphabet)
        for i, letter in enumerate(sample_alphabet):
            assert dec[i] == letter

    def test_encoder_decoder_roundtrip(self, sample_alphabet: list[str]) -> None:
        """Encoding then decoding should return original letter."""
        enc = encoder_dict(sample_alphabet)
        dec = decoder_dict(sample_alphabet)
        for letter in sample_alphabet:
            assert dec[enc[letter]] == letter


class TestIntEncoder:
    """Tests for IntEncoder class."""

    def test_init_with_valid_alphabet(self, sample_alphabet: list[str]) -> None:
        """IntEncoder should initialize with a valid alphabet."""
        encoder = IntEncoder(sample_alphabet)
        assert len(encoder) == len(sample_alphabet)
        assert encoder.alphabet == sample_alphabet

    def test_init_with_duplicate_alphabet_raises(self) -> None:
        """IntEncoder should raise ValueError for duplicate alphabet elements."""
        with pytest.raises(ValueError, match="unique elements"):
            IntEncoder(["A", "B", "A"])

    def test_encode_to_list(self, sample_alphabet: list[str]) -> None:
        """Encoding to list should return list of integers."""
        encoder = IntEncoder(sample_alphabet)
        result = encoder.encode("ACE", tensor=False)
        assert isinstance(result, list)
        assert len(result) == 3
        assert all(isinstance(x, int) for x in result)

    def test_encode_to_tensor(
        self, sample_alphabet: list[str], cpu_device: str
    ) -> None:
        """Encoding to tensor should return torch.Tensor."""
        encoder = IntEncoder(sample_alphabet)
        result = encoder.encode("ACE", tensor=True, device=cpu_device)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3,)
        assert result.dtype == torch.int32

    def test_decode_from_list(self, sample_alphabet: list[str]) -> None:
        """Decoding from list should return original string."""
        encoder = IntEncoder(sample_alphabet)
        encoded = encoder.encode("ACDEF", tensor=False)
        decoded = encoder.decode(encoded)
        assert decoded == "ACDEF"

    def test_decode_from_tensor(
        self, sample_alphabet: list[str], cpu_device: str
    ) -> None:
        """Decoding from tensor should return original string."""
        encoder = IntEncoder(sample_alphabet)
        encoded = encoder.encode("ACDEF", tensor=True, device=cpu_device)
        decoded = encoder.decode(encoded)
        assert decoded == "ACDEF"

    def test_encode_decode_roundtrip(
        self, sample_alphabet: list[str], sample_sequence: str
    ) -> None:
        """Encoding then decoding should return original sequence."""
        encoder = IntEncoder(sample_alphabet)
        encoded = encoder.encode(sample_sequence, tensor=False)
        decoded = encoder.decode(encoded)
        assert decoded == sample_sequence

    def test_batch_encode(
        self, sample_alphabet: list[str], cpu_device: str
    ) -> None:
        """Batch encoding should return stacked tensor."""
        encoder = IntEncoder(sample_alphabet)
        sequences = ["ACE", "DEF", "GHI"]
        result = encoder.batch_encode(sequences, device=cpu_device)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 3)

    def test_batch_decode(self, sample_alphabet: list[str]) -> None:
        """Batch decoding should return list of strings."""
        encoder = IntEncoder(sample_alphabet)
        sequences = ["ACE", "DEF", "GHI"]
        encoded = encoder.batch_encode(sequences)
        decoded = encoder.batch_decode(encoded)
        assert decoded == sequences

    def test_len(self, sample_alphabet: list[str]) -> None:
        """Length should return alphabet size."""
        encoder = IntEncoder(sample_alphabet)
        assert len(encoder) == len(sample_alphabet)

    def test_lru_cache_enabled(self, sample_alphabet: list[str]) -> None:
        """LRU cache should be enabled when cache_size is provided."""
        encoder = IntEncoder(sample_alphabet, lru_cache_size=100)
        # Encode same sequence twice - should use cache
        result1 = encoder.encode("ACE", tensor=False)
        result2 = encoder.encode("ACE", tensor=False)
        assert result1 == result2


class TestGetDefaultIntEncoder:
    """Tests for get_default_int_encoder function."""

    def test_returns_encoder(self) -> None:
        """Should return an IntEncoder instance."""
        encoder = get_default_int_encoder()
        assert isinstance(encoder, IntEncoder)

    def test_uses_base_alphabet(self) -> None:
        """Default encoder should use BASE_ALPHA alphabet."""
        encoder = get_default_int_encoder()
        assert encoder.alphabet == BASE_ALPHA

    def test_with_cache_size(self) -> None:
        """Should accept cache_size parameter."""
        encoder = get_default_int_encoder(cache_size=100)
        assert isinstance(encoder, IntEncoder)


class TestIntToFloatOnehot:
    """Tests for int_to_floatonehot function."""

    def test_basic_conversion(self, cpu_device: str) -> None:
        """Should convert integer tensor to one-hot float tensor."""
        int_tensor = torch.tensor([0, 1, 2], device=cpu_device, dtype=torch.int32)
        result = int_to_floatonehot(int_tensor, num_classes=3)
        expected = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            device=cpu_device,
        )
        assert torch.allclose(result, expected)

    def test_output_dtype(self, cpu_device: str) -> None:
        """Output should be float32."""
        int_tensor = torch.tensor([0, 1], device=cpu_device, dtype=torch.int32)
        result = int_to_floatonehot(int_tensor, num_classes=2)
        assert result.dtype == torch.float32

    def test_auto_num_classes(self, cpu_device: str) -> None:
        """Should infer num_classes when set to -1."""
        int_tensor = torch.tensor([0, 1, 2], device=cpu_device, dtype=torch.int32)
        result = int_to_floatonehot(int_tensor, num_classes=-1)
        assert result.shape == (3, 3)

    def test_2d_input(self, cpu_device: str) -> None:
        """Should handle 2D input tensors."""
        int_tensor = torch.tensor(
            [[0, 1], [2, 0]], device=cpu_device, dtype=torch.int32
        )
        result = int_to_floatonehot(int_tensor, num_classes=3)
        assert result.shape == (2, 2, 3)

    def test_specified_num_classes_larger(self, cpu_device: str) -> None:
        """Should handle num_classes larger than max value."""
        int_tensor = torch.tensor([0, 1], device=cpu_device, dtype=torch.int32)
        result = int_to_floatonehot(int_tensor, num_classes=5)
        assert result.shape == (2, 5)
