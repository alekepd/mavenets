"""Tests for mavenets.data.load module."""

import pytest
import torch
from torch.utils.data import TensorDataset, Dataset

from mavenets.data.load import SequenceDataset


class TestSequenceDataset:
    """Tests for SequenceDataset class."""

    @pytest.fixture
    def sample_tensor_dataset(self, cpu_device: str) -> TensorDataset:
        """Create a simple TensorDataset for testing."""
        features = torch.randn(5, 10, device=cpu_device)
        signals = torch.randn(5, device=cpu_device)
        exp_ids = torch.randint(0, 3, (5,), device=cpu_device)
        return TensorDataset(features, signals, exp_ids)

    @pytest.fixture
    def sample_sequences(self) -> tuple:
        """Create sample sequences matching the dataset size."""
        return (
            "ACDEFGHIKL",
            "MNPQRSTVWY",
            "ACMNPQRSVW",
            "DEFGHIKLMN",
            "PQRSTVWYAC",
        )

    def test_init_with_valid_inputs(
        self, sample_tensor_dataset: TensorDataset, sample_sequences: tuple
    ) -> None:
        """SequenceDataset should initialize with matching dataset and sequences."""
        seq_dataset = SequenceDataset(sample_tensor_dataset, sample_sequences)
        assert len(seq_dataset) == 5
        assert seq_dataset.sequences == sample_sequences

    def test_init_with_mismatched_lengths_raises(
        self, sample_tensor_dataset: TensorDataset
    ) -> None:
        """SequenceDataset should raise ValueError for mismatched lengths."""
        wrong_sequences = ("SEQ1", "SEQ2", "SEQ3")  # Only 3, but dataset has 5
        with pytest.raises(ValueError, match="must match"):
            SequenceDataset(sample_tensor_dataset, wrong_sequences)

    def test_len_returns_dataset_length(
        self, sample_tensor_dataset: TensorDataset, sample_sequences: tuple
    ) -> None:
        """__len__ should return the length of the underlying dataset."""
        seq_dataset = SequenceDataset(sample_tensor_dataset, sample_sequences)
        assert len(seq_dataset) == len(sample_tensor_dataset)

    def test_getitem_returns_underlying_data(
        self, sample_tensor_dataset: TensorDataset, sample_sequences: tuple
    ) -> None:
        """__getitem__ should return the same data as the underlying dataset."""
        seq_dataset = SequenceDataset(sample_tensor_dataset, sample_sequences)

        for i in range(len(seq_dataset)):
            # TensorDataset returns a tuple of tensors
            seq_item = seq_dataset[i]
            base_item = sample_tensor_dataset[i]
            assert len(seq_item) == len(base_item)
            for seq_tensor, base_tensor in zip(seq_item, base_item):
                assert torch.equal(seq_tensor, base_tensor)

    def test_get_sequence_returns_correct_sequence(
        self, sample_tensor_dataset: TensorDataset, sample_sequences: tuple
    ) -> None:
        """get_sequence should return the sequence at the given index."""
        seq_dataset = SequenceDataset(sample_tensor_dataset, sample_sequences)

        for i, expected_seq in enumerate(sample_sequences):
            assert seq_dataset.get_sequence(i) == expected_seq

    def test_sequences_property_returns_all_sequences(
        self, sample_tensor_dataset: TensorDataset, sample_sequences: tuple
    ) -> None:
        """sequences property should return the full tuple of sequences."""
        seq_dataset = SequenceDataset(sample_tensor_dataset, sample_sequences)
        assert seq_dataset.sequences == sample_sequences
        assert seq_dataset.sequences is sample_sequences  # Should be same object

    def test_dataset_property_returns_underlying_dataset(
        self, sample_tensor_dataset: TensorDataset, sample_sequences: tuple
    ) -> None:
        """dataset property should return the underlying dataset."""
        seq_dataset = SequenceDataset(sample_tensor_dataset, sample_sequences)
        assert seq_dataset.dataset is sample_tensor_dataset

    def test_is_subclass_of_dataset(
        self, sample_tensor_dataset: TensorDataset, sample_sequences: tuple
    ) -> None:
        """SequenceDataset should be a subclass of torch Dataset."""
        seq_dataset = SequenceDataset(sample_tensor_dataset, sample_sequences)
        assert isinstance(seq_dataset, Dataset)

    def test_works_with_dataloader(
        self, sample_tensor_dataset: TensorDataset, sample_sequences: tuple
    ) -> None:
        """SequenceDataset should work correctly with torch DataLoader."""
        from torch.utils.data import DataLoader

        seq_dataset = SequenceDataset(sample_tensor_dataset, sample_sequences)
        loader = DataLoader(seq_dataset, batch_size=2, shuffle=False)

        batches = list(loader)
        assert len(batches) == 3  # 5 items with batch_size=2 -> 3 batches

        # First batch should have 2 items
        first_batch = batches[0]
        assert first_batch[0].shape[0] == 2  # features batch dim
        assert first_batch[1].shape[0] == 2  # signals batch dim
        assert first_batch[2].shape[0] == 2  # exp_ids batch dim

    def test_empty_dataset(self, cpu_device: str) -> None:
        """SequenceDataset should handle empty datasets."""
        empty_features = torch.randn(0, 10, device=cpu_device)
        empty_signals = torch.randn(0, device=cpu_device)
        empty_exp_ids = torch.randint(0, 3, (0,), device=cpu_device)
        empty_dataset = TensorDataset(empty_features, empty_signals, empty_exp_ids)
        empty_sequences: tuple = ()

        seq_dataset = SequenceDataset(empty_dataset, empty_sequences)
        assert len(seq_dataset) == 0
        assert seq_dataset.sequences == ()

    def test_single_item_dataset(self, cpu_device: str) -> None:
        """SequenceDataset should handle single-item datasets."""
        features = torch.randn(1, 10, device=cpu_device)
        signals = torch.randn(1, device=cpu_device)
        exp_ids = torch.randint(0, 3, (1,), device=cpu_device)
        dataset = TensorDataset(features, signals, exp_ids)
        sequences = ("SINGLESEQ",)

        seq_dataset = SequenceDataset(dataset, sequences)
        assert len(seq_dataset) == 1
        assert seq_dataset.get_sequence(0) == "SINGLESEQ"
        # Compare each tensor in the tuple individually
        seq_item = seq_dataset[0]
        base_item = dataset[0]
        for seq_tensor, base_tensor in zip(seq_item, base_item):
            assert torch.equal(seq_tensor, base_tensor)
