"""Unit tests for the report module.

These tests verify the prediction reporting utilities.
"""

from typing import Tuple

import pytest
import torch
import numpy as np
from torch.utils.data import TensorDataset
import pandas as pd

from mavenets.report import (  # type: ignore[import-not-found]
    predict,
    REFERENCE_KEY,
    TUNED_PRED_KEY,
    RAW_PRED_KEY,
    EXPID_KEY,
)
from mavenets.network.tune import LinearTuner  # type: ignore[import-not-found]
from mavenets.network.base import MLP  # type: ignore[import-not-found]


class TestConstants:
    """Test module constants."""

    def test_reference_key(self) -> None:
        """Test REFERENCE_KEY constant."""
        assert REFERENCE_KEY == "reference"

    def test_tuned_pred_key(self) -> None:
        """Test TUNED_PRED_KEY constant."""
        assert TUNED_PRED_KEY == "tuned"

    def test_raw_pred_key(self) -> None:
        """Test RAW_PRED_KEY constant."""
        assert RAW_PRED_KEY == "raw"

    def test_expid_key(self) -> None:
        """Test EXPID_KEY constant."""
        assert EXPID_KEY == "experiment"


class TestPredict:
    """Test the predict function."""

    @pytest.fixture
    def simple_tuner(self, cpu_device: str) -> LinearTuner[torch.Tensor]:
        """Create a simple tuner for testing."""
        mlp = MLP(in_size=5, out_size=1, hidden_sizes=[8], post_squeeze=True).to(
            cpu_device
        )
        return LinearTuner(mlp, n_heads=3).to(cpu_device)

    @pytest.fixture
    def simple_dataset(self, cpu_device: str) -> TensorDataset:
        """Create a simple dataset for testing."""
        torch.manual_seed(42)
        X = torch.randn(20, 5, device=cpu_device)
        y = torch.randn(20, device=cpu_device)
        # Use experiment indices 0, 1, 2 (matching tuner n_heads=3)
        exp_idx = torch.tensor([0, 1, 2] * 6 + [0, 1], dtype=torch.long, device=cpu_device)
        return TensorDataset(X, y, exp_idx)

    def test_returns_dataframe(
        self,
        simple_tuner: LinearTuner[torch.Tensor],
        simple_dataset: TensorDataset,
    ) -> None:
        """Test that predict returns a pandas DataFrame."""
        result = predict(
            model=simple_tuner,
            dataset=simple_dataset,
            graph=False,
            translate_experiment_ids=False,
            batch_size=10,
        )
        assert isinstance(result, pd.DataFrame)

    def test_dataframe_has_correct_columns(
        self,
        simple_tuner: LinearTuner[torch.Tensor],
        simple_dataset: TensorDataset,
    ) -> None:
        """Test that the DataFrame has the expected columns."""
        result = predict(
            model=simple_tuner,
            dataset=simple_dataset,
            graph=False,
            translate_experiment_ids=False,
            batch_size=10,
        )
        expected_columns = {REFERENCE_KEY, TUNED_PRED_KEY, RAW_PRED_KEY, EXPID_KEY}
        assert set(result.columns) == expected_columns

    def test_dataframe_has_correct_length(
        self,
        simple_tuner: LinearTuner[torch.Tensor],
        simple_dataset: TensorDataset,
    ) -> None:
        """Test that the DataFrame has the same number of rows as the dataset."""
        result = predict(
            model=simple_tuner,
            dataset=simple_dataset,
            graph=False,
            translate_experiment_ids=False,
            batch_size=10,
        )
        assert len(result) == len(simple_dataset)

    def test_reference_values_match_dataset(
        self,
        simple_tuner: LinearTuner[torch.Tensor],
        simple_dataset: TensorDataset,
        cpu_device: str,
    ) -> None:
        """Test that reference values in DataFrame match dataset targets."""
        result = predict(
            model=simple_tuner,
            dataset=simple_dataset,
            graph=False,
            translate_experiment_ids=False,
            batch_size=10,
        )
        # Extract y values from dataset
        y_values = simple_dataset.tensors[1].numpy()
        np.testing.assert_array_almost_equal(result[REFERENCE_KEY].values, y_values)

    def test_experiment_ids_match_dataset(
        self,
        simple_tuner: LinearTuner[torch.Tensor],
        simple_dataset: TensorDataset,
    ) -> None:
        """Test that experiment IDs in DataFrame match dataset indices."""
        result = predict(
            model=simple_tuner,
            dataset=simple_dataset,
            graph=False,
            translate_experiment_ids=False,
            batch_size=10,
        )
        # Extract experiment indices from dataset
        exp_values = simple_dataset.tensors[2].numpy()
        np.testing.assert_array_equal(result[EXPID_KEY].values, exp_values)

    def test_predictions_are_numeric(
        self,
        simple_tuner: LinearTuner[torch.Tensor],
        simple_dataset: TensorDataset,
    ) -> None:
        """Test that tuned and raw predictions are numeric values."""
        result = predict(
            model=simple_tuner,
            dataset=simple_dataset,
            graph=False,
            translate_experiment_ids=False,
            batch_size=10,
        )
        assert result[TUNED_PRED_KEY].dtype in [np.float32, np.float64]
        assert result[RAW_PRED_KEY].dtype in [np.float32, np.float64]

    def test_predictions_not_nan(
        self,
        simple_tuner: LinearTuner[torch.Tensor],
        simple_dataset: TensorDataset,
    ) -> None:
        """Test that predictions don't contain NaN values."""
        result = predict(
            model=simple_tuner,
            dataset=simple_dataset,
            graph=False,
            translate_experiment_ids=False,
            batch_size=10,
        )
        assert not result[TUNED_PRED_KEY].isna().any()
        assert not result[RAW_PRED_KEY].isna().any()

    def test_model_set_to_eval_mode(
        self,
        simple_tuner: LinearTuner[torch.Tensor],
        simple_dataset: TensorDataset,
    ) -> None:
        """Test that the model is set to eval mode during prediction."""
        simple_tuner.train()  # Start in train mode
        predict(
            model=simple_tuner,
            dataset=simple_dataset,
            graph=False,
            translate_experiment_ids=False,
            batch_size=10,
        )
        # Model should be in eval mode after predict
        assert not simple_tuner.training

    def test_different_batch_sizes(
        self,
        simple_tuner: LinearTuner[torch.Tensor],
        simple_dataset: TensorDataset,
    ) -> None:
        """Test that different batch sizes produce same results.

        Uses approximate comparison for floating-point columns since different
        batch sizes can lead to small numerical differences due to floating-point
        non-associativity.
        """
        result_small = predict(
            model=simple_tuner,
            dataset=simple_dataset,
            graph=False,
            translate_experiment_ids=False,
            batch_size=3,
        )
        result_large = predict(
            model=simple_tuner,
            dataset=simple_dataset,
            graph=False,
            translate_experiment_ids=False,
            batch_size=100,
        )
        pd.testing.assert_frame_equal(
            result_small, result_large, check_exact=False, rtol=1e-4
        )

    def test_translate_experiment_ids_false(
        self,
        simple_tuner: LinearTuner[torch.Tensor],
        simple_dataset: TensorDataset,
    ) -> None:
        """Test that experiment IDs remain as integers when translate=False."""
        result = predict(
            model=simple_tuner,
            dataset=simple_dataset,
            graph=False,
            translate_experiment_ids=False,
            batch_size=10,
        )
        # Experiment IDs should be integers
        assert result[EXPID_KEY].dtype in [np.int32, np.int64]

    def test_translate_experiment_ids_true(
        self,
        simple_tuner: LinearTuner[torch.Tensor],
        simple_dataset: TensorDataset,
    ) -> None:
        """Test that experiment IDs are translated to names when translate=True."""
        result = predict(
            model=simple_tuner,
            dataset=simple_dataset,
            graph=False,
            translate_experiment_ids=True,
            batch_size=10,
        )
        # Experiment IDs should be strings (names from DataSpec)
        assert result[EXPID_KEY].dtype == object  # pandas uses object for strings
        # All values should be strings
        assert all(isinstance(v, str) for v in result[EXPID_KEY].values)

    def test_small_dataset(
        self,
        simple_tuner: LinearTuner[torch.Tensor],
        cpu_device: str,
    ) -> None:
        """Test prediction with a small dataset (2 samples)."""
        X = torch.randn(2, 5, device=cpu_device)
        y = torch.randn(2, device=cpu_device)
        exp_idx = torch.tensor([0, 1], dtype=torch.long, device=cpu_device)
        dataset = TensorDataset(X, y, exp_idx)

        result = predict(
            model=simple_tuner,
            dataset=dataset,
            graph=False,
            translate_experiment_ids=False,
            batch_size=10,
        )
        assert len(result) == 2

    def test_large_dataset(
        self,
        simple_tuner: LinearTuner[torch.Tensor],
        cpu_device: str,
    ) -> None:
        """Test prediction with a larger dataset spanning multiple batches."""
        torch.manual_seed(42)
        n_samples = 500
        X = torch.randn(n_samples, 5, device=cpu_device)
        y = torch.randn(n_samples, device=cpu_device)
        exp_idx = torch.randint(0, 3, (n_samples,), device=cpu_device)
        dataset = TensorDataset(X, y, exp_idx)

        result = predict(
            model=simple_tuner,
            dataset=dataset,
            graph=False,
            translate_experiment_ids=False,
            batch_size=64,
        )
        assert len(result) == n_samples


class TestPredictDeterminism:
    """Test that predictions are deterministic."""

    @pytest.fixture
    def simple_tuner(self, cpu_device: str) -> LinearTuner[torch.Tensor]:
        """Create a simple tuner for testing."""
        mlp = MLP(in_size=5, out_size=1, hidden_sizes=[8], post_squeeze=True).to(
            cpu_device
        )
        return LinearTuner(mlp, n_heads=2).to(cpu_device)

    @pytest.fixture
    def simple_dataset(self, cpu_device: str) -> TensorDataset:
        """Create a simple dataset for testing."""
        torch.manual_seed(42)
        X = torch.randn(15, 5, device=cpu_device)
        y = torch.randn(15, device=cpu_device)
        exp_idx = torch.randint(0, 2, (15,), device=cpu_device)
        return TensorDataset(X, y, exp_idx)

    def test_multiple_calls_same_result(
        self,
        simple_tuner: LinearTuner[torch.Tensor],
        simple_dataset: TensorDataset,
    ) -> None:
        """Test that multiple predict calls give the same result."""
        result1 = predict(
            model=simple_tuner,
            dataset=simple_dataset,
            graph=False,
            translate_experiment_ids=False,
            batch_size=5,
        )
        result2 = predict(
            model=simple_tuner,
            dataset=simple_dataset,
            graph=False,
            translate_experiment_ids=False,
            batch_size=5,
        )
        pd.testing.assert_frame_equal(result1, result2)


class TestPredictWithTunedModel:
    """Test that predictions reflect tuner behavior."""

    @pytest.fixture
    def trained_tuner(self, cpu_device: str) -> LinearTuner[torch.Tensor]:
        """Create and minimally train a tuner."""
        torch.manual_seed(42)
        mlp = MLP(in_size=5, out_size=1, hidden_sizes=[8], post_squeeze=True).to(
            cpu_device
        )
        tuner = LinearTuner(mlp, n_heads=2, residual_connection=True).to(cpu_device)

        # Quick training to differentiate heads
        optimizer = torch.optim.Adam(tuner.parameters(), lr=0.1)
        X = torch.randn(20, 5, device=cpu_device)
        y = torch.randn(20, device=cpu_device)
        head_idx = torch.randint(0, 2, (20,), device=cpu_device)

        for _ in range(50):
            optimizer.zero_grad()
            pred = tuner(X, head_idx)
            loss = torch.nn.functional.mse_loss(pred, y)
            loss.backward()
            optimizer.step()

        return tuner

    def test_tuned_differs_from_raw(
        self,
        trained_tuner: LinearTuner[torch.Tensor],
        cpu_device: str,
    ) -> None:
        """Test that tuned predictions differ from raw predictions."""
        X = torch.randn(10, 5, device=cpu_device)
        y = torch.randn(10, device=cpu_device)
        exp_idx = torch.randint(0, 2, (10,), device=cpu_device)
        dataset = TensorDataset(X, y, exp_idx)

        result = predict(
            model=trained_tuner,
            dataset=dataset,
            graph=False,
            translate_experiment_ids=False,
            batch_size=10,
        )

        # Tuned and raw predictions should generally differ
        # (unless the tuner learned identity, which is unlikely)
        tuned = result[TUNED_PRED_KEY].values
        raw = result[RAW_PRED_KEY].values
        # At least some predictions should differ
        assert not np.allclose(tuned, raw, atol=1e-6)
