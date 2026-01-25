"""Regression tests for train_tunable_model.

These tests ensure that modifications to train_tunable_model do not change
the training behavior or outputs. They use a fixed random seed and simple
linear model to produce deterministic, reproducible results.

The expected outputs are stored in fixtures/train_tunable_model_expected.npz
and were generated with the baseline version of the code.

To regenerate the baseline (only do this when intentionally changing behavior):
    REGENERATE_BASELINE=1 pytest tests/regression/test_train_tunable_model_regression.py -k test_generate_baseline
"""

import os
from pathlib import Path
from typing import Tuple

import numpy as np
import numpy.typing as npt
import pytest
import torch
from torch import Tensor
from torch.utils.data import TensorDataset

from mavenets.network.base import MLP  # type: ignore[import-not-found]
from mavenets.network.tune import LinearTuner  # type: ignore[import-not-found]
from mavenets.tools import train_tunable_model  # type: ignore[import-not-found]

# Constants for deterministic test setup
RANDOM_SEED = 98765
N_FEATURES = 10
N_HEADS = 2
N_TRAIN = 50
N_VAL = 20
N_EPOCHS = 25
TRAIN_BATCH_SIZE = 10
REPORT_STRIDE = 5
LEARNING_RATE = 0.05

FIXTURES_DIR = Path(__file__).parent / "fixtures"
EXPECTED_FILE = FIXTURES_DIR / "train_tunable_model_expected.npz"


def _create_deterministic_dataset(
    n_samples: int,
    n_features: int,
    n_heads: int,
    seed_offset: int = 0,
) -> TensorDataset:
    """Create a deterministic synthetic dataset.

    Arguments:
    ---------
    n_samples:
        Number of samples to generate.
    n_features:
        Number of input features.
    n_heads:
        Number of experiment heads.
    seed_offset:
        Offset added to random seed for different datasets.

    Returns:
    -------
    TensorDataset with (X, y, experiment_index) tensors.

    """
    torch.manual_seed(RANDOM_SEED + seed_offset)

    x_data = torch.randn(n_samples, n_features)
    exp_idx = torch.arange(n_samples) % n_heads

    # Create deterministic target: linear combination with head-specific weights
    y_data = torch.zeros(n_samples)
    for head in range(n_heads):
        mask = exp_idx == head
        # Fixed weights per head based on seed
        torch.manual_seed(RANDOM_SEED + head + 1000)
        weights = torch.randn(n_features) * 0.5
        y_data[mask] = x_data[mask] @ weights

    return TensorDataset(x_data, y_data, exp_idx)


def _create_deterministic_model() -> LinearTuner[Tensor]:
    """Create a deterministic linear model (0-hidden-layer MLP with LinearTuner).

    Returns:
    -------
    LinearTuner wrapping a linear MLP.

    """
    torch.manual_seed(RANDOM_SEED + 2000)

    # Linear model: MLP with no hidden layers
    base_model = MLP(
        in_size=N_FEATURES,
        out_size=1,
        hidden_sizes=[],  # No hidden layers = linear model
        post_squeeze=True,
    )
    return LinearTuner(base_model, n_heads=N_HEADS)


def _get_model_predictions(
    model: LinearTuner[Tensor],
    dataset: TensorDataset,
) -> npt.NDArray[np.float64]:
    """Get predictions from model on dataset.

    Arguments:
    ---------
    model:
        The trained model.
    dataset:
        Dataset to get predictions for.

    Returns:
    -------
    Numpy array of predictions.

    """
    model.eval()
    x_data: Tensor = dataset.tensors[0]
    exp_idx: Tensor = dataset.tensors[2]

    with torch.no_grad():
        predictions = model(x_data, exp_idx)

    return predictions.numpy()


def _run_deterministic_training() -> (
    Tuple[
        LinearTuner[Tensor],
        TensorDataset,
        TensorDataset,
        int,
        float,
        npt.NDArray[np.float64],
    ]
):
    """Run training with deterministic settings.

    Returns:
    -------
    Tuple of (model, train_dataset, valid_dataset, best_epoch, best_val, loss_table).

    """
    # Set all seeds for reproducibility
    torch.manual_seed(RANDOM_SEED)

    # Create datasets
    train_dataset = _create_deterministic_dataset(
        n_samples=N_TRAIN,
        n_features=N_FEATURES,
        n_heads=N_HEADS,
        seed_offset=100,
    )
    valid_dataset = _create_deterministic_dataset(
        n_samples=N_VAL,
        n_features=N_FEATURES,
        n_heads=N_HEADS,
        seed_offset=200,
    )

    # Create model
    model = _create_deterministic_model()

    # Create optimizer with fixed seed
    torch.manual_seed(RANDOM_SEED + 3000)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    # Train
    best_epoch, best_val, table = train_tunable_model(
        model=model,
        optimizer=optimizer,
        device="cpu",
        n_epochs=N_EPOCHS,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        train_batch_size=TRAIN_BATCH_SIZE,
        reporting_batch_size=N_TRAIN,  # Evaluate on full dataset
        report_stride=REPORT_STRIDE,
        compile=False,
        train_bfloat16=False,
        progress_bar=False,
        patience=100,  # High patience to avoid early stopping
        start_loss_param=0.5,
        end_loss_param=0.5,  # Fixed loss param (no annealing)
        loss_param_ramp_size=1,
    )

    return (
        model,
        train_dataset,
        valid_dataset,
        int(best_epoch),
        float(best_val),
        table.values,
    )


@pytest.mark.regression
class TestTrainTunableModelRegression:
    """Regression tests for train_tunable_model deterministic behavior."""

    def test_predictions_match_expected(self) -> None:
        """Test that trained model predictions match stored expected values."""
        if not EXPECTED_FILE.exists():
            pytest.skip(
                f"Expected file {EXPECTED_FILE} not found. "
                "Run with REGENERATE_BASELINE=1 to generate."
            )

        # Run training
        model, train_ds, valid_ds, best_epoch, best_val, loss_table = (
            _run_deterministic_training()
        )

        # Load expected values
        expected = np.load(EXPECTED_FILE)

        # Get predictions
        train_preds = _get_model_predictions(model, train_ds)
        valid_preds = _get_model_predictions(model, valid_ds)

        # Compare predictions
        np.testing.assert_allclose(
            train_preds,
            expected["train_predictions"],
            rtol=1e-5,
            atol=1e-6,
            err_msg="Training predictions do not match expected values",
        )

        np.testing.assert_allclose(
            valid_preds,
            expected["valid_predictions"],
            rtol=1e-5,
            atol=1e-6,
            err_msg="Validation predictions do not match expected values",
        )

    def test_training_metrics_match_expected(self) -> None:
        """Test that training metrics (best_epoch, best_val) match expected."""
        if not EXPECTED_FILE.exists():
            pytest.skip(
                f"Expected file {EXPECTED_FILE} not found. "
                "Run with REGENERATE_BASELINE=1 to generate."
            )

        # Run training
        _model, _train_ds, _valid_ds, best_epoch, best_val, loss_table = (
            _run_deterministic_training()
        )

        # Load expected values
        expected = np.load(EXPECTED_FILE)

        # Compare metrics
        assert best_epoch == int(
            expected["best_epoch"]
        ), f"best_epoch mismatch: {best_epoch} != {expected['best_epoch']}"

        np.testing.assert_allclose(
            best_val,
            float(expected["best_val"]),
            rtol=1e-5,
            atol=1e-6,
            err_msg="best_val does not match expected value",
        )

    def test_loss_table_matches_expected(self) -> None:
        """Test that the full loss table matches expected values."""
        if not EXPECTED_FILE.exists():
            pytest.skip(
                f"Expected file {EXPECTED_FILE} not found. "
                "Run with REGENERATE_BASELINE=1 to generate."
            )

        # Run training
        _model, _train_ds, _valid_ds, _best_epoch, _best_val, loss_table = (
            _run_deterministic_training()
        )

        # Load expected values
        expected = np.load(EXPECTED_FILE)

        # Compare loss table
        np.testing.assert_allclose(
            loss_table,
            expected["loss_table"],
            rtol=1e-5,
            atol=1e-6,
            err_msg="Loss table does not match expected values",
        )

    def test_generate_baseline(self) -> None:
        """Generate baseline expected outputs.

        This test only runs when REGENERATE_BASELINE=1 environment variable is set.
        It saves the expected outputs to the fixtures file.
        """
        if not os.environ.get("REGENERATE_BASELINE"):
            pytest.skip("Set REGENERATE_BASELINE=1 to regenerate baseline")

        # Run training
        model, train_ds, valid_ds, best_epoch, best_val, loss_table = (
            _run_deterministic_training()
        )

        # Get predictions
        train_preds = _get_model_predictions(model, train_ds)
        valid_preds = _get_model_predictions(model, valid_ds)

        # Ensure fixtures directory exists
        FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

        # Save expected values
        np.savez(
            EXPECTED_FILE,
            train_predictions=train_preds,
            valid_predictions=valid_preds,
            best_epoch=best_epoch,
            best_val=best_val,
            loss_table=loss_table,
        )

        # Verify the file was created
        assert EXPECTED_FILE.exists(), f"Failed to create {EXPECTED_FILE}"

        # Print summary for user verification
        print(f"\nBaseline saved to {EXPECTED_FILE}")  # noqa: T201
        print(f"  best_epoch: {best_epoch}")  # noqa: T201
        print(f"  best_val: {best_val:.6f}")  # noqa: T201
        print(f"  train_predictions shape: {train_preds.shape}")  # noqa: T201
        print(f"  valid_predictions shape: {valid_preds.shape}")  # noqa: T201
        print(f"  loss_table shape: {loss_table.shape}")  # noqa: T201
