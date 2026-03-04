"""Tests for the linear calibration feature of the predict function."""

import torch
import numpy as np
from torch.utils.data import TensorDataset
import pytest
from sklearn.linear_model import LinearRegression  # type: ignore[import-untyped]

from mavenets.report import (  # type: ignore[import-not-found]
    predict,
    REFERENCE_KEY,
    TUNED_PRED_KEY,
    RAW_PRED_KEY,
    EXPID_KEY,
    TUNED_CALIBRATED_KEY,
    RAW_CALIBRATED_KEY,
    SEQUENCE_KEY,
    MUTCOUNT_KEY,
)
from mavenets.network.tune import NullTuner  # type: ignore[import-not-found]
from mavenets.network.base import MLP  # type: ignore[import-not-found]
from mavenets.data.load import SequenceDataset  # type: ignore[import-not-found]
from mavenets.data import SARS_COV2_SEQ  # type: ignore[import-not-found]


@pytest.fixture
def tuner(cpu_device: str) -> NullTuner[torch.Tensor]:
    """Create a NullTuner wrapping a small MLP."""
    mlp = MLP(in_size=5, out_size=1, hidden_sizes=[8], post_squeeze=True).to(
        cpu_device
    )
    return NullTuner(mlp).to(cpu_device)


@pytest.fixture
def multi_experiment_dataset(cpu_device: str) -> TensorDataset:
    """Create a dataset with two experiments and enough samples for regression."""
    torch.manual_seed(42)
    n = 40
    X = torch.randn(n, 5, device=cpu_device)
    y = torch.randn(n, device=cpu_device)
    # 20 samples per experiment
    exp_idx = torch.tensor([0] * 20 + [1] * 20, dtype=torch.long, device=cpu_device)
    return TensorDataset(X, y, exp_idx)


@pytest.fixture
def sequence_dataset(cpu_device: str) -> SequenceDataset[tuple[torch.Tensor, ...]]:
    """Create a SequenceDataset with two experiments."""
    torch.manual_seed(42)
    n = 10
    X = torch.randn(n, 5, device=cpu_device)
    y = torch.randn(n, device=cpu_device)
    exp_idx = torch.tensor([0] * 5 + [1] * 5, dtype=torch.long, device=cpu_device)
    base_dataset = TensorDataset(X, y, exp_idx)
    seq_len = len(SARS_COV2_SEQ)
    sequences = tuple(
        SARS_COV2_SEQ[:i] + "X" + SARS_COV2_SEQ[i + 1 :] if i < seq_len else SARS_COV2_SEQ
        for i in range(n)
    )
    return SequenceDataset(base_dataset, sequences)


class TestLinearCalibrationFlag:
    """Test that the linear_calibration flag controls column presence."""

    def test_columns_absent_when_flag_false(
        self, tuner: NullTuner[torch.Tensor], multi_experiment_dataset: TensorDataset
    ) -> None:
        """Calibrated columns should not appear when flag is False."""
        result = predict(
            model=tuner,
            dataset=multi_experiment_dataset,
            graph=False,
            translate_experiment_ids=False,
            linear_calibration=False,
        )
        assert TUNED_CALIBRATED_KEY not in result.columns
        assert RAW_CALIBRATED_KEY not in result.columns

    def test_columns_present_when_flag_true(
        self, tuner: NullTuner[torch.Tensor], multi_experiment_dataset: TensorDataset
    ) -> None:
        """Calibrated columns should appear when flag is True."""
        result = predict(
            model=tuner,
            dataset=multi_experiment_dataset,
            graph=False,
            translate_experiment_ids=False,
            linear_calibration=True,
        )
        assert TUNED_CALIBRATED_KEY in result.columns
        assert RAW_CALIBRATED_KEY in result.columns

    def test_default_flag_is_false(
        self, tuner: NullTuner[torch.Tensor], multi_experiment_dataset: TensorDataset
    ) -> None:
        """Default behavior should not include calibrated columns."""
        result = predict(
            model=tuner,
            dataset=multi_experiment_dataset,
            graph=False,
            translate_experiment_ids=False,
        )
        assert TUNED_CALIBRATED_KEY not in result.columns
        assert RAW_CALIBRATED_KEY not in result.columns


class TestLinearCalibrationValues:
    """Test the correctness of calibrated values."""

    def test_calibrated_values_no_nans(
        self, tuner: NullTuner[torch.Tensor], multi_experiment_dataset: TensorDataset
    ) -> None:
        """Calibrated columns should not contain NaN values."""
        result = predict(
            model=tuner,
            dataset=multi_experiment_dataset,
            graph=False,
            translate_experiment_ids=False,
            linear_calibration=True,
        )
        assert not bool(result[TUNED_CALIBRATED_KEY].isna().any())
        assert not bool(result[RAW_CALIBRATED_KEY].isna().any())

    def test_correct_number_of_rows(
        self, tuner: NullTuner[torch.Tensor], multi_experiment_dataset: TensorDataset
    ) -> None:
        """Calibrated columns should have the same number of rows as the dataset."""
        result = predict(
            model=tuner,
            dataset=multi_experiment_dataset,
            graph=False,
            translate_experiment_ids=False,
            linear_calibration=True,
        )
        assert len(result[TUNED_CALIBRATED_KEY]) == len(multi_experiment_dataset)
        assert len(result[RAW_CALIBRATED_KEY]) == len(multi_experiment_dataset)

    def test_calibration_matches_manual_sklearn(
        self, tuner: NullTuner[torch.Tensor], multi_experiment_dataset: TensorDataset
    ) -> None:
        """Calibrated values should match manually fitting sklearn LinearRegression."""
        result = predict(
            model=tuner,
            dataset=multi_experiment_dataset,
            graph=False,
            translate_experiment_ids=False,
            linear_calibration=True,
        )

        # Manually replicate per-experiment linear calibration
        for exp_id in result[EXPID_KEY].unique():
            mask = result[EXPID_KEY] == exp_id
            ref = result.loc[mask, REFERENCE_KEY].values

            for pred_key, cal_key in [
                (TUNED_PRED_KEY, TUNED_CALIBRATED_KEY),
                (RAW_PRED_KEY, RAW_CALIBRATED_KEY),
            ]:
                pred = result.loc[mask, pred_key].values
                reg = LinearRegression()
                reg.fit(pred.reshape(-1, 1), ref)
                expected = reg.predict(pred.reshape(-1, 1))
                np.testing.assert_allclose(
                    result.loc[mask, cal_key].values,
                    expected,
                    rtol=1e-5,
                )

    def test_per_experiment_independence(
        self, tuner: NullTuner[torch.Tensor], multi_experiment_dataset: TensorDataset
    ) -> None:
        """Each experiment should get its own independent linear model.

        Verify that the calibration for one experiment is not affected by data
        from another experiment.
        """
        result_full = predict(
            model=tuner,
            dataset=multi_experiment_dataset,
            graph=False,
            translate_experiment_ids=False,
            linear_calibration=True,
        )

        # For each experiment, verify the fit is done only on that experiment's data
        for exp_id in result_full[EXPID_KEY].unique():
            mask = result_full[EXPID_KEY] == exp_id
            ref = result_full.loc[mask, REFERENCE_KEY].values
            tuned = result_full.loc[mask, TUNED_PRED_KEY].values
            cal = result_full.loc[mask, TUNED_CALIBRATED_KEY].values

            # Fit on just this experiment
            reg = LinearRegression()
            reg.fit(tuned.reshape(-1, 1), ref)
            expected = reg.predict(tuned.reshape(-1, 1))
            np.testing.assert_allclose(cal, expected, rtol=1e-5)


class TestLinearCalibrationWithTranslatedIds:
    """Test calibration works with translated experiment IDs."""

    def test_calibration_with_translated_ids(
        self, tuner: NullTuner[torch.Tensor], multi_experiment_dataset: TensorDataset
    ) -> None:
        """Calibration should work when experiment IDs are translated to names."""
        result = predict(
            model=tuner,
            dataset=multi_experiment_dataset,
            graph=False,
            translate_experiment_ids=True,
            linear_calibration=True,
        )
        assert TUNED_CALIBRATED_KEY in result.columns
        assert RAW_CALIBRATED_KEY in result.columns
        assert not bool(result[TUNED_CALIBRATED_KEY].isna().any())
        assert not bool(result[RAW_CALIBRATED_KEY].isna().any())


class TestLinearCalibrationWithSequenceDataset:
    """Test calibration works alongside SequenceDataset columns."""

    def test_all_columns_present(
        self,
        tuner: NullTuner[torch.Tensor],
        sequence_dataset: SequenceDataset[tuple[torch.Tensor, ...]],
    ) -> None:
        """All columns should be present when both features are enabled."""
        result = predict(
            model=tuner,
            dataset=sequence_dataset,
            graph=False,
            translate_experiment_ids=False,
            linear_calibration=True,
        )
        expected_columns = {
            REFERENCE_KEY,
            TUNED_PRED_KEY,
            RAW_PRED_KEY,
            EXPID_KEY,
            TUNED_CALIBRATED_KEY,
            RAW_CALIBRATED_KEY,
            SEQUENCE_KEY,
            MUTCOUNT_KEY,
        }
        assert set(result.columns) == expected_columns

    def test_calibrated_values_valid(
        self,
        tuner: NullTuner[torch.Tensor],
        sequence_dataset: SequenceDataset[tuple[torch.Tensor, ...]],
    ) -> None:
        """Calibrated values should be valid floats with SequenceDataset."""
        result = predict(
            model=tuner,
            dataset=sequence_dataset,
            graph=False,
            translate_experiment_ids=False,
            linear_calibration=True,
        )
        assert not bool(result[TUNED_CALIBRATED_KEY].isna().any())
        assert not bool(result[RAW_CALIBRATED_KEY].isna().any())
