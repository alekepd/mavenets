"""Unit tests for the tools module.

These tests verify the training utilities and helper functions.
"""

from typing import Callable, Optional, Tuple

import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

# Type aliases for better type checker compatibility
LossFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
EvalerFn = Callable[[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor], torch.Tensor]

from mavenets.tools import (  # type: ignore[import-not-found]
    mixed_MSE,
    _create_parameterized_train_stepper,
    _create_parameterized_evaler,
    _MHTunerWrapper,
    _eval_dataset,
    train_tunable_model,
    SIGNAL_PYGBATCHKEY,
    EXP_PYGBATCHKEY,
)
from mavenets.network.tune import LinearTuner  # type: ignore[import-not-found]
from mavenets.network.base import MLP  # type: ignore[import-not-found]


class TestMixedMSE:
    """Test the mixed_MSE loss function."""

    def test_basic_calculation(self, cpu_device: str) -> None:
        """Test basic mixed MSE calculation."""
        guess1 = torch.tensor([1.0, 2.0, 3.0], device=cpu_device)
        guess2 = torch.tensor([2.0, 3.0, 4.0], device=cpu_device)
        reference = torch.tensor([1.5, 2.5, 3.5], device=cpu_device)
        mix = torch.tensor(0.5, device=cpu_device)

        result = mixed_MSE((guess1, guess2), reference, mix)

        # MSE1 = mean((1-1.5)^2, (2-2.5)^2, (3-3.5)^2) = mean(0.25, 0.25, 0.25) = 0.25
        # MSE2 = mean((2-1.5)^2, (3-2.5)^2, (4-3.5)^2) = mean(0.25, 0.25, 0.25) = 0.25
        # mixed = 0.5 * 0.25 + 0.5 * 0.25 = 0.25
        assert result.shape == ()
        assert torch.isclose(result, torch.tensor(0.25, device=cpu_device))

    def test_mix_zero_uses_first_only(self, cpu_device: str) -> None:
        """Test that mix=0 uses only the first guess."""
        guess1 = torch.tensor([1.0, 2.0], device=cpu_device)
        guess2 = torch.tensor([100.0, 200.0], device=cpu_device)  # Very different
        reference = torch.tensor([1.0, 2.0], device=cpu_device)
        mix = torch.tensor(0.0, device=cpu_device)

        result = mixed_MSE((guess1, guess2), reference, mix)

        # With mix=0, only guess1 matters, and it equals reference
        assert torch.isclose(result, torch.tensor(0.0, device=cpu_device))

    def test_mix_one_uses_second_only(self, cpu_device: str) -> None:
        """Test that mix=1 uses only the second guess."""
        guess1 = torch.tensor([100.0, 200.0], device=cpu_device)  # Very different
        guess2 = torch.tensor([1.0, 2.0], device=cpu_device)
        reference = torch.tensor([1.0, 2.0], device=cpu_device)
        mix = torch.tensor(1.0, device=cpu_device)

        result = mixed_MSE((guess1, guess2), reference, mix)

        # With mix=1, only guess2 matters, and it equals reference
        assert torch.isclose(result, torch.tensor(0.0, device=cpu_device))

    def test_gradient_flows(self, cpu_device: str) -> None:
        """Test that gradients flow through the loss."""
        guess1 = torch.tensor([1.0, 2.0], device=cpu_device, requires_grad=True)
        guess2 = torch.tensor([2.0, 3.0], device=cpu_device, requires_grad=True)
        reference = torch.tensor([1.5, 2.5], device=cpu_device)
        mix = torch.tensor(0.5, device=cpu_device)

        result = mixed_MSE((guess1, guess2), reference, mix)
        result.backward()

        assert guess1.grad is not None
        assert guess2.grad is not None

    def test_batch_input(self, cpu_device: str) -> None:
        """Test with batched input."""
        batch_size = 4
        guess1 = torch.randn(batch_size, device=cpu_device)
        guess2 = torch.randn(batch_size, device=cpu_device)
        reference = torch.randn(batch_size, device=cpu_device)
        mix = torch.tensor(0.3, device=cpu_device)

        result = mixed_MSE((guess1, guess2), reference, mix)

        assert result.shape == ()
        assert not torch.isnan(result)


class TestMHTunerWrapper:
    """Test the _MHTunerWrapper class."""

    @pytest.fixture
    def simple_tuner(self, cpu_device: str) -> LinearTuner[torch.Tensor]:
        """Create a simple tuner for testing."""
        mlp = MLP(in_size=5, out_size=1, hidden_sizes=[4], post_squeeze=True).to(
            cpu_device
        )
        return LinearTuner(mlp, n_heads=2).to(cpu_device)

    def test_init_non_graph(self, simple_tuner: LinearTuner[torch.Tensor]) -> None:
        """Test initialization in non-graph mode."""
        wrapper = _MHTunerWrapper(simple_tuner, graph=False)
        assert wrapper.graph is False
        assert wrapper.tuned_model is simple_tuner

    def test_init_graph(self, simple_tuner: LinearTuner[torch.Tensor]) -> None:
        """Test initialization in graph mode."""
        wrapper = _MHTunerWrapper(simple_tuner, graph=True)
        assert wrapper.graph is True

    def test_forward_non_graph(
        self, simple_tuner: LinearTuner[torch.Tensor], cpu_device: str
    ) -> None:
        """Test forward pass in non-graph mode."""
        wrapper = _MHTunerWrapper(simple_tuner, graph=False)

        X = torch.randn(3, 5, device=cpu_device)
        head_idx = torch.tensor([0, 1, 0], dtype=torch.long, device=cpu_device)
        inp = (X, head_idx)

        tuned, raw = wrapper(inp)

        assert tuned.shape == (3,)
        assert raw.shape == (3,)

    def test_forward_graph_mode(self, cpu_device: str) -> None:
        """Test forward pass in graph mode with mock data."""
        # Create a mock graph-like input that also serves as a callable
        class MockBatch:
            def __init__(self, device: str) -> None:
                self.x = torch.randn(3, 5, device=device)
                self.experiment = torch.tensor(
                    [0, 1, 0], dtype=torch.long, device=device
                )

            def __getitem__(self, key: str) -> torch.Tensor:
                if key == EXP_PYGBATCHKEY:
                    return self.experiment
                raise KeyError(key)

        # Create a mock tuner that can handle the graph input
        class MockTuner(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.called_with_head_index: Optional[torch.Tensor] = None
                self.called_with_return_raw: Optional[bool] = None

            def forward(
                self, inp: MockBatch, head_index: torch.Tensor, return_raw: bool = False
            ) -> Tuple[torch.Tensor, torch.Tensor]:
                self.called_with_head_index = head_index
                self.called_with_return_raw = return_raw
                return torch.randn(3), torch.randn(3)

        mock_tuner = MockTuner()
        wrapper = _MHTunerWrapper(mock_tuner, graph=True)  # type: ignore

        batch = MockBatch(cpu_device)
        tuned, raw = wrapper(batch)

        assert mock_tuner.called_with_head_index is not None
        assert torch.equal(mock_tuner.called_with_head_index, batch.experiment)
        assert mock_tuner.called_with_return_raw is True


class TestCreateParameterizedTrainStepper:
    """Test the _create_parameterized_train_stepper function."""

    @pytest.fixture
    def simple_model(self, cpu_device: str) -> nn.Module:
        """Create a simple model for testing."""
        return nn.Linear(5, 1).to(cpu_device)

    @pytest.fixture
    def simple_loss(self) -> LossFn:
        """Create a simple loss function."""

        def loss_fn(
            pred: torch.Tensor, ref: torch.Tensor, param: torch.Tensor
        ) -> torch.Tensor:
            return nn.functional.mse_loss(pred.squeeze(), ref)

        return loss_fn

    def test_creates_callable(
        self, simple_model: nn.Module, simple_loss: LossFn, cpu_device: str
    ) -> None:
        """Test that function returns a callable."""
        optimizer = torch.optim.SGD(simple_model.parameters(), lr=0.01)
        stepper = _create_parameterized_train_stepper(
            model=simple_model,
            optimizer=optimizer,
            loss_function=simple_loss,
            device=cpu_device,
        )
        assert callable(stepper)

    def test_stepper_updates_model(
        self, simple_model: nn.Module, simple_loss: LossFn, cpu_device: str
    ) -> None:
        """Test that the stepper updates model parameters."""
        optimizer = torch.optim.SGD(simple_model.parameters(), lr=0.1)
        stepper = _create_parameterized_train_stepper(
            model=simple_model,
            optimizer=optimizer,
            loss_function=simple_loss,
            device=cpu_device,
        )

        # Get initial weights
        initial_weights = simple_model.weight.clone()

        # Run a training step
        X = torch.randn(4, 5, device=cpu_device)
        y = torch.randn(4, device=cpu_device)
        loss_param = torch.tensor(0.5, device=cpu_device)

        stepper(X, y, loss_param)

        # Weights should have changed
        assert not torch.allclose(simple_model.weight, initial_weights)

    def test_stepper_sets_train_mode(
        self, simple_model: nn.Module, simple_loss: LossFn, cpu_device: str
    ) -> None:
        """Test that stepper sets model to train mode."""
        optimizer = torch.optim.SGD(simple_model.parameters(), lr=0.01)
        stepper = _create_parameterized_train_stepper(
            model=simple_model,
            optimizer=optimizer,
            loss_function=simple_loss,
            device=cpu_device,
        )

        simple_model.eval()  # Start in eval mode

        X = torch.randn(4, 5, device=cpu_device)
        y = torch.randn(4, device=cpu_device)
        loss_param = torch.tensor(0.5, device=cpu_device)

        stepper(X, y, loss_param)

        assert simple_model.training is True

    def test_grad_clip_applied(
        self, simple_model: nn.Module, simple_loss: LossFn, cpu_device: str
    ) -> None:
        """Test that gradient clipping is applied."""
        optimizer = torch.optim.SGD(simple_model.parameters(), lr=0.01)
        stepper = _create_parameterized_train_stepper(
            model=simple_model,
            optimizer=optimizer,
            loss_function=simple_loss,
            device=cpu_device,
            grad_clip=0.1,  # Very low clip value
        )

        X = torch.randn(4, 5, device=cpu_device) * 100  # Large input
        y = torch.randn(4, device=cpu_device) * 100
        loss_param = torch.tensor(0.5, device=cpu_device)

        stepper(X, y, loss_param)

        # After the step, the model should still have reasonable weights
        # (not NaN due to exploding gradients)
        assert not torch.isnan(simple_model.weight).any()


class TestCreateParameterizedEvaler:
    """Test the _create_parameterized_evaler function."""

    @pytest.fixture
    def simple_model(self, cpu_device: str) -> nn.Module:
        """Create a simple model for testing."""
        return nn.Linear(5, 1).to(cpu_device)

    @pytest.fixture
    def simple_loss(self) -> LossFn:
        """Create a simple loss function."""

        def loss_fn(
            pred: torch.Tensor, ref: torch.Tensor, param: torch.Tensor
        ) -> torch.Tensor:
            return nn.functional.mse_loss(pred.squeeze(), ref)

        return loss_fn

    def test_creates_callable(
        self, simple_model: nn.Module, simple_loss: LossFn, cpu_device: str
    ) -> None:
        """Test that function returns a callable."""
        evaler = _create_parameterized_evaler(
            model=simple_model,
            loss_function=simple_loss,
            device=cpu_device,
            optimizer=None,
        )
        assert callable(evaler)

    def test_evaler_returns_loss(
        self, simple_model: nn.Module, simple_loss: LossFn, cpu_device: str
    ) -> None:
        """Test that evaler returns a loss value."""
        evaler = _create_parameterized_evaler(
            model=simple_model,
            loss_function=simple_loss,
            device=cpu_device,
            optimizer=None,
        )

        X = torch.randn(4, 5, device=cpu_device)
        y = torch.randn(4, device=cpu_device)
        loss_param = torch.tensor(0.5, device=cpu_device)

        loss = evaler(X, y, loss_param)

        assert loss.shape == ()
        assert not torch.isnan(loss)

    def test_evaler_sets_eval_mode(
        self, simple_model: nn.Module, simple_loss: LossFn, cpu_device: str
    ) -> None:
        """Test that evaler sets model to eval mode."""
        evaler = _create_parameterized_evaler(
            model=simple_model,
            loss_function=simple_loss,
            device=cpu_device,
            optimizer=None,
        )

        simple_model.train()  # Start in train mode

        X = torch.randn(4, 5, device=cpu_device)
        y = torch.randn(4, device=cpu_device)
        loss_param = torch.tensor(0.5, device=cpu_device)

        evaler(X, y, loss_param)

        assert simple_model.training is False

    def test_evaler_no_gradient_required(
        self, simple_model: nn.Module, simple_loss: LossFn, cpu_device: str
    ) -> None:
        """Test that evaler doesn't require gradients for model update."""
        evaler = _create_parameterized_evaler(
            model=simple_model,
            loss_function=simple_loss,
            device=cpu_device,
            optimizer=None,
        )

        initial_weights = simple_model.weight.clone()

        X = torch.randn(4, 5, device=cpu_device)
        y = torch.randn(4, device=cpu_device)
        loss_param = torch.tensor(0.5, device=cpu_device)

        # Call multiple times
        for _ in range(10):
            evaler(X, y, loss_param)

        # Weights should not have changed
        assert torch.allclose(simple_model.weight, initial_weights)

    def test_call_opt_eval_requires_optimizer(
        self, simple_model: nn.Module, simple_loss: LossFn, cpu_device: str
    ) -> None:
        """Test that call_opt_eval=True requires an optimizer."""
        with pytest.raises(ValueError, match="call_opt_eval must be False"):
            _create_parameterized_evaler(
                model=simple_model,
                loss_function=simple_loss,
                device=cpu_device,
                optimizer=None,
                call_opt_eval=True,
            )


class TestEvalDataset:
    """Test the _eval_dataset function."""

    @pytest.fixture
    def simple_evaler(self, cpu_device: str) -> EvalerFn:
        """Create a simple evaler for testing."""

        def evaler(
            inp: Tuple[torch.Tensor, torch.Tensor],
            reference: torch.Tensor,
            loss_param: torch.Tensor,
        ) -> torch.Tensor:
            pred = inp[0].mean(dim=-1)  # Simple prediction
            return nn.functional.mse_loss(pred, reference)

        return evaler

    def test_evaluates_dataset(
        self, simple_evaler: EvalerFn, cpu_device: str
    ) -> None:
        """Test that _eval_dataset evaluates the entire dataset."""
        # Create a simple dataset
        X = torch.randn(20, 5, device=cpu_device)
        y = torch.randn(20, device=cpu_device)
        exp_idx = torch.zeros(20, dtype=torch.long, device=cpu_device)

        dataset = TensorDataset(X, y, exp_idx)
        loss_param = torch.tensor(0.5, device=cpu_device)

        result = _eval_dataset(
            evaler=simple_evaler,
            dataset=dataset,
            batch_size=5,
            loss_param=loss_param,
            graph=False,
        )

        assert isinstance(result, float)
        assert not torch.isnan(torch.tensor(result))

    def test_handles_different_batch_sizes(
        self, simple_evaler: EvalerFn, cpu_device: str
    ) -> None:
        """Test that function handles various batch sizes."""
        X = torch.randn(17, 5, device=cpu_device)  # Not evenly divisible
        y = torch.randn(17, device=cpu_device)
        exp_idx = torch.zeros(17, dtype=torch.long, device=cpu_device)

        dataset = TensorDataset(X, y, exp_idx)
        loss_param = torch.tensor(0.5, device=cpu_device)

        # Should work with batch size that doesn't divide evenly
        result = _eval_dataset(
            evaler=simple_evaler,
            dataset=dataset,
            batch_size=5,
            loss_param=loss_param,
            graph=False,
        )

        assert isinstance(result, float)


class TestTrainTunableModel:
    """Test the train_tunable_model function."""

    @pytest.fixture
    def simple_tuner(self, cpu_device: str) -> LinearTuner[torch.Tensor]:
        """Create a simple tuner for testing."""
        mlp = MLP(in_size=5, out_size=1, hidden_sizes=[8], post_squeeze=True).to(
            cpu_device
        )
        return LinearTuner(mlp, n_heads=2).to(cpu_device)

    @pytest.fixture
    def simple_datasets(
        self, cpu_device: str
    ) -> Tuple[TensorDataset, TensorDataset]:
        """Create simple train and validation datasets."""
        torch.manual_seed(42)

        # Training data
        X_train = torch.randn(50, 5, device=cpu_device)
        y_train = X_train[:, 0] + 0.5 * X_train[:, 1]
        exp_train = torch.randint(0, 2, (50,), device=cpu_device)
        train_dataset = TensorDataset(X_train, y_train, exp_train)

        # Validation data
        X_val = torch.randn(20, 5, device=cpu_device)
        y_val = X_val[:, 0] + 0.5 * X_val[:, 1]
        exp_val = torch.randint(0, 2, (20,), device=cpu_device)
        val_dataset = TensorDataset(X_val, y_val, exp_val)

        return train_dataset, val_dataset

    def test_returns_correct_types(
        self,
        simple_tuner: LinearTuner[torch.Tensor],
        simple_datasets: Tuple[TensorDataset, TensorDataset],
        cpu_device: str,
    ) -> None:
        """Test that train_tunable_model returns correct types."""
        train_dataset, val_dataset = simple_datasets
        optimizer = torch.optim.Adam(simple_tuner.parameters(), lr=0.01)

        best_epoch, best_val, table = train_tunable_model(
            model=simple_tuner,
            optimizer=optimizer,
            device=cpu_device,
            n_epochs=10,
            train_dataset=train_dataset,
            valid_dataset=val_dataset,
            train_batch_size=10,
            report_stride=5,
            train_bfloat16=False,
            patience=100,  # High patience to avoid early stopping
            progress_bar=False,
        )

        assert isinstance(best_epoch, (int, type(table.index[0])))
        assert isinstance(best_val, float)
        import pandas as pd

        assert isinstance(table, pd.DataFrame)

    def test_training_reduces_loss(
        self,
        simple_tuner: LinearTuner[torch.Tensor],
        simple_datasets: Tuple[TensorDataset, TensorDataset],
        cpu_device: str,
    ) -> None:
        """Test that training reduces the loss."""
        train_dataset, val_dataset = simple_datasets
        optimizer = torch.optim.Adam(simple_tuner.parameters(), lr=0.01)

        _, _, table = train_tunable_model(
            model=simple_tuner,
            optimizer=optimizer,
            device=cpu_device,
            n_epochs=50,
            train_dataset=train_dataset,
            valid_dataset=val_dataset,
            train_batch_size=10,
            report_stride=10,
            train_bfloat16=False,
            patience=100,
            progress_bar=False,
        )

        # Check that training loss decreased
        train_losses = table["train"].values
        assert train_losses[-1] < train_losses[0], (
            f"Training loss did not decrease: {train_losses[0]:.4f} -> {train_losses[-1]:.4f}"
        )

    def test_early_stopping(
        self,
        simple_tuner: LinearTuner[torch.Tensor],
        cpu_device: str,
    ) -> None:
        """Test that early stopping works."""
        # Create datasets where validation loss won't improve
        # by using different distributions for train and validation
        torch.manual_seed(42)

        # Training data - one pattern
        X_train = torch.randn(50, 5, device=cpu_device)
        y_train = X_train[:, 0] + 0.5 * X_train[:, 1]
        exp_train = torch.randint(0, 2, (50,), device=cpu_device)
        train_dataset = TensorDataset(X_train, y_train, exp_train)

        # Validation data - completely random targets (unlearnable relationship)
        X_val = torch.randn(20, 5, device=cpu_device)
        y_val = torch.randn(20, device=cpu_device) * 10  # Random targets
        exp_val = torch.randint(0, 2, (20,), device=cpu_device)
        val_dataset = TensorDataset(X_val, y_val, exp_val)

        optimizer = torch.optim.Adam(simple_tuner.parameters(), lr=0.1)  # High LR

        _, _, table = train_tunable_model(
            model=simple_tuner,
            optimizer=optimizer,
            device=cpu_device,
            n_epochs=500,  # High epoch count
            train_dataset=train_dataset,
            valid_dataset=val_dataset,
            train_batch_size=10,
            report_stride=1,
            train_bfloat16=False,
            patience=10,  # Low patience
            progress_bar=False,
        )

        # Should stop before reaching max epochs due to validation not improving
        assert len(table) < 500, f"Expected early stopping but ran {len(table)} epochs"

    def test_report_datasets_included(
        self,
        simple_tuner: LinearTuner[torch.Tensor],
        simple_datasets: Tuple[TensorDataset, TensorDataset],
        cpu_device: str,
    ) -> None:
        """Test that report_datasets are included in the output table."""
        train_dataset, val_dataset = simple_datasets

        # Create an extra report dataset
        X_extra = torch.randn(10, 5, device=cpu_device)
        y_extra = torch.randn(10, device=cpu_device)
        exp_extra = torch.zeros(10, dtype=torch.long, device=cpu_device)
        extra_dataset = TensorDataset(X_extra, y_extra, exp_extra)

        optimizer = torch.optim.Adam(simple_tuner.parameters(), lr=0.01)

        _, _, table = train_tunable_model(
            model=simple_tuner,
            optimizer=optimizer,
            device=cpu_device,
            n_epochs=10,
            train_dataset=train_dataset,
            valid_dataset=val_dataset,
            report_datasets={"extra": extra_dataset},
            train_batch_size=10,
            report_stride=5,
            train_bfloat16=False,
            patience=100,
            progress_bar=False,
        )

        assert "extra" in table.columns

    def test_loss_param_schedule(
        self,
        simple_tuner: LinearTuner[torch.Tensor],
        simple_datasets: Tuple[TensorDataset, TensorDataset],
        cpu_device: str,
    ) -> None:
        """Test that loss parameter schedule works."""
        train_dataset, val_dataset = simple_datasets
        optimizer = torch.optim.Adam(simple_tuner.parameters(), lr=0.01)

        _, _, table = train_tunable_model(
            model=simple_tuner,
            optimizer=optimizer,
            device=cpu_device,
            n_epochs=30,
            train_dataset=train_dataset,
            valid_dataset=val_dataset,
            train_batch_size=10,
            report_stride=5,
            train_bfloat16=False,
            patience=100,
            progress_bar=False,
            start_loss_param=1.0,
            end_loss_param=0.0,
            loss_param_ramp_size=20,
        )

        # loss_param should decrease over time
        loss_params = table["loss_param"].values
        assert loss_params[0] >= loss_params[-1]


class TestConstants:
    """Test module constants."""

    def test_signal_pygbatchkey(self) -> None:
        """Test SIGNAL_PYGBATCHKEY constant."""
        assert SIGNAL_PYGBATCHKEY == "y"

    def test_exp_pygbatchkey(self) -> None:
        """Test EXP_PYGBATCHKEY constant."""
        assert EXP_PYGBATCHKEY == "experiment"
