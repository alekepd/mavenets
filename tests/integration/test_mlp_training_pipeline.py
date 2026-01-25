"""Integration tests for the MLP training pipeline.

This test mirrors the structure of mavenets/example/run_mlp.py but uses
synthetic data and smaller models to run quickly on CPU.
"""

import pytest
import torch
from torch.utils.data import TensorDataset

from tests.conftest import create_synthetic_datasets
from mavenets.network.base import MLP  # type: ignore[import-not-found]
from mavenets.network.tune import SharedFanTuner, LinearTuner  # type: ignore[import-not-found]
from mavenets.tools import train_tunable_model  # type: ignore[import-not-found]


@pytest.mark.integration
class TestMLPTrainingPipeline:
    """Integration tests for the MLP training pipeline.

    These tests mirror the structure of run_mlp.py but use synthetic data
    and smaller models to run quickly on CPU.
    """

    def test_mlp_with_sharedfantuner_training(self, cpu_device: str) -> None:
        """Test training an MLP wrapped in SharedFanTuner.

        This mirrors the run_mlp.py workflow:
        1. Create datasets
        2. Build MLP with SharedFanTuner
        3. Train with train_tunable_model
        4. Verify training reduces loss
        """
        # Create synthetic datasets
        n_features = 50  # Smaller than 21*201 used in run_mlp.py
        n_heads = 3  # Smaller than 8 used in run_mlp.py
        train_dataset, valid_dataset = create_synthetic_datasets(
            n_train=100,
            n_val=30,
            n_features=n_features,
            n_heads=n_heads,
            device=cpu_device,
        )

        # Build model (smaller than run_mlp.py)
        underlying_model = MLP(
            in_size=n_features,
            out_size=1,
            hidden_sizes=[16, 8],  # Smaller than run_mlp.py configs
            pre_flatten=False,  # Already flat
            post_squeeze=True,
        )
        model = SharedFanTuner(
            underlying_model,
            n_heads=n_heads,
            fan_size=8,  # Smaller than 16 in run_mlp.py
        ).to(cpu_device)

        # Create optimizer (no fused option on CPU)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=0.01,  # Higher LR for faster convergence in test
            weight_decay=0.005,
        )

        # Train model
        best_epoch, best_val, table = train_tunable_model(
            model=model,
            optimizer=optimizer,
            device=cpu_device,
            n_epochs=50,  # Much less than 1000 in run_mlp.py
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            train_batch_size=32,
            reporting_batch_size=64,
            compile=False,  # No compilation for CPU test
            grad_clip=300,
            report_stride=10,
            progress_bar=False,
            train_bfloat16=False,  # CPU may not support bfloat16
            patience=20,
        )

        # Verify results
        assert isinstance(best_epoch, (int, type(table.index[0])))
        assert isinstance(best_val, float)
        assert not table.empty

        # Check that training reduced loss
        train_losses = table["train"].values
        assert train_losses[-1] < train_losses[0], (
            f"Training loss did not decrease: {train_losses[0]:.4f} -> {train_losses[-1]:.4f}"
        )

    def test_mlp_with_lineartuner_training(self, cpu_device: str) -> None:
        """Test training an MLP wrapped in LinearTuner."""
        n_features = 50
        n_heads = 2
        train_dataset, valid_dataset = create_synthetic_datasets(
            n_train=100,
            n_val=30,
            n_features=n_features,
            n_heads=n_heads,
            device=cpu_device,
        )

        underlying_model = MLP(
            in_size=n_features,
            out_size=1,
            hidden_sizes=[16],
            post_squeeze=True,
        )
        model = LinearTuner(
            underlying_model,
            n_heads=n_heads,
            residual_connection=True,
        ).to(cpu_device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        best_epoch, best_val, table = train_tunable_model(
            model=model,
            optimizer=optimizer,
            device=cpu_device,
            n_epochs=50,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            train_batch_size=32,
            reporting_batch_size=64,
            compile=False,
            report_stride=10,
            progress_bar=False,
            train_bfloat16=False,
            patience=20,
        )

        train_losses = table["train"].values
        assert train_losses[-1] < train_losses[0]

    def test_training_with_report_datasets(self, cpu_device: str) -> None:
        """Test training with additional report datasets (mimics run_mlp.py behavior)."""
        n_features = 50
        n_heads = 3
        train_dataset, valid_dataset = create_synthetic_datasets(
            n_train=100,
            n_val=30,
            n_features=n_features,
            n_heads=n_heads,
            device=cpu_device,
        )

        # Create additional report datasets (mimics per-experiment evaluation)
        report_datasets = {}
        for i in range(n_heads):
            # Create small dataset for each "experiment"
            torch.manual_seed(100 + i)
            X = torch.randn(20, n_features, device=cpu_device)
            y = torch.randn(20, device=cpu_device)
            exp = torch.full((20,), i, dtype=torch.long, device=cpu_device)
            report_datasets[f"exp_{i}"] = TensorDataset(X, y, exp)

        underlying_model = MLP(
            in_size=n_features,
            out_size=1,
            hidden_sizes=[16, 8],
            post_squeeze=True,
        )
        model = SharedFanTuner(underlying_model, n_heads=n_heads, fan_size=8).to(cpu_device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

        best_epoch, best_val, table = train_tunable_model(
            model=model,
            optimizer=optimizer,
            device=cpu_device,
            n_epochs=30,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            report_datasets=report_datasets,
            train_batch_size=32,
            reporting_batch_size=64,
            compile=False,
            report_stride=10,
            progress_bar=False,
            train_bfloat16=False,
            patience=20,
        )

        # Verify report dataset columns are in the table
        for name in report_datasets:
            assert name in table.columns, f"Missing report dataset column: {name}"

    def test_different_hidden_layer_configurations(self, cpu_device: str) -> None:
        """Test different hidden layer configurations (mimics run_mlp.py scan)."""
        n_features = 50
        n_heads = 2

        # Test a few different configurations (subset of run_mlp.py scan)
        layer_configs = [
            [16],
            [32, 16],
            [16, 16, 8],
        ]

        train_dataset, valid_dataset = create_synthetic_datasets(
            n_train=80,
            n_val=20,
            n_features=n_features,
            n_heads=n_heads,
            device=cpu_device,
        )

        results = []
        for hidden_sizes in layer_configs:
            underlying_model = MLP(
                in_size=n_features,
                out_size=1,
                hidden_sizes=hidden_sizes,
                post_squeeze=True,
            )
            model = SharedFanTuner(underlying_model, n_heads=n_heads, fan_size=8).to(cpu_device)

            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

            best_epoch, best_val, table = train_tunable_model(
                model=model,
                optimizer=optimizer,
                device=cpu_device,
                n_epochs=30,
                train_dataset=train_dataset,
                valid_dataset=valid_dataset,
                train_batch_size=32,
                reporting_batch_size=64,
                compile=False,
                report_stride=10,
                progress_bar=False,
                train_bfloat16=False,
                patience=15,
            )

            results.append((hidden_sizes, best_val))

        # Verify all configurations trained successfully
        assert len(results) == len(layer_configs)
        for hidden_sizes, best_val in results:
            assert isinstance(best_val, float)
            assert not torch.isnan(torch.tensor(best_val))


@pytest.mark.integration
class TestMLPPipelineEdgeCases:
    """Test edge cases in the MLP training pipeline."""

    def test_single_head_tuner(self, cpu_device: str) -> None:
        """Test training with a single tuning head."""
        n_features = 30
        n_heads = 1

        train_dataset, valid_dataset = create_synthetic_datasets(
            n_train=50,
            n_val=15,
            n_features=n_features,
            n_heads=n_heads,
            device=cpu_device,
        )

        model = SharedFanTuner(
            MLP(in_size=n_features, out_size=1, hidden_sizes=[8], post_squeeze=True),
            n_heads=n_heads,
            fan_size=4,
        ).to(cpu_device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        best_epoch, best_val, table = train_tunable_model(
            model=model,
            optimizer=optimizer,
            device=cpu_device,
            n_epochs=20,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            train_batch_size=16,
            reporting_batch_size=32,
            compile=False,
            report_stride=5,
            progress_bar=False,
            train_bfloat16=False,
            patience=10,
        )

        assert not table.empty

    def test_small_batch_size(self, cpu_device: str) -> None:
        """Test training with small batch sizes."""
        n_features = 30
        n_heads = 2

        train_dataset, valid_dataset = create_synthetic_datasets(
            n_train=50,
            n_val=15,
            n_features=n_features,
            n_heads=n_heads,
            device=cpu_device,
        )

        model = SharedFanTuner(
            MLP(in_size=n_features, out_size=1, hidden_sizes=[8], post_squeeze=True),
            n_heads=n_heads,
            fan_size=4,
        ).to(cpu_device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Very small batch size
        best_epoch, best_val, table = train_tunable_model(
            model=model,
            optimizer=optimizer,
            device=cpu_device,
            n_epochs=20,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            train_batch_size=4,  # Very small
            reporting_batch_size=8,
            compile=False,
            report_stride=5,
            progress_bar=False,
            train_bfloat16=False,
            patience=10,
        )

        assert not table.empty

    def test_loss_param_annealing(self, cpu_device: str) -> None:
        """Test training with loss parameter annealing schedule."""
        n_features = 30
        n_heads = 2

        train_dataset, valid_dataset = create_synthetic_datasets(
            n_train=50,
            n_val=15,
            n_features=n_features,
            n_heads=n_heads,
            device=cpu_device,
        )

        model = SharedFanTuner(
            MLP(in_size=n_features, out_size=1, hidden_sizes=[8], post_squeeze=True),
            n_heads=n_heads,
            fan_size=4,
        ).to(cpu_device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        best_epoch, best_val, table = train_tunable_model(
            model=model,
            optimizer=optimizer,
            device=cpu_device,
            n_epochs=30,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            train_batch_size=16,
            reporting_batch_size=32,
            compile=False,
            report_stride=5,
            progress_bar=False,
            train_bfloat16=False,
            patience=15,
            # Loss param annealing settings
            start_loss_param=1.0,
            end_loss_param=0.0,
            loss_param_ramp_size=15,
        )

        # Verify loss_param column shows annealing
        loss_params = table["loss_param"].values
        assert loss_params[0] > loss_params[-1], "Loss param should decrease over time"
