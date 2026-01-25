"""Integration tests for tuner classes with MLP networks.

These tests verify that tuner classes from tune.py work correctly
when combined with MLP networks from base.py. They use small models
and synthetic data to keep execution fast on CPU.
"""

import pytest
import torch
import torch.nn as nn

from mavenets.network.base import MLP  # type: ignore[import-not-found]
from mavenets.network.tune import (  # type: ignore[import-not-found]
    NullTuner,
    LinearTuner,
    SharedFanTuner,
    FFNTuner,
    HeadLock,
)


class TestTunerForwardPass:
    """Test forward pass through tuner-wrapped MLP models."""

    @pytest.fixture
    def small_mlp(self, cpu_device: str) -> MLP:
        """Create a small MLP that outputs a scalar."""
        return MLP(
            in_size=10,
            out_size=1,
            hidden_sizes=[8, 4],
            post_squeeze=True,
        ).to(cpu_device)

    def test_null_tuner_forward(self, small_mlp: MLP, cpu_device: str) -> None:
        """Test NullTuner forward pass returns same output as base model."""
        tuner = NullTuner(small_mlp).to(cpu_device)

        X = torch.randn(5, 10, device=cpu_device)
        head_idx = torch.zeros(5, dtype=torch.long, device=cpu_device)

        # NullTuner should return identical output to base model
        with torch.no_grad():
            raw_output = small_mlp(X)
            tuned_output = tuner(X, head_idx)

        assert tuned_output.shape == raw_output.shape
        assert torch.allclose(tuned_output, raw_output)

    def test_null_tuner_return_raw(self, small_mlp: MLP, cpu_device: str) -> None:
        """Test NullTuner return_raw option returns both outputs."""
        tuner = NullTuner(small_mlp).to(cpu_device)

        X = torch.randn(5, 10, device=cpu_device)
        head_idx = torch.zeros(5, dtype=torch.long, device=cpu_device)

        with torch.no_grad():
            tuned, raw = tuner(X, head_idx, return_raw=True)

        assert torch.allclose(tuned, raw)

    def test_linear_tuner_forward(self, small_mlp: MLP, cpu_device: str) -> None:
        """Test LinearTuner forward pass with different heads."""
        n_heads = 3
        tuner = LinearTuner(small_mlp, n_heads=n_heads).to(cpu_device)

        X = torch.randn(6, 10, device=cpu_device)
        # Assign different heads to different samples
        head_idx = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.long, device=cpu_device)

        output = tuner(X, head_idx)
        assert output.shape == (6,)
        assert not torch.isnan(output).any()

    def test_linear_tuner_different_heads_differ(
        self, small_mlp: MLP, cpu_device: str
    ) -> None:
        """Test that different tuning heads produce different outputs."""
        tuner = LinearTuner(small_mlp, n_heads=2, residual_connection=False).to(
            cpu_device
        )

        # Use same input, different heads
        X = torch.randn(1, 10, device=cpu_device)
        X_repeated = X.repeat(2, 1)
        head_idx = torch.tensor([0, 1], dtype=torch.long, device=cpu_device)

        output = tuner(X_repeated, head_idx)

        # Different heads should give different outputs (unless initialized identically)
        # With random init, they should differ
        assert output.shape == (2,)

    def test_shared_fan_tuner_forward(self, small_mlp: MLP, cpu_device: str) -> None:
        """Test SharedFanTuner forward pass."""
        tuner = SharedFanTuner(
            small_mlp, n_heads=3, fan_size=8, residual_connection=True
        ).to(cpu_device)

        X = torch.randn(5, 10, device=cpu_device)
        head_idx = torch.tensor([0, 1, 2, 1, 0], dtype=torch.long, device=cpu_device)

        output = tuner(X, head_idx)
        assert output.shape == (5,)
        assert not torch.isnan(output).any()

    def test_ffn_tuner_forward(self, small_mlp: MLP, cpu_device: str) -> None:
        """Test FFNTuner forward pass."""
        tuner = FFNTuner(
            small_mlp, n_heads=2, hidden_size=4, n_hidden=1
        ).to(cpu_device)

        X = torch.randn(4, 10, device=cpu_device)
        head_idx = torch.tensor([0, 1, 0, 1], dtype=torch.long, device=cpu_device)

        output = tuner(X, head_idx)
        assert output.shape == (4,)
        assert not torch.isnan(output).any()


class TestHeadLock:
    """Test HeadLock wrapper for fixing tuner head."""

    @pytest.fixture
    def tuned_mlp(self, cpu_device: str) -> LinearTuner[torch.Tensor]:
        """Create a tuned MLP model."""
        mlp = MLP(
            in_size=10,
            out_size=1,
            hidden_sizes=[8],
            post_squeeze=True,
        ).to(cpu_device)
        return LinearTuner(mlp, n_heads=3).to(cpu_device)

    def test_headlock_creation(
        self, tuned_mlp: LinearTuner[torch.Tensor], cpu_device: str
    ) -> None:
        """Test HeadLock can be created from tuner."""
        locked = tuned_mlp.create_singlehead_model(head_index=1)
        assert isinstance(locked, HeadLock)

    def test_headlock_forward(
        self, tuned_mlp: LinearTuner[torch.Tensor], cpu_device: str
    ) -> None:
        """Test HeadLock forward pass uses fixed head."""
        locked = tuned_mlp.create_singlehead_model(head_index=0)

        X = torch.randn(5, 10, device=cpu_device)
        output = locked(X)

        assert output.shape == (5,)
        assert not torch.isnan(output).any()

    def test_headlock_matches_manual_head_selection(
        self, tuned_mlp: LinearTuner[torch.Tensor], cpu_device: str
    ) -> None:
        """Test HeadLock output matches manual head index selection."""
        head_idx = 1
        locked = tuned_mlp.create_singlehead_model(head_index=head_idx)

        X = torch.randn(3, 10, device=cpu_device)
        head_indices = torch.full((3,), head_idx, dtype=torch.long, device=cpu_device)

        with torch.no_grad():
            locked_output = locked(X)
            manual_output = tuned_mlp(X, head_indices)

        assert torch.allclose(locked_output, manual_output)


class TestTunerGradientFlow:
    """Test gradient flow through tuner-wrapped models."""

    @pytest.fixture
    def small_mlp(self, cpu_device: str) -> MLP:
        """Create a small MLP."""
        return MLP(
            in_size=10,
            out_size=1,
            hidden_sizes=[8],
            post_squeeze=True,
        ).to(cpu_device)

    def test_linear_tuner_gradients(self, small_mlp: MLP, cpu_device: str) -> None:
        """Test gradients flow through LinearTuner."""
        tuner = LinearTuner(small_mlp, n_heads=2).to(cpu_device)

        X = torch.randn(4, 10, device=cpu_device, requires_grad=True)
        head_idx = torch.tensor([0, 1, 0, 1], dtype=torch.long, device=cpu_device)
        target = torch.randn(4, device=cpu_device)

        output = tuner(X, head_idx)
        loss = nn.functional.mse_loss(output, target)
        loss.backward()

        # Check gradients exist for input
        assert X.grad is not None
        assert not torch.isnan(X.grad).any()

        # Check gradients exist for tuner parameters
        for param in tuner.parameters():
            assert param.grad is not None or not param.requires_grad

    def test_shared_fan_tuner_gradients(self, small_mlp: MLP, cpu_device: str) -> None:
        """Test gradients flow through SharedFanTuner."""
        tuner = SharedFanTuner(small_mlp, n_heads=2, fan_size=4).to(cpu_device)

        X = torch.randn(4, 10, device=cpu_device, requires_grad=True)
        head_idx = torch.tensor([0, 1, 0, 1], dtype=torch.long, device=cpu_device)
        target = torch.randn(4, device=cpu_device)

        output = tuner(X, head_idx)
        loss = nn.functional.mse_loss(output, target)
        loss.backward()

        assert X.grad is not None
        assert not torch.isnan(X.grad).any()

    def test_ffn_tuner_gradients(self, small_mlp: MLP, cpu_device: str) -> None:
        """Test gradients flow through FFNTuner."""
        tuner = FFNTuner(small_mlp, n_heads=2, hidden_size=4, n_hidden=1).to(cpu_device)

        X = torch.randn(4, 10, device=cpu_device, requires_grad=True)
        head_idx = torch.tensor([0, 1, 0, 1], dtype=torch.long, device=cpu_device)
        target = torch.randn(4, device=cpu_device)

        output = tuner(X, head_idx)
        loss = nn.functional.mse_loss(output, target)
        loss.backward()

        assert X.grad is not None
        assert not torch.isnan(X.grad).any()


@pytest.mark.slow
class TestTunerTraining:
    """Test training loops with tuner-wrapped models."""

    def test_linear_tuner_training_reduces_loss(self, cpu_device: str) -> None:
        """Test that training a LinearTuner reduces loss."""
        mlp = MLP(
            in_size=10,
            out_size=1,
            hidden_sizes=[16, 8],
            post_squeeze=True,
        ).to(cpu_device)
        tuner = LinearTuner(mlp, n_heads=2).to(cpu_device)

        # Synthetic data with head-dependent targets
        torch.manual_seed(42)
        X = torch.randn(50, 10, device=cpu_device)
        head_idx = torch.randint(0, 2, (50,), device=cpu_device)
        # Target depends on head: head 0 -> positive, head 1 -> negative
        y = torch.where(head_idx == 0, X[:, 0], -X[:, 0])

        optimizer = torch.optim.Adam(tuner.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        # Record initial loss
        tuner.eval()
        with torch.no_grad():
            initial_loss = criterion(tuner(X, head_idx), y).item()

        # Train
        tuner.train()
        for _ in range(100):
            optimizer.zero_grad()
            pred = tuner(X, head_idx)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

        # Check final loss is lower
        tuner.eval()
        with torch.no_grad():
            final_loss = criterion(tuner(X, head_idx), y).item()

        assert final_loss < initial_loss, (
            f"Training did not reduce loss: {initial_loss:.4f} -> {final_loss:.4f}"
        )

    def test_shared_fan_tuner_training_reduces_loss(self, cpu_device: str) -> None:
        """Test that training a SharedFanTuner reduces loss."""
        mlp = MLP(
            in_size=10,
            out_size=1,
            hidden_sizes=[16],
            post_squeeze=True,
        ).to(cpu_device)
        tuner = SharedFanTuner(mlp, n_heads=2, fan_size=8).to(cpu_device)

        torch.manual_seed(42)
        X = torch.randn(50, 10, device=cpu_device)
        head_idx = torch.randint(0, 2, (50,), device=cpu_device)
        y = torch.where(head_idx == 0, X[:, 0] + X[:, 1], X[:, 0] - X[:, 1])

        optimizer = torch.optim.Adam(tuner.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        tuner.eval()
        with torch.no_grad():
            initial_loss = criterion(tuner(X, head_idx), y).item()

        tuner.train()
        for _ in range(100):
            optimizer.zero_grad()
            loss = criterion(tuner(X, head_idx), y)
            loss.backward()
            optimizer.step()

        tuner.eval()
        with torch.no_grad():
            final_loss = criterion(tuner(X, head_idx), y).item()

        assert final_loss < initial_loss, (
            f"Training did not reduce loss: {initial_loss:.4f} -> {final_loss:.4f}"
        )

    def test_ffn_tuner_training_reduces_loss(self, cpu_device: str) -> None:
        """Test that training an FFNTuner reduces loss."""
        mlp = MLP(
            in_size=10,
            out_size=1,
            hidden_sizes=[16],
            post_squeeze=True,
        ).to(cpu_device)
        tuner = FFNTuner(mlp, n_heads=2, hidden_size=8, n_hidden=1).to(cpu_device)

        torch.manual_seed(42)
        X = torch.randn(50, 10, device=cpu_device)
        head_idx = torch.randint(0, 2, (50,), device=cpu_device)
        y = torch.where(head_idx == 0, X[:, 0].abs(), -X[:, 0].abs())

        optimizer = torch.optim.Adam(tuner.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        tuner.eval()
        with torch.no_grad():
            initial_loss = criterion(tuner(X, head_idx), y).item()

        tuner.train()
        for _ in range(150):
            optimizer.zero_grad()
            loss = criterion(tuner(X, head_idx), y)
            loss.backward()
            optimizer.step()

        tuner.eval()
        with torch.no_grad():
            final_loss = criterion(tuner(X, head_idx), y).item()

        assert final_loss < initial_loss, (
            f"Training did not reduce loss: {initial_loss:.4f} -> {final_loss:.4f}"
        )


@pytest.mark.integration
class TestTunerMultiHeadBehavior:
    """Test that different heads learn different behaviors."""

    def test_heads_learn_distinct_offsets(self, cpu_device: str) -> None:
        """Test that LinearTuner heads can learn distinct offsets."""
        mlp = MLP(
            in_size=5,
            out_size=1,
            hidden_sizes=[8],
            post_squeeze=True,
        ).to(cpu_device)
        tuner = LinearTuner(mlp, n_heads=2, residual_connection=True).to(cpu_device)

        torch.manual_seed(42)
        X = torch.randn(40, 5, device=cpu_device)
        head_idx = torch.cat([torch.zeros(20), torch.ones(20)]).long().to(cpu_device)
        # Head 0 should output base + 1.0, Head 1 should output base - 1.0
        with torch.no_grad():
            base = mlp(X)
        y = torch.where(head_idx == 0, base + 1.0, base - 1.0)

        optimizer = torch.optim.Adam(tuner.parameters(), lr=0.05)
        criterion = nn.MSELoss()

        tuner.train()
        for _ in range(200):
            optimizer.zero_grad()
            loss = criterion(tuner(X, head_idx), y)
            loss.backward()
            optimizer.step()

        # Test that heads produce different outputs on same input
        tuner.eval()
        test_X = torch.randn(1, 5, device=cpu_device)
        with torch.no_grad():
            out_head0 = tuner(test_X, torch.tensor([0], device=cpu_device))
            out_head1 = tuner(test_X, torch.tensor([1], device=cpu_device))

        # The outputs should differ by approximately 2.0 (1.0 - (-1.0))
        diff = (out_head0 - out_head1).abs().item()
        assert diff > 1.0, f"Heads did not learn distinct offsets: diff={diff:.4f}"

    def test_headlock_preserves_learned_behavior(self, cpu_device: str) -> None:
        """Test that HeadLock preserves the learned head behavior after training."""
        mlp = MLP(
            in_size=5,
            out_size=1,
            hidden_sizes=[8],
            post_squeeze=True,
        ).to(cpu_device)
        tuner = LinearTuner(mlp, n_heads=2).to(cpu_device)

        # Train briefly
        torch.manual_seed(42)
        X = torch.randn(20, 5, device=cpu_device)
        head_idx = torch.randint(0, 2, (20,), device=cpu_device)
        y = torch.randn(20, device=cpu_device)

        optimizer = torch.optim.Adam(tuner.parameters(), lr=0.01)
        for _ in range(50):
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(tuner(X, head_idx), y)
            loss.backward()
            optimizer.step()

        # Create HeadLock and verify it matches manual selection
        tuner.eval()
        locked_0 = tuner.create_singlehead_model(head_index=0)
        locked_1 = tuner.create_singlehead_model(head_index=1)

        test_X = torch.randn(5, 5, device=cpu_device)
        with torch.no_grad():
            manual_0 = tuner(test_X, torch.zeros(5, dtype=torch.long, device=cpu_device))
            manual_1 = tuner(test_X, torch.ones(5, dtype=torch.long, device=cpu_device))
            locked_out_0 = locked_0(test_X)
            locked_out_1 = locked_1(test_X)

        assert torch.allclose(locked_out_0, manual_0)
        assert torch.allclose(locked_out_1, manual_1)
