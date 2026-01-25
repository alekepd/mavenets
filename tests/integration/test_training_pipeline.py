"""Integration tests for the training pipeline.

These tests verify that components work together correctly.
They use small models and synthetic data to keep execution fast.
"""

from typing import List

import pytest
import torch
import torch.nn as nn

from mavenets.network.base import MLP  # type: ignore[import-not-found]
from mavenets.data.featurize.core import IntEncoder, int_to_floatonehot  # type: ignore[import-not-found]


class TestEncodingToNetworkPipeline:
    """Test the pipeline from encoding to network forward pass."""

    @pytest.fixture
    def small_encoder(self) -> IntEncoder:
        """Create a small encoder for testing."""
        return IntEncoder(["A", "C", "D", "E", "F"])

    @pytest.fixture
    def small_mlp(self, cpu_device: str) -> MLP:
        """Create a small MLP for testing."""
        # 5 amino acids * 4 positions = 20 input features (one-hot)
        return MLP(
            in_size=20,
            out_size=1,
            hidden_sizes=[8, 4],
            post_squeeze=True,
        ).to(cpu_device)

    def test_encode_onehot_forward(
        self,
        small_encoder: IntEncoder,
        small_mlp: MLP,
        cpu_device: str,
    ) -> None:
        """Test encoding sequences and passing through network."""
        sequences = ["ACED", "DEFA", "ACDF"]

        # Encode sequences
        encoded = small_encoder.batch_encode(sequences, device=cpu_device)
        assert encoded.shape == (3, 4)

        # Convert to one-hot
        onehot = int_to_floatonehot(encoded, num_classes=5)
        assert onehot.shape == (3, 4, 5)

        # Flatten for MLP
        flat = onehot.view(3, -1)
        assert flat.shape == (3, 20)

        # Forward pass
        output = small_mlp(flat)
        assert output.shape == (3,)

    def test_gradient_flow_through_pipeline(
        self,
        small_encoder: IntEncoder,
        small_mlp: MLP,
        cpu_device: str,
    ) -> None:
        """Test that gradients flow through the pipeline."""
        sequences = ["ACED", "DEFA"]

        # Encode and convert to one-hot
        encoded = small_encoder.batch_encode(sequences, device=cpu_device)
        onehot = int_to_floatonehot(encoded, num_classes=5)
        flat = onehot.view(2, -1)
        flat.requires_grad_(True)

        # Forward pass
        output = small_mlp(flat)

        # Compute loss and backward
        target = torch.tensor([1.0, 0.5], device=cpu_device)
        loss = nn.functional.mse_loss(output, target)
        loss.backward()  # type: ignore[no-untyped-call]

        # Check gradients exist
        assert flat.grad is not None
        assert not torch.isnan(flat.grad).any()


@pytest.mark.slow
class TestTrainingLoop:
    """Test a minimal training loop."""

    def test_mlp_training_reduces_loss(self, cpu_device: str) -> None:
        """Test that training actually reduces loss."""
        # Create small model
        model = MLP(
            in_size=10,
            out_size=1,
            hidden_sizes=[16, 8],
            post_squeeze=True,
        ).to(cpu_device)

        # Create synthetic data
        torch.manual_seed(42)  # type: ignore[no-untyped-call]
        X = torch.randn(50, 10, device=cpu_device)
        y = X[:, 0] + 0.5 * X[:, 1]  # Simple linear relationship

        # Training setup
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        # Record initial loss
        model.eval()
        with torch.no_grad():
            initial_loss = criterion(model(X), y).item()

        # Train for a few epochs
        model.train()
        for _ in range(100):
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

        # Check final loss is lower
        model.eval()
        with torch.no_grad():
            final_loss = criterion(model(X), y).item()

        assert final_loss < initial_loss, (
            f"Training did not reduce loss: {initial_loss:.4f} -> {final_loss:.4f}"
        )

    def test_mlp_overfits_small_dataset(self, cpu_device: str) -> None:
        """Test that MLP can overfit a small dataset (sanity check)."""
        model = MLP(
            in_size=5,
            out_size=1,
            hidden_sizes=[32, 32],
            post_squeeze=True,
        ).to(cpu_device)

        # Very small dataset
        torch.manual_seed(42)  # type: ignore[no-untyped-call]
        X = torch.randn(5, 5, device=cpu_device)
        y = torch.randn(5, device=cpu_device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        # Train until overfit
        model.train()
        for _ in range(500):
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()

        # Should achieve very low loss on training data
        model.eval()
        with torch.no_grad():
            final_loss = criterion(model(X), y).item()

        assert final_loss < 0.01, f"Model failed to overfit small dataset: loss={final_loss:.4f}"


@pytest.mark.integration
class TestMCMCSimulation:
    """Test MCMC simulation components working together."""

    def test_metsim_basic_run(self, cpu_device: str) -> None:
        """Test MetSim can run a basic simulation."""
        from mavenets.sample.step import MetSim, IntMutate, State  # type: ignore[import-not-found]

        # Simple energy function (prefer lower values)
        def energy_fn(x: torch.Tensor) -> torch.Tensor:
            result: torch.Tensor = x.float().mean(dim=-1)
            return result

        # Create mutator and simulator
        mutator = IntMutate(min_int=0, max_int=10)
        sim = MetSim(
            model=energy_fn,
            proposer=mutator,
            batch_size=32,
            beta=1.0,
            jump_stride=5,
        )

        # Run short simulation
        start_seq = torch.tensor([5, 5, 5, 5, 5], device=cpu_device, dtype=torch.int32)
        start_state = State(index=0, sequence=start_seq)

        frames: List[State] = []
        state: State = start_state
        frames.append(state)
        for _ in range(10):
            state = sim.stepper(state)
            frames.append(state)

        # Check we got states
        assert len(frames) == 11
        assert all(isinstance(f, State) for f in frames)
        # Indices should be increasing
        assert all(frames[i].index <= frames[i + 1].index for i in range(len(frames) - 1))
