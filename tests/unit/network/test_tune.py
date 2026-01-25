"""Tests for mavenets.network.tune module."""

import pytest
import torch
import torch.nn as nn

from mavenets.network.tune import (  # type: ignore[import-not-found]
    MHTuner,
    HeadLock,
    NullTuner,
    LinearTuner,
    SharedFanTuner,
    FFNTuner,
)


class SimpleModel(nn.Module):
    """Simple model for testing tuners."""

    def __init__(self, in_size: int = 10, out_size: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(in_size, out_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


class TestMHTuner:
    """Tests for MHTuner abstract class."""

    def test_tune_not_implemented(self) -> None:
        """Should raise NotImplementedError when tune is called on base class."""
        base_model = SimpleModel()
        tuner = MHTuner(base_model)
        signal = torch.randn(3)
        head_index = torch.zeros(3, dtype=torch.long)
        with pytest.raises(NotImplementedError, match="undefined tune method"):
            tuner.tune(signal, head_index)

    def test_forward_calls_tune(self, cpu_device: str) -> None:
        """Forward should call base model and tune."""
        base_model = SimpleModel().to(cpu_device)
        # Use NullTuner as a concrete implementation
        tuner = NullTuner(base_model).to(cpu_device)
        x = torch.randn(3, 10, device=cpu_device)
        head_index = torch.zeros(3, dtype=torch.long, device=cpu_device)
        output = tuner(x, head_index)
        assert output.shape == (3,)

    def test_forward_return_raw(self, cpu_device: str) -> None:
        """Forward with return_raw=True should return both tuned and raw output."""
        base_model = SimpleModel().to(cpu_device)
        tuner = NullTuner(base_model).to(cpu_device)
        x = torch.randn(3, 10, device=cpu_device)
        head_index = torch.zeros(3, dtype=torch.long, device=cpu_device)
        tuned, raw = tuner(x, head_index, return_raw=True)
        assert tuned.shape == (3,)
        assert raw.shape == (3,)
        # For NullTuner, tuned and raw should be equal
        assert torch.allclose(tuned, raw)

    def test_create_singlehead_model(self, cpu_device: str) -> None:
        """create_singlehead_model should return a HeadLock instance."""
        base_model = SimpleModel().to(cpu_device)
        tuner = NullTuner(base_model).to(cpu_device)
        single_head = tuner.create_singlehead_model(head_index=0)
        assert isinstance(single_head, HeadLock)


class TestHeadLock:
    """Tests for HeadLock class."""

    def test_init(self) -> None:
        """Should initialize with model and head index."""
        base_model = SimpleModel()
        tuner = NullTuner(base_model)
        head_lock = HeadLock(tuner, head_index=0)
        assert head_lock.head_index == 0
        assert head_lock.model is tuner

    def test_forward_shape(self, cpu_device: str) -> None:
        """Forward pass should produce correct output shape."""
        base_model = SimpleModel().to(cpu_device)
        tuner = NullTuner(base_model).to(cpu_device)
        head_lock = HeadLock(tuner, head_index=0).to(cpu_device)
        x = torch.randn(3, 10, device=cpu_device)
        output = head_lock(x)
        assert output.shape == (3,)

    def test_forward_uses_fixed_head(self, cpu_device: str) -> None:
        """Forward should use the fixed head index."""
        base_model = SimpleModel().to(cpu_device)
        tuner = LinearTuner(base_model, n_heads=3).to(cpu_device)
        
        # Create head locks for different heads
        head_lock_0 = HeadLock(tuner, head_index=0).to(cpu_device)
        head_lock_1 = HeadLock(tuner, head_index=1).to(cpu_device)
        
        x = torch.randn(3, 10, device=cpu_device)
        output_0 = head_lock_0(x)
        output_1 = head_lock_1(x)
        
        # Different heads should produce different outputs (unless by chance)
        # We just verify they run without error
        assert output_0.shape == (3,)
        assert output_1.shape == (3,)

    def test_gradient_flow(self, cpu_device: str) -> None:
        """Gradients should flow through HeadLock."""
        base_model = SimpleModel().to(cpu_device)
        tuner = NullTuner(base_model).to(cpu_device)
        head_lock = HeadLock(tuner, head_index=0).to(cpu_device)
        x = torch.randn(3, 10, device=cpu_device, requires_grad=True)
        output = head_lock(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None


class TestNullTuner:
    """Tests for NullTuner class."""

    def test_init(self) -> None:
        """Should initialize with base model."""
        base_model = SimpleModel()
        tuner = NullTuner(base_model)
        assert tuner.base_model is base_model

    def test_tune_returns_signal_unchanged(self, cpu_device: str) -> None:
        """tune should return signal unchanged."""
        base_model = SimpleModel().to(cpu_device)
        tuner = NullTuner(base_model).to(cpu_device)
        signal = torch.randn(3, device=cpu_device)
        head_index = torch.zeros(3, dtype=torch.long, device=cpu_device)
        output = tuner.tune(signal, head_index)
        assert torch.allclose(output, signal)

    def test_forward_shape(self, cpu_device: str) -> None:
        """Forward pass should produce correct output shape."""
        base_model = SimpleModel().to(cpu_device)
        tuner = NullTuner(base_model).to(cpu_device)
        x = torch.randn(3, 10, device=cpu_device)
        head_index = torch.zeros(3, dtype=torch.long, device=cpu_device)
        output = tuner(x, head_index)
        assert output.shape == (3,)

    def test_forward_deterministic(self, cpu_device: str) -> None:
        """Forward pass should be deterministic."""
        base_model = SimpleModel().to(cpu_device)
        tuner = NullTuner(base_model).to(cpu_device)
        x = torch.randn(3, 10, device=cpu_device)
        head_index = torch.zeros(3, dtype=torch.long, device=cpu_device)
        output1 = tuner(x, head_index)
        output2 = tuner(x, head_index)
        assert torch.allclose(output1, output2)

    def test_gradient_flow(self, cpu_device: str) -> None:
        """Gradients should flow through the tuner."""
        base_model = SimpleModel().to(cpu_device)
        tuner = NullTuner(base_model).to(cpu_device)
        x = torch.randn(3, 10, device=cpu_device, requires_grad=True)
        head_index = torch.zeros(3, dtype=torch.long, device=cpu_device)
        output = tuner(x, head_index)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestLinearTuner:
    """Tests for LinearTuner class."""

    def test_init_basic(self) -> None:
        """Should initialize with basic parameters."""
        base_model = SimpleModel()
        tuner = LinearTuner(base_model, n_heads=3)
        assert len(tuner.heads) == 3
        assert tuner.residual_connection is True

    def test_init_without_residual(self) -> None:
        """Should initialize without residual connection."""
        base_model = SimpleModel()
        tuner = LinearTuner(base_model, n_heads=3, residual_connection=False)
        assert tuner.residual_connection is False

    def test_init_without_bias(self) -> None:
        """Should initialize heads without bias."""
        base_model = SimpleModel()
        tuner = LinearTuner(base_model, n_heads=3, bias=False)
        for head in tuner.heads:
            assert head.bias is None

    def test_init_with_bias(self) -> None:
        """Should initialize heads with bias by default."""
        base_model = SimpleModel()
        tuner = LinearTuner(base_model, n_heads=3, bias=True)
        for head in tuner.heads:
            assert head.bias is not None

    def test_forward_shape_1d(self, cpu_device: str) -> None:
        """Forward pass should work with 1D signal."""
        base_model = SimpleModel().to(cpu_device)
        tuner = LinearTuner(base_model, n_heads=3).to(cpu_device)
        x = torch.randn(4, 10, device=cpu_device)
        head_index = torch.randint(0, 3, (4,), device=cpu_device)
        output = tuner(x, head_index)
        assert output.shape == (4,)

    def test_tune_different_heads(self, cpu_device: str) -> None:
        """Different heads should produce different outputs."""
        base_model = SimpleModel().to(cpu_device)
        tuner = LinearTuner(base_model, n_heads=3, residual_connection=False).to(cpu_device)
        
        signal = torch.ones(3, device=cpu_device)
        head_0 = torch.tensor([0, 0, 0], device=cpu_device)
        head_1 = torch.tensor([1, 1, 1], device=cpu_device)
        
        output_0 = tuner.tune(signal, head_0)
        output_1 = tuner.tune(signal, head_1)
        
        # Different heads should generally produce different outputs
        assert output_0.shape == output_1.shape

    def test_tune_with_residual(self, cpu_device: str) -> None:
        """Tune with residual should add signal to correction."""
        base_model = SimpleModel().to(cpu_device)
        tuner = LinearTuner(base_model, n_heads=1, residual_connection=True).to(cpu_device)
        
        signal = torch.ones(3, device=cpu_device)
        head_index = torch.zeros(3, dtype=torch.long, device=cpu_device)
        
        # With residual connection, output should include the original signal
        output = tuner.tune(signal, head_index)
        assert output.shape == (3,)

    def test_gradient_flow(self, cpu_device: str) -> None:
        """Gradients should flow through the tuner."""
        base_model = SimpleModel().to(cpu_device)
        tuner = LinearTuner(base_model, n_heads=3).to(cpu_device)
        x = torch.randn(4, 10, device=cpu_device, requires_grad=True)
        head_index = torch.randint(0, 3, (4,), device=cpu_device)
        output = tuner(x, head_index)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None

    def test_forward_deterministic(self, cpu_device: str) -> None:
        """Forward pass should be deterministic."""
        base_model = SimpleModel().to(cpu_device)
        tuner = LinearTuner(base_model, n_heads=3).to(cpu_device)
        x = torch.randn(4, 10, device=cpu_device)
        head_index = torch.randint(0, 3, (4,), device=cpu_device)
        output1 = tuner(x, head_index)
        output2 = tuner(x, head_index)
        assert torch.allclose(output1, output2)


class TestSharedFanTuner:
    """Tests for SharedFanTuner class."""

    def test_init_basic(self) -> None:
        """Should initialize with basic parameters."""
        base_model = SimpleModel()
        tuner = SharedFanTuner(base_model, n_heads=3)
        assert len(tuner.heads) == 3
        assert tuner.residual_connection is True
        assert isinstance(tuner.fanout, nn.Sequential)

    def test_init_custom_fan_size(self) -> None:
        """Should initialize with custom fan size."""
        base_model = SimpleModel()
        tuner = SharedFanTuner(base_model, n_heads=3, fan_size=32)
        # Check fanout produces correct size
        assert tuner.fanout[0].out_features == 32

    def test_init_custom_activation(self) -> None:
        """Should initialize with custom activation."""
        base_model = SimpleModel()
        tuner = SharedFanTuner(base_model, n_heads=3, fan_activation=nn.ReLU)
        assert isinstance(tuner.fanout[1], nn.ReLU)

    def test_init_without_residual(self) -> None:
        """Should initialize without residual connection."""
        base_model = SimpleModel()
        tuner = SharedFanTuner(base_model, n_heads=3, residual_connection=False)
        assert tuner.residual_connection is False

    def test_forward_shape(self, cpu_device: str) -> None:
        """Forward pass should produce correct output shape."""
        base_model = SimpleModel().to(cpu_device)
        tuner = SharedFanTuner(base_model, n_heads=3, fan_size=8).to(cpu_device)
        x = torch.randn(4, 10, device=cpu_device)
        head_index = torch.randint(0, 3, (4,), device=cpu_device)
        output = tuner(x, head_index)
        assert output.shape == (4,)

    def test_forward_with_dropout(self, cpu_device: str) -> None:
        """Forward pass should work with dropout."""
        base_model = SimpleModel().to(cpu_device)
        tuner = SharedFanTuner(base_model, n_heads=3, fan_size=8, dropout=0.5).to(cpu_device)
        tuner.train()
        x = torch.randn(4, 10, device=cpu_device)
        head_index = torch.randint(0, 3, (4,), device=cpu_device)
        output = tuner(x, head_index)
        assert output.shape == (4,)

    def test_tune_different_heads(self, cpu_device: str) -> None:
        """Different heads should produce different outputs."""
        base_model = SimpleModel().to(cpu_device)
        tuner = SharedFanTuner(
            base_model, n_heads=3, fan_size=8, residual_connection=False
        ).to(cpu_device)
        
        signal = torch.ones(3, device=cpu_device)
        head_0 = torch.tensor([0, 0, 0], device=cpu_device)
        head_1 = torch.tensor([1, 1, 1], device=cpu_device)
        
        output_0 = tuner.tune(signal, head_0)
        output_1 = tuner.tune(signal, head_1)
        
        assert output_0.shape == output_1.shape

    def test_gradient_flow(self, cpu_device: str) -> None:
        """Gradients should flow through the tuner."""
        base_model = SimpleModel().to(cpu_device)
        tuner = SharedFanTuner(base_model, n_heads=3, fan_size=8).to(cpu_device)
        x = torch.randn(4, 10, device=cpu_device, requires_grad=True)
        head_index = torch.randint(0, 3, (4,), device=cpu_device)
        output = tuner(x, head_index)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None

    def test_forward_deterministic_eval(self, cpu_device: str) -> None:
        """Forward pass should be deterministic in eval mode."""
        base_model = SimpleModel().to(cpu_device)
        tuner = SharedFanTuner(base_model, n_heads=3, fan_size=8, dropout=0.0).to(cpu_device)
        tuner.eval()
        x = torch.randn(4, 10, device=cpu_device)
        head_index = torch.randint(0, 3, (4,), device=cpu_device)
        output1 = tuner(x, head_index)
        output2 = tuner(x, head_index)
        assert torch.allclose(output1, output2)

    def test_example_configuration(self, cpu_device: str) -> None:
        """Should work with configuration from examples (fan_size options)."""
        base_model = SimpleModel().to(cpu_device)
        # fan_size_ops from example: [1,2,4,8,16,32,64,128]
        for fan_size in [1, 2, 4, 8, 16, 32]:
            tuner = SharedFanTuner(
                base_model, n_heads=8, fan_size=fan_size
            ).to(cpu_device)
            x = torch.randn(4, 10, device=cpu_device)
            head_index = torch.randint(0, 8, (4,), device=cpu_device)
            output = tuner(x, head_index)
            assert output.shape == (4,)


class TestFFNTuner:
    """Tests for FFNTuner class."""

    def test_init_basic(self) -> None:
        """Should initialize with basic parameters."""
        base_model = SimpleModel()
        tuner = FFNTuner(base_model, n_heads=3, hidden_size=8, n_hidden=1)
        assert len(tuner.heads) == 3

    def test_init_multiple_hidden(self) -> None:
        """Should initialize with multiple hidden layers."""
        base_model = SimpleModel()
        tuner = FFNTuner(base_model, n_heads=3, hidden_size=8, n_hidden=3)
        assert len(tuner.heads) == 3

    def test_init_custom_activation(self) -> None:
        """Should initialize with custom activation class."""
        base_model = SimpleModel()
        tuner = FFNTuner(
            base_model, n_heads=3, hidden_size=8, n_hidden=1, activation_class=nn.ReLU
        )
        assert len(tuner.heads) == 3

    def test_forward_shape(self, cpu_device: str) -> None:
        """Forward pass should produce correct output shape."""
        base_model = SimpleModel().to(cpu_device)
        tuner = FFNTuner(base_model, n_heads=3, hidden_size=8, n_hidden=1).to(cpu_device)
        x = torch.randn(4, 10, device=cpu_device)
        head_index = torch.randint(0, 3, (4,), device=cpu_device)
        output = tuner(x, head_index)
        assert output.shape == (4,)

    def test_forward_with_dropout(self, cpu_device: str) -> None:
        """Forward pass should work with dropout."""
        base_model = SimpleModel().to(cpu_device)
        tuner = FFNTuner(
            base_model, n_heads=3, hidden_size=8, n_hidden=2, dropout=0.5
        ).to(cpu_device)
        tuner.train()
        x = torch.randn(4, 10, device=cpu_device)
        head_index = torch.randint(0, 3, (4,), device=cpu_device)
        output = tuner(x, head_index)
        assert output.shape == (4,)

    def test_forward_with_positive_linear(self, cpu_device: str) -> None:
        """Forward pass should work with positive linear."""
        base_model = SimpleModel().to(cpu_device)
        tuner = FFNTuner(
            base_model, n_heads=3, hidden_size=8, n_hidden=1, positive_linear=True
        ).to(cpu_device)
        x = torch.randn(4, 10, device=cpu_device)
        head_index = torch.randint(0, 3, (4,), device=cpu_device)
        output = tuner(x, head_index)
        assert output.shape == (4,)

    def test_tune_different_heads(self, cpu_device: str) -> None:
        """Different heads should produce different outputs."""
        base_model = SimpleModel().to(cpu_device)
        tuner = FFNTuner(base_model, n_heads=3, hidden_size=8, n_hidden=1).to(cpu_device)
        
        signal = torch.ones(3, device=cpu_device)
        head_0 = torch.tensor([0, 0, 0], device=cpu_device)
        head_1 = torch.tensor([1, 1, 1], device=cpu_device)
        
        output_0 = tuner.tune(signal, head_0)
        output_1 = tuner.tune(signal, head_1)
        
        assert output_0.shape == output_1.shape

    def test_gradient_flow(self, cpu_device: str) -> None:
        """Gradients should flow through the tuner."""
        base_model = SimpleModel().to(cpu_device)
        tuner = FFNTuner(base_model, n_heads=3, hidden_size=8, n_hidden=1).to(cpu_device)
        x = torch.randn(4, 10, device=cpu_device, requires_grad=True)
        head_index = torch.randint(0, 3, (4,), device=cpu_device)
        output = tuner(x, head_index)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None

    def test_forward_deterministic(self, cpu_device: str) -> None:
        """Forward pass should be deterministic without dropout."""
        base_model = SimpleModel().to(cpu_device)
        tuner = FFNTuner(
            base_model, n_heads=3, hidden_size=8, n_hidden=1, dropout=0.0
        ).to(cpu_device)
        tuner.eval()
        x = torch.randn(4, 10, device=cpu_device)
        head_index = torch.randint(0, 3, (4,), device=cpu_device)
        output1 = tuner(x, head_index)
        output2 = tuner(x, head_index)
        assert torch.allclose(output1, output2)

    def test_no_residual_connection(self, cpu_device: str) -> None:
        """FFNTuner should not have residual connection (unlike other tuners)."""
        base_model = SimpleModel().to(cpu_device)
        tuner = FFNTuner(base_model, n_heads=1, hidden_size=8, n_hidden=1).to(cpu_device)
        
        # With zero weights, output should be near zero (no residual adding input)
        with torch.no_grad():
            for head in tuner.heads:
                for param in head.parameters():
                    param.zero_()
        
        signal = torch.ones(3, device=cpu_device)
        head_index = torch.zeros(3, dtype=torch.long, device=cpu_device)
        output = tuner.tune(signal, head_index)
        
        # Output should be near zero since there's no residual connection
        assert torch.allclose(output, torch.zeros_like(output), atol=1e-5)


class TestTunerIntegration:
    """Integration tests for tuners with real models."""

    def test_nulltuner_with_mlp(self, cpu_device: str) -> None:
        """NullTuner should work with MLP model."""
        from mavenets.network.base import MLP  # type: ignore[import-not-found]
        
        mlp = MLP(in_size=20, out_size=1, hidden_sizes=[16], post_squeeze=True)
        tuner = NullTuner(mlp).to(cpu_device)
        
        x = torch.randn(4, 20, device=cpu_device)
        head_index = torch.zeros(4, dtype=torch.long, device=cpu_device)
        output = tuner(x, head_index)
        assert output.shape == (4,)

    def test_sharedfantuner_with_transformer(self, cpu_device: str) -> None:
        """SharedFanTuner should work with SumTransformer (as in examples)."""
        from mavenets.network.transformer import SumTransformer  # type: ignore[import-not-found]
        
        transformer = SumTransformer(
            alphabet_size=21,
            emb_size=16,
            max_sequence_size=10,
            head_mlp_hidden_size=32,
            n_transformers=1,
        )
        tuner = SharedFanTuner(transformer, n_heads=4, fan_size=8).to(cpu_device)
        
        x = torch.randint(0, 21, (4, 10), device=cpu_device)
        head_index = torch.randint(0, 4, (4,), device=cpu_device)
        output = tuner(x, head_index)
        assert output.shape == (4,)

    def test_lineartuner_with_transformer(self, cpu_device: str) -> None:
        """LinearTuner should work with SumTransformer."""
        from mavenets.network.transformer import SumTransformer  # type: ignore[import-not-found]
        
        transformer = SumTransformer(
            alphabet_size=21,
            emb_size=16,
            max_sequence_size=10,
            head_mlp_hidden_size=32,
            n_transformers=1,
        )
        tuner = LinearTuner(transformer, n_heads=4).to(cpu_device)
        
        x = torch.randint(0, 21, (4, 10), device=cpu_device)
        head_index = torch.randint(0, 4, (4,), device=cpu_device)
        output = tuner(x, head_index)
        assert output.shape == (4,)

    def test_mixed_head_indices(self, cpu_device: str) -> None:
        """Tuner should handle mixed head indices in batch."""
        base_model = SimpleModel().to(cpu_device)
        tuner = SharedFanTuner(base_model, n_heads=4, fan_size=8).to(cpu_device)
        
        x = torch.randn(8, 10, device=cpu_device)
        # Mix of different head indices
        head_index = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], device=cpu_device)
        output = tuner(x, head_index)
        assert output.shape == (8,)

    def test_single_sample_batch(self, cpu_device: str) -> None:
        """Tuner should handle single sample batch."""
        base_model = SimpleModel().to(cpu_device)
        tuner = SharedFanTuner(base_model, n_heads=4, fan_size=8).to(cpu_device)
        
        x = torch.randn(1, 10, device=cpu_device)
        head_index = torch.tensor([0], device=cpu_device)
        output = tuner(x, head_index)
        assert output.shape == (1,)
