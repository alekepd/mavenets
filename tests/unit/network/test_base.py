"""Tests for mavenets.network.base module."""

import pytest
import torch
import torch.nn as nn

from mavenets.network.base import ELULinear, FFLayer, BaseFFN, MLP


class TestELULinear:
    """Tests for ELULinear class."""

    def test_init(self, cpu_device: str) -> None:
        """Should initialize with correct sizes."""
        layer = ELULinear(10, 5)
        assert layer.in_size == 10
        assert layer.out_size == 5
        assert layer.weights.shape == (10, 5)

    def test_init_with_bias(self) -> None:
        """Should initialize bias when bias=True."""
        layer = ELULinear(10, 5, bias=True)
        assert layer.bias is not None
        assert layer.bias.shape == (5,)

    def test_init_without_bias(self) -> None:
        """Should not have bias when bias=False."""
        layer = ELULinear(10, 5, bias=False)
        assert layer.bias is None

    def test_init_negative_leak_raises(self) -> None:
        """Should raise ValueError for negative leak."""
        with pytest.raises(ValueError, match="leak must be positive"):
            ELULinear(10, 5, leak=-0.1)

    def test_forward_shape(self, cpu_device: str) -> None:
        """Forward pass should produce correct output shape."""
        layer = ELULinear(10, 5).to(cpu_device)
        x = torch.randn(3, 10, device=cpu_device)
        output = layer(x)
        assert output.shape == (3, 5)

    def test_forward_deterministic(self, cpu_device: str) -> None:
        """Forward pass should be deterministic."""
        layer = ELULinear(10, 5).to(cpu_device)
        x = torch.randn(3, 10, device=cpu_device)
        output1 = layer(x)
        output2 = layer(x)
        assert torch.allclose(output1, output2)


class TestFFLayer:
    """Tests for FFLayer class."""

    def test_init_basic(self) -> None:
        """Should initialize with basic parameters."""
        layer = FFLayer(10, 5)
        assert isinstance(layer.affine, nn.Linear)
        assert isinstance(layer.activation, nn.LeakyReLU)

    def test_init_with_residual(self) -> None:
        """Should allow residual connection when sizes match."""
        layer = FFLayer(10, 10, residual_connection=True)
        assert layer.residual_connection is True

    def test_init_residual_size_mismatch_raises(self) -> None:
        """Should raise ValueError for residual with mismatched sizes."""
        with pytest.raises(ValueError, match="residual connection"):
            FFLayer(10, 5, residual_connection=True)

    def test_forward_shape(self, cpu_device: str) -> None:
        """Forward pass should produce correct output shape."""
        layer = FFLayer(10, 5).to(cpu_device)
        x = torch.randn(3, 10, device=cpu_device)
        output = layer(x)
        assert output.shape == (3, 5)

    def test_forward_with_residual(self, cpu_device: str) -> None:
        """Forward with residual should add input to output."""
        layer = FFLayer(10, 10, residual_connection=True).to(cpu_device)
        x = torch.randn(3, 10, device=cpu_device)
        output = layer(x)
        assert output.shape == (3, 10)

    def test_forward_with_layer_norm(self, cpu_device: str) -> None:
        """Forward with layer norm should normalize input."""
        layer = FFLayer(10, 5, pre_layer_norm=True).to(cpu_device)
        x = torch.randn(3, 10, device=cpu_device)
        output = layer(x)
        assert output.shape == (3, 5)

    def test_forward_with_dropout(self, cpu_device: str) -> None:
        """Forward with dropout should work in train mode."""
        layer = FFLayer(10, 5, dropout=0.5).to(cpu_device)
        layer.train()
        x = torch.randn(3, 10, device=cpu_device)
        output = layer(x)
        assert output.shape == (3, 5)


class TestBaseFFN:
    """Tests for BaseFFN class."""

    def test_init_basic(self) -> None:
        """Should initialize with basic parameters."""
        net = BaseFFN(in_size=10, hidden_size=20, n_hidden=2, out_size=5)
        assert isinstance(net, nn.Module)

    def test_init_invalid_hidden_raises(self) -> None:
        """Should raise ValueError for non-positive hidden layers."""
        with pytest.raises(ValueError, match="positive number of hidden layers"):
            BaseFFN(in_size=10, hidden_size=20, n_hidden=0, out_size=5)

    def test_init_global_residual_size_mismatch_raises(self) -> None:
        """Should raise ValueError for global residual with mismatched sizes."""
        with pytest.raises(ValueError, match="Global residual connection"):
            BaseFFN(
                in_size=10,
                hidden_size=20,
                n_hidden=2,
                out_size=5,
                global_residual_connection=True,
            )

    def test_forward_shape(self, cpu_device: str) -> None:
        """Forward pass should produce correct output shape."""
        net = BaseFFN(in_size=10, hidden_size=20, n_hidden=2, out_size=5).to(cpu_device)
        x = torch.randn(3, 10, device=cpu_device)
        output = net(x)
        assert output.shape == (3, 5)

    def test_forward_default_out_size(self, cpu_device: str) -> None:
        """Default out_size should equal in_size."""
        net = BaseFFN(in_size=10, hidden_size=20, n_hidden=2).to(cpu_device)
        x = torch.randn(3, 10, device=cpu_device)
        output = net(x)
        assert output.shape == (3, 10)

    def test_forward_with_residual(self, cpu_device: str) -> None:
        """Forward with residual connections should work."""
        net = BaseFFN(
            in_size=10, hidden_size=10, n_hidden=2, out_size=10, residual_connection=True
        ).to(cpu_device)
        x = torch.randn(3, 10, device=cpu_device)
        output = net(x)
        assert output.shape == (3, 10)

    def test_forward_with_global_residual(self, cpu_device: str) -> None:
        """Forward with global residual should add input to output."""
        net = BaseFFN(
            in_size=10,
            hidden_size=20,
            n_hidden=2,
            out_size=10,
            global_residual_connection=True,
        ).to(cpu_device)
        x = torch.randn(3, 10, device=cpu_device)
        output = net(x)
        assert output.shape == (3, 10)

    def test_forward_with_scale(self, cpu_device: str) -> None:
        """Forward with scale should multiply output."""
        net = BaseFFN(
            in_size=10, hidden_size=20, n_hidden=2, out_size=5, scale=2.0
        ).to(cpu_device)
        x = torch.randn(3, 10, device=cpu_device)
        output = net(x)
        assert output.shape == (3, 5)

    def test_forward_with_positive_linear(self, cpu_device: str) -> None:
        """Forward with positive linear should use ELULinear."""
        net = BaseFFN(
            in_size=10, hidden_size=20, n_hidden=2, out_size=5, positive_linear=True
        ).to(cpu_device)
        x = torch.randn(3, 10, device=cpu_device)
        output = net(x)
        assert output.shape == (3, 5)


class TestMLP:
    """Tests for MLP class."""

    def test_init_basic(self) -> None:
        """Should initialize with basic parameters."""
        net = MLP(in_size=10, out_size=5, hidden_sizes=[20, 15])
        assert isinstance(net, nn.Module)

    def test_forward_shape(self, cpu_device: str) -> None:
        """Forward pass should produce correct output shape."""
        net = MLP(in_size=10, out_size=5, hidden_sizes=[20, 15]).to(cpu_device)
        x = torch.randn(3, 10, device=cpu_device)
        output = net(x)
        assert output.shape == (3, 5)

    def test_forward_empty_hidden(self, cpu_device: str) -> None:
        """Forward with no hidden layers should work."""
        net = MLP(in_size=10, out_size=5, hidden_sizes=[]).to(cpu_device)
        x = torch.randn(3, 10, device=cpu_device)
        output = net(x)
        assert output.shape == (3, 5)

    def test_forward_with_pre_flatten(self, cpu_device: str) -> None:
        """Forward with pre_flatten should flatten input."""
        net = MLP(
            in_size=20, out_size=5, hidden_sizes=[15], pre_flatten=True
        ).to(cpu_device)
        x = torch.randn(3, 4, 5, device=cpu_device)  # 4*5 = 20
        output = net(x)
        assert output.shape == (3, 5)

    def test_forward_with_post_squeeze(self, cpu_device: str) -> None:
        """Forward with post_squeeze should squeeze output."""
        net = MLP(
            in_size=10, out_size=1, hidden_sizes=[15], post_squeeze=True
        ).to(cpu_device)
        x = torch.randn(3, 10, device=cpu_device)
        output = net(x)
        assert output.shape == (3,)

    def test_forward_with_dropout(self, cpu_device: str) -> None:
        """Forward with dropout should work in train mode."""
        net = MLP(
            in_size=10, out_size=5, hidden_sizes=[20], dropout=0.5
        ).to(cpu_device)
        net.train()
        x = torch.randn(3, 10, device=cpu_device)
        output = net(x)
        assert output.shape == (3, 5)

    def test_triangle_network(self, cpu_device: str) -> None:
        """Triangle network should create decreasing layer sizes."""
        net = MLP.triangle_network(
            in_size=10, out_size=5, hidden_size=32, n_hidden=3
        ).to(cpu_device)
        x = torch.randn(3, 10, device=cpu_device)
        output = net(x)
        assert output.shape == (3, 5)

    def test_triangle_network_invalid_size_raises(self) -> None:
        """Triangle network should raise for sizes becoming non-positive."""
        with pytest.raises(ValueError, match="non-positive sizes"):
            MLP.triangle_network(in_size=10, out_size=5, hidden_size=2, n_hidden=5)

    def test_gradient_flow(self, cpu_device: str) -> None:
        """Gradients should flow through the network."""
        net = MLP(in_size=10, out_size=5, hidden_sizes=[20, 15]).to(cpu_device)
        x = torch.randn(3, 10, device=cpu_device, requires_grad=True)
        output = net(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape
