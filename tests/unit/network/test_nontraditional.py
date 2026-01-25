"""Tests for mavenets.network.nontraditional module."""

import pytest
import torch
import torch.nn as nn

from mavenets.network.nontraditional import LRMLP  # type: ignore[import-not-found]


class TestLRMLP:
    """Tests for LRMLP class."""

    def test_init_basic(self) -> None:
        """Should initialize with basic parameters."""
        net = LRMLP(in_size=10, out_size=5, augment_channel_size=8)
        assert isinstance(net, nn.Module)
        assert isinstance(net.side_transform, nn.Sequential)
        assert isinstance(net.mix_transform, nn.Linear)

    def test_init_side_transform_structure(self) -> None:
        """Side transform should have linear layer and activation."""
        net = LRMLP(in_size=10, out_size=5, augment_channel_size=8)
        assert isinstance(net.side_transform[0], nn.Linear)
        assert net.side_transform[0].in_features == 10
        assert net.side_transform[0].out_features == 8
        assert isinstance(net.side_transform[1], nn.LeakyReLU)

    def test_init_mix_transform_size(self) -> None:
        """Mix transform should take concatenated input."""
        net = LRMLP(in_size=10, out_size=5, augment_channel_size=8)
        # Mix transform input = in_size + augment_channel_size
        assert net.mix_transform.in_features == 10 + 8
        assert net.mix_transform.out_features == 5

    def test_init_custom_activation(self) -> None:
        """Should initialize with custom activation class."""
        net = LRMLP(
            in_size=10, out_size=5, augment_channel_size=8, activation_class=nn.ReLU
        )
        assert isinstance(net.side_transform[1], nn.ReLU)

    def test_init_with_pre_flatten(self) -> None:
        """Should initialize with pre_flatten."""
        net = LRMLP(
            in_size=20, out_size=5, augment_channel_size=8, pre_flatten=True
        )
        assert isinstance(net.preprocess, nn.Flatten)

    def test_init_without_pre_flatten(self) -> None:
        """Should not flatten by default."""
        net = LRMLP(in_size=10, out_size=5, augment_channel_size=8)
        # preprocess should be identity-like
        x = torch.randn(3, 10)
        assert torch.allclose(net.preprocess(x), x)

    def test_forward_shape(self, cpu_device: str) -> None:
        """Forward pass should produce correct output shape."""
        net = LRMLP(in_size=10, out_size=5, augment_channel_size=8).to(cpu_device)
        x = torch.randn(3, 10, device=cpu_device)
        output = net(x)
        assert output.shape == (3, 5)

    def test_forward_single_output(self, cpu_device: str) -> None:
        """Forward pass should work with single output dimension."""
        net = LRMLP(in_size=10, out_size=1, augment_channel_size=8).to(cpu_device)
        x = torch.randn(3, 10, device=cpu_device)
        output = net(x)
        assert output.shape == (3, 1)

    def test_forward_with_post_squeeze(self, cpu_device: str) -> None:
        """Forward with post_squeeze should squeeze output."""
        net = LRMLP(
            in_size=10, out_size=1, augment_channel_size=8, post_squeeze=True
        ).to(cpu_device)
        x = torch.randn(3, 10, device=cpu_device)
        output = net(x)
        assert output.shape == (3,)

    def test_forward_with_pre_flatten(self, cpu_device: str) -> None:
        """Forward with pre_flatten should flatten input."""
        net = LRMLP(
            in_size=20, out_size=5, augment_channel_size=8, pre_flatten=True
        ).to(cpu_device)
        x = torch.randn(3, 4, 5, device=cpu_device)  # 4*5 = 20
        output = net(x)
        assert output.shape == (3, 5)

    def test_forward_with_pre_flatten_and_post_squeeze(self, cpu_device: str) -> None:
        """Forward with both pre_flatten and post_squeeze."""
        net = LRMLP(
            in_size=20,
            out_size=1,
            augment_channel_size=8,
            pre_flatten=True,
            post_squeeze=True,
        ).to(cpu_device)
        x = torch.randn(3, 4, 5, device=cpu_device)  # 4*5 = 20
        output = net(x)
        assert output.shape == (3,)

    def test_forward_deterministic(self, cpu_device: str) -> None:
        """Forward pass should be deterministic."""
        net = LRMLP(in_size=10, out_size=5, augment_channel_size=8).to(cpu_device)
        x = torch.randn(3, 10, device=cpu_device)
        output1 = net(x)
        output2 = net(x)
        assert torch.allclose(output1, output2)

    def test_gradient_flow(self, cpu_device: str) -> None:
        """Gradients should flow through the network."""
        net = LRMLP(in_size=10, out_size=5, augment_channel_size=8).to(cpu_device)
        x = torch.randn(3, 10, device=cpu_device, requires_grad=True)
        output = net(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_gradient_flow_all_parameters(self, cpu_device: str) -> None:
        """All parameters should receive gradients."""
        net = LRMLP(in_size=10, out_size=5, augment_channel_size=8).to(cpu_device)
        x = torch.randn(3, 10, device=cpu_device)
        output = net(x)
        loss = output.sum()
        loss.backward()
        for name, param in net.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_different_augment_sizes(self, cpu_device: str) -> None:
        """Should work with different augment channel sizes."""
        for augment_size in [1, 4, 16, 32]:
            net = LRMLP(
                in_size=10, out_size=5, augment_channel_size=augment_size
            ).to(cpu_device)
            x = torch.randn(3, 10, device=cpu_device)
            output = net(x)
            assert output.shape == (3, 5)

    def test_different_batch_sizes(self, cpu_device: str) -> None:
        """Should work with different batch sizes."""
        net = LRMLP(in_size=10, out_size=5, augment_channel_size=8).to(cpu_device)
        for batch_size in [1, 4, 16, 32]:
            x = torch.randn(batch_size, 10, device=cpu_device)
            output = net(x)
            assert output.shape == (batch_size, 5)

    def test_large_augment_size(self, cpu_device: str) -> None:
        """Should work with augment size larger than input."""
        net = LRMLP(in_size=10, out_size=5, augment_channel_size=100).to(cpu_device)
        x = torch.randn(3, 10, device=cpu_device)
        output = net(x)
        assert output.shape == (3, 5)

    def test_small_augment_size(self, cpu_device: str) -> None:
        """Should work with small augment size."""
        net = LRMLP(in_size=10, out_size=5, augment_channel_size=1).to(cpu_device)
        x = torch.randn(3, 10, device=cpu_device)
        output = net(x)
        assert output.shape == (3, 5)

    def test_side_transform_affects_output(self, cpu_device: str) -> None:
        """Side transform should affect the output."""
        net = LRMLP(in_size=10, out_size=5, augment_channel_size=8).to(cpu_device)
        net.eval()
        x = torch.randn(3, 10, device=cpu_device)
        
        # Get original output
        output1 = net(x).clone()
        
        # Modify side transform weights
        with torch.no_grad():
            net.side_transform[0].weight.add_(1.0)
        
        # Output should be different
        output2 = net(x)
        assert not torch.allclose(output1, output2)

    def test_activation_applied(self, cpu_device: str) -> None:
        """Activation should be applied after linear transform."""
        # Use ReLU to make the effect observable
        net = LRMLP(
            in_size=10, out_size=5, augment_channel_size=8, activation_class=nn.ReLU
        ).to(cpu_device)
        
        # Set weights to produce negative pre-activation values
        with torch.no_grad():
            net.side_transform[0].weight.fill_(-1.0)
            net.side_transform[0].bias.fill_(-1.0)
        
        x = torch.ones(3, 10, device=cpu_device)
        
        # Get the side transform output (after ReLU, should be zeros)
        side_output = net.side_transform(x)
        assert (side_output >= 0).all()  # ReLU output should be non-negative

    def test_elu_activation(self, cpu_device: str) -> None:
        """Should work with ELU activation."""
        net = LRMLP(
            in_size=10, out_size=5, augment_channel_size=8, activation_class=nn.ELU
        ).to(cpu_device)
        assert isinstance(net.side_transform[1], nn.ELU)
        x = torch.randn(3, 10, device=cpu_device)
        output = net(x)
        assert output.shape == (3, 5)

    def test_silu_activation(self, cpu_device: str) -> None:
        """Should work with SiLU activation."""
        net = LRMLP(
            in_size=10, out_size=5, augment_channel_size=8, activation_class=nn.SiLU
        ).to(cpu_device)
        assert isinstance(net.side_transform[1], nn.SiLU)
        x = torch.randn(3, 10, device=cpu_device)
        output = net(x)
        assert output.shape == (3, 5)
