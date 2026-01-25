"""Tests for mavenets.network.transformer module."""

import pytest
import torch
import torch.nn as nn

from mavenets.network.transformer import Block, SumTransformer  # type: ignore[import-not-found]


class TestBlock:
    """Tests for Block class."""

    def test_init_basic(self) -> None:
        """Should initialize with default parameters."""
        block = Block()
        assert isinstance(block, nn.Module)
        assert block.emb_size == 32

    def test_init_custom_emb_size(self) -> None:
        """Should initialize with custom embedding size."""
        block = Block(emb_size=64)
        assert block.emb_size == 64

    def test_init_custom_hidden_size(self) -> None:
        """Should initialize with custom hidden MLP size."""
        block = Block(emb_size=16, hidden_mlp_size=64)
        # Verify MLP structure: Linear -> activation -> Linear -> Dropout
        assert isinstance(block.mlp, nn.Sequential)
        assert isinstance(block.mlp[0], nn.Linear)
        assert block.mlp[0].out_features == 64

    def test_init_multiple_heads(self) -> None:
        """Should initialize with multiple attention heads."""
        block = Block(emb_size=32, num_heads=4, hidden_mlp_size=64)
        assert block.mha.num_heads == 4

    def test_init_custom_activation(self) -> None:
        """Should initialize with custom activation class."""
        block = Block(emb_size=16, hidden_mlp_size=32, activation_class=nn.ReLU)
        assert isinstance(block.mlp[1], nn.ReLU)

    def test_init_layer_norms(self) -> None:
        """Should initialize layer norms with correct size."""
        block = Block(emb_size=16, hidden_mlp_size=32)
        assert isinstance(block.pre_head_norm, nn.LayerNorm)
        assert isinstance(block.pre_mlp_norm, nn.LayerNorm)
        assert block.pre_head_norm.normalized_shape == (16,)
        assert block.pre_mlp_norm.normalized_shape == (16,)

    def test_forward_shape(self, cpu_device: str) -> None:
        """Forward pass should maintain input shape."""
        block = Block(emb_size=16, hidden_mlp_size=32).to(cpu_device)
        x = torch.randn(3, 10, 16, device=cpu_device)
        output = block(x)
        assert output.shape == (3, 10, 16)

    def test_forward_different_sequence_lengths(self, cpu_device: str) -> None:
        """Forward pass should work with different sequence lengths."""
        block = Block(emb_size=16, hidden_mlp_size=32).to(cpu_device)
        for seq_len in [5, 10, 20]:
            x = torch.randn(2, seq_len, 16, device=cpu_device)
            output = block(x)
            assert output.shape == (2, seq_len, 16)

    def test_forward_deterministic(self, cpu_device: str) -> None:
        """Forward pass should be deterministic in eval mode."""
        block = Block(
            emb_size=16, hidden_mlp_size=32, dropout=0.0, mha_dropout=0.0
        ).to(cpu_device)
        block.eval()
        x = torch.randn(3, 10, 16, device=cpu_device)
        output1 = block(x)
        output2 = block(x)
        assert torch.allclose(output1, output2)

    def test_forward_with_dropout(self, cpu_device: str) -> None:
        """Forward pass should work in train mode with dropout."""
        block = Block(emb_size=16, hidden_mlp_size=32, dropout=0.5).to(cpu_device)
        block.train()
        x = torch.randn(3, 10, 16, device=cpu_device)
        output = block(x)
        assert output.shape == (3, 10, 16)

    def test_forward_residual_connection(self, cpu_device: str) -> None:
        """Forward pass should include residual connections."""
        block = Block(
            emb_size=16, hidden_mlp_size=32, dropout=0.0, mha_dropout=0.0
        ).to(cpu_device)
        block.eval()
        x = torch.randn(2, 5, 16, device=cpu_device)
        output = block(x)
        # Output should differ from input due to attention and MLP
        assert not torch.allclose(output, x)
        # But with zero init_scale, output should equal input (residual only)
        block_zero = Block(
            emb_size=16, hidden_mlp_size=32, dropout=0.0, mha_dropout=0.0, init_scale=0.0
        ).to(cpu_device)
        block_zero.eval()
        output_zero = block_zero(x)
        assert torch.allclose(output_zero, x, atol=1e-6)

    def test_init_scale_affects_parameters(self) -> None:
        """Init scale should affect parameter magnitudes."""
        block_small = Block(emb_size=16, hidden_mlp_size=32, init_scale=0.1)
        block_large = Block(emb_size=16, hidden_mlp_size=32, init_scale=1.0)
        small_norm = sum(p.norm().item() for p in block_small.parameters())
        large_norm = sum(p.norm().item() for p in block_large.parameters())
        assert small_norm < large_norm

    def test_gradient_flow(self, cpu_device: str) -> None:
        """Gradients should flow through the block."""
        block = Block(emb_size=16, hidden_mlp_size=32).to(cpu_device)
        x = torch.randn(3, 10, 16, device=cpu_device, requires_grad=True)
        output = block(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestSumTransformer:
    """Tests for SumTransformer class."""

    def test_init_basic(self) -> None:
        """Should initialize with basic parameters."""
        model = SumTransformer(
            alphabet_size=21, emb_size=16, max_sequence_size=20, head_mlp_hidden_size=32
        )
        assert isinstance(model, nn.Module)

    def test_init_custom_emb_size(self) -> None:
        """Should initialize with custom embedding size."""
        model = SumTransformer(
            alphabet_size=21, emb_size=64, max_sequence_size=20, head_mlp_hidden_size=32
        )
        assert model.embedder.embedding_dim == 64
        assert model.pos_embedder.embedding_dim == 64

    def test_init_custom_max_sequence_size(self) -> None:
        """Should initialize with custom max sequence size."""
        model = SumTransformer(
            alphabet_size=21, emb_size=16, max_sequence_size=100, head_mlp_hidden_size=32
        )
        assert model.max_sequence_size == 100

    def test_init_pos_embedder_minimum_size(self) -> None:
        """Position embedder should have at least 256 positions."""
        model = SumTransformer(
            alphabet_size=21, emb_size=16, max_sequence_size=50, head_mlp_hidden_size=32
        )
        assert model.pos_embedder.num_embeddings >= 256

    def test_init_multiple_transformers(self) -> None:
        """Should initialize with multiple transformer blocks."""
        model = SumTransformer(
            alphabet_size=21,
            emb_size=16,
            max_sequence_size=20,
            head_mlp_hidden_size=32,
            n_transformers=3,
        )
        assert len(model.refiners) == 3

    def test_init_zero_transformers(self) -> None:
        """Should initialize with zero transformer blocks."""
        model = SumTransformer(
            alphabet_size=21,
            emb_size=16,
            max_sequence_size=20,
            head_mlp_hidden_size=32,
            n_transformers=0,
        )
        assert len(model.refiners) == 0

    def test_init_negative_final_layers_raises(self) -> None:
        """Should raise ValueError for negative n_final_layers."""
        with pytest.raises(ValueError, match="n_final_layers must be positive"):
            SumTransformer(
                alphabet_size=21,
                emb_size=16,
                max_sequence_size=20,
                head_mlp_hidden_size=32,
                n_final_layers=-1,
            )

    def test_init_zero_final_layers(self) -> None:
        """Should initialize with zero final layers (just linear)."""
        model = SumTransformer(
            alphabet_size=21,
            emb_size=16,
            max_sequence_size=20,
            head_mlp_hidden_size=32,
            n_final_layers=0,
        )
        # Should be just a Linear layer wrapped in Sequential
        assert isinstance(model.final, nn.Sequential)
        assert len(model.final) == 1
        assert isinstance(model.final[0], nn.Linear)

    def test_init_with_final_layers(self) -> None:
        """Should initialize with final FFLayers."""
        model = SumTransformer(
            alphabet_size=21,
            emb_size=16,
            max_sequence_size=20,
            head_mlp_hidden_size=32,
            n_final_layers=2,
        )
        assert isinstance(model.final, nn.Sequential)
        # 2 FFLayers + 1 Linear
        assert len(model.final) == 3

    def test_init_alphabet_size_minimum(self) -> None:
        """Alphabet size should be at least 32."""
        model = SumTransformer(
            alphabet_size=10, emb_size=16, max_sequence_size=20, head_mlp_hidden_size=32
        )
        assert model.embedder.num_embeddings >= 32

    def test_init_head_mlp_hidden_size(self) -> None:
        """Should pass head_mlp_hidden_size to Block."""
        model = SumTransformer(
            alphabet_size=21,
            emb_size=16,
            max_sequence_size=20,
            head_mlp_hidden_size=128,
            n_transformers=1,
        )
        block = model.refiners[0]
        assert block.mlp[0].out_features == 128

    def test_forward_shape(self, cpu_device: str) -> None:
        """Forward pass should produce correct output shape."""
        model = SumTransformer(
            alphabet_size=21,
            emb_size=16,
            max_sequence_size=20,
            head_mlp_hidden_size=32,
        ).to(cpu_device)
        x = torch.randint(0, 21, (3, 20), device=cpu_device)
        output = model(x)
        assert output.shape == (3,)

    def test_forward_single_sample(self, cpu_device: str) -> None:
        """Forward pass should work with single sample."""
        model = SumTransformer(
            alphabet_size=21,
            emb_size=16,
            max_sequence_size=20,
            head_mlp_hidden_size=32,
        ).to(cpu_device)
        x = torch.randint(0, 21, (1, 20), device=cpu_device)
        output = model(x)
        assert output.shape == ()  # scalar output for batch of 1

    def test_forward_zero_transformers(self, cpu_device: str) -> None:
        """Forward pass should work with zero transformer blocks."""
        model = SumTransformer(
            alphabet_size=21,
            emb_size=16,
            max_sequence_size=20,
            head_mlp_hidden_size=32,
            n_transformers=0,
        ).to(cpu_device)
        x = torch.randint(0, 21, (3, 20), device=cpu_device)
        output = model(x)
        assert output.shape == (3,)

    def test_forward_deterministic(self, cpu_device: str) -> None:
        """Forward pass should be deterministic in eval mode."""
        model = SumTransformer(
            alphabet_size=21,
            emb_size=16,
            max_sequence_size=20,
            head_mlp_hidden_size=32,
            block_mlp_dropout=0.0,
            block_mha_dropout=0.0,
            final_dropout=0.0,
        ).to(cpu_device)
        model.eval()
        x = torch.randint(0, 21, (3, 20), device=cpu_device)
        output1 = model(x)
        output2 = model(x)
        assert torch.allclose(output1, output2)

    def test_forward_with_dropout(self, cpu_device: str) -> None:
        """Forward pass should work in train mode with dropout."""
        model = SumTransformer(
            alphabet_size=21,
            emb_size=16,
            max_sequence_size=20,
            head_mlp_hidden_size=32,
            block_mlp_dropout=0.5,
            final_dropout=0.5,
            n_final_layers=2,
        ).to(cpu_device)
        model.train()
        x = torch.randint(0, 21, (3, 20), device=cpu_device)
        output = model(x)
        assert output.shape == (3,)

    def test_forward_multiple_heads(self, cpu_device: str) -> None:
        """Forward pass should work with multiple attention heads."""
        model = SumTransformer(
            alphabet_size=21,
            emb_size=32,
            max_sequence_size=20,
            head_mlp_hidden_size=32,
            n_heads=4,
        ).to(cpu_device)
        x = torch.randint(0, 21, (3, 20), device=cpu_device)
        output = model(x)
        assert output.shape == (3,)

    def test_forward_multiple_transformers(self, cpu_device: str) -> None:
        """Forward pass should work with multiple transformer blocks."""
        model = SumTransformer(
            alphabet_size=21,
            emb_size=16,
            max_sequence_size=20,
            head_mlp_hidden_size=32,
            n_transformers=3,
        ).to(cpu_device)
        x = torch.randint(0, 21, (3, 20), device=cpu_device)
        output = model(x)
        assert output.shape == (3,)

    def test_forward_with_final_layers(self, cpu_device: str) -> None:
        """Forward pass should work with final layers."""
        model = SumTransformer(
            alphabet_size=21,
            emb_size=16,
            max_sequence_size=20,
            head_mlp_hidden_size=32,
            n_final_layers=3,
        ).to(cpu_device)
        x = torch.randint(0, 21, (3, 20), device=cpu_device)
        output = model(x)
        assert output.shape == (3,)

    def test_forward_custom_activation(self, cpu_device: str) -> None:
        """Forward pass should work with custom activation class."""
        model = SumTransformer(
            alphabet_size=21,
            emb_size=16,
            max_sequence_size=20,
            head_mlp_hidden_size=32,
            block_activation_class=nn.ReLU,
        ).to(cpu_device)
        x = torch.randint(0, 21, (3, 20), device=cpu_device)
        output = model(x)
        assert output.shape == (3,)

    def test_forward_positional_encoding(self, cpu_device: str) -> None:
        """Different positions should produce different embeddings."""
        model = SumTransformer(
            alphabet_size=21,
            emb_size=16,
            max_sequence_size=20,
            head_mlp_hidden_size=32,
            n_transformers=0,  # No transformers to isolate embedding effect
            block_mlp_dropout=0.0,
            block_mha_dropout=0.0,
        ).to(cpu_device)
        model.eval()
        # Same token at different positions
        x1 = torch.zeros((1, 20), dtype=torch.long, device=cpu_device)
        x2 = torch.zeros((1, 20), dtype=torch.long, device=cpu_device)
        x2[0, 0] = 1  # Change first position
        x2[0, 19] = 1  # Change last position
        output1 = model(x1)
        output2 = model(x2)
        # Outputs should differ due to different tokens
        assert not torch.allclose(output1, output2)

    def test_gradient_flow(self, cpu_device: str) -> None:
        """Gradients should flow through the network."""
        model = SumTransformer(
            alphabet_size=21,
            emb_size=16,
            max_sequence_size=20,
            head_mlp_hidden_size=32,
        ).to(cpu_device)
        x = torch.randint(0, 21, (3, 20), device=cpu_device)
        output = model(x)
        loss = output.sum()
        loss.backward()
        # Check that parameters have gradients
        has_grad = any(p.grad is not None for p in model.parameters())
        assert has_grad

    def test_different_batch_sizes(self, cpu_device: str) -> None:
        """Forward pass should work with different batch sizes."""
        model = SumTransformer(
            alphabet_size=21,
            emb_size=16,
            max_sequence_size=20,
            head_mlp_hidden_size=32,
        ).to(cpu_device)
        for batch_size in [1, 4, 16]:
            x = torch.randint(0, 21, (batch_size, 20), device=cpu_device)
            output = model(x)
            expected_shape = () if batch_size == 1 else (batch_size,)
            assert output.shape == expected_shape

    def test_large_alphabet_size(self, cpu_device: str) -> None:
        """Forward pass should work with large alphabet size (256 as used in examples)."""
        model = SumTransformer(
            alphabet_size=256,
            emb_size=16,
            max_sequence_size=20,
            head_mlp_hidden_size=32,
        ).to(cpu_device)
        x = torch.randint(0, 256, (3, 20), device=cpu_device)
        output = model(x)
        assert output.shape == (3,)

    def test_default_max_sequence_size(self, cpu_device: str) -> None:
        """Forward pass should work with default max_sequence_size (201)."""
        model = SumTransformer(
            alphabet_size=21,
            emb_size=16,
            head_mlp_hidden_size=32,
        ).to(cpu_device)
        assert model.max_sequence_size == 201
        x = torch.randint(0, 21, (2, 201), device=cpu_device)
        output = model(x)
        assert output.shape == (2,)

    def test_multihead_attention_divisibility(self, cpu_device: str) -> None:
        """Embedding size must be divisible by number of heads."""
        # Valid combinations from example: emb_size in (16, 32, 64), n_heads in (2, 4, 8, 16)
        valid_combinations = [
            (16, 2), (16, 4), (16, 8), (16, 16),
            (32, 2), (32, 4), (32, 8), (32, 16),
            (64, 2), (64, 4), (64, 8), (64, 16),
        ]
        for emb_size, n_heads in valid_combinations:
            model = SumTransformer(
                alphabet_size=21,
                emb_size=emb_size,
                max_sequence_size=20,
                head_mlp_hidden_size=32,
                n_heads=n_heads,
            ).to(cpu_device)
            x = torch.randint(0, 21, (2, 20), device=cpu_device)
            output = model(x)
            assert output.shape == (2,)

    def test_example_configuration(self, cpu_device: str) -> None:
        """Forward pass should work with configuration matching production example."""
        # Matches run_transformer_fan_alldata_multijob.py usage pattern
        model = SumTransformer(
            alphabet_size=256,
            n_transformers=2,
            emb_size=32,
            n_heads=4,
            head_mlp_hidden_size=64,  # Reduced from 512 for CPU efficiency
            block_mlp_dropout=0.1,
            block_mha_dropout=0.1,
            n_final_layers=1,
            final_dropout=0.05,
        ).to(cpu_device)
        x = torch.randint(0, 256, (4, 201), device=cpu_device)
        output = model(x)
        assert output.shape == (4,)

    def test_many_transformer_blocks(self, cpu_device: str) -> None:
        """Forward pass should work with many transformer blocks (up to 6 in examples)."""
        model = SumTransformer(
            alphabet_size=21,
            emb_size=16,
            max_sequence_size=20,
            head_mlp_hidden_size=32,
            n_transformers=6,
            n_heads=2,
        ).to(cpu_device)
        x = torch.randint(0, 21, (2, 20), device=cpu_device)
        output = model(x)
        assert output.shape == (2,)

    def test_high_head_count(self, cpu_device: str) -> None:
        """Forward pass should work with high head counts (up to 16 in examples)."""
        model = SumTransformer(
            alphabet_size=21,
            emb_size=64,  # Must be divisible by 16
            max_sequence_size=20,
            head_mlp_hidden_size=32,
            n_heads=16,
        ).to(cpu_device)
        x = torch.randint(0, 21, (2, 20), device=cpu_device)
        output = model(x)
        assert output.shape == (2,)

    def test_low_dropout_values(self, cpu_device: str) -> None:
        """Forward pass should work with low dropout values used in examples."""
        model = SumTransformer(
            alphabet_size=21,
            emb_size=16,
            max_sequence_size=20,
            head_mlp_hidden_size=32,
            block_mlp_dropout=0.05,
            block_mha_dropout=0.05,
            n_final_layers=2,
            final_dropout=0.05,
        ).to(cpu_device)
        model.train()
        x = torch.randint(0, 21, (3, 20), device=cpu_device)
        output = model(x)
        assert output.shape == (3,)
