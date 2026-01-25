"""Integration tests for the transformer training pipeline.

These tests verify that transformer components work together correctly.
They use small models and synthetic data to keep execution fast.
"""

import pytest
import torch
import torch.nn as nn

from mavenets.network.transformer import Block, SumTransformer  # type: ignore[import-not-found]


class TestBlockIntegration:
    """Test Block component integration."""

    def test_stacked_blocks_forward(self, cpu_device: str) -> None:
        """Test that multiple blocks can be stacked and run forward pass."""
        blocks = nn.ModuleList([
            Block(emb_size=16, hidden_mlp_size=32, num_heads=2)
            for _ in range(3)
        ]).to(cpu_device)

        x = torch.randn(4, 10, 16, device=cpu_device)
        out = x
        for block in blocks:
            out = block(out)

        assert out.shape == x.shape
        assert not torch.isnan(out).any()

    def test_stacked_blocks_gradient_flow(self, cpu_device: str) -> None:
        """Test that gradients flow through stacked blocks."""
        blocks = nn.ModuleList([
            Block(emb_size=16, hidden_mlp_size=32, num_heads=2)
            for _ in range(3)
        ]).to(cpu_device)

        x = torch.randn(4, 10, 16, device=cpu_device, requires_grad=True)
        out = x
        for block in blocks:
            out = block(out)

        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        # Check all block parameters have gradients
        for block in blocks:
            for param in block.parameters():
                assert param.grad is not None


class TestTransformerEncodingPipeline:
    """Test the pipeline from sequence encoding to transformer forward pass."""

    def test_integer_encoding_forward(self, cpu_device: str) -> None:
        """Test encoding integer sequences and passing through transformer."""
        model = SumTransformer(
            alphabet_size=21,
            emb_size=16,
            max_sequence_size=10,
            head_mlp_hidden_size=32,
            n_transformers=2,
            n_heads=2,
        ).to(cpu_device)

        # Simulate integer-encoded amino acid sequences
        sequences = torch.randint(0, 21, (4, 10), device=cpu_device)

        output = model(sequences)
        assert output.shape == (4,)
        assert not torch.isnan(output).any()

    def test_gradient_flow_through_transformer(self, cpu_device: str) -> None:
        """Test that gradients flow through the transformer pipeline."""
        model = SumTransformer(
            alphabet_size=21,
            emb_size=16,
            max_sequence_size=10,
            head_mlp_hidden_size=32,
            n_transformers=2,
            n_heads=2,
        ).to(cpu_device)

        sequences = torch.randint(0, 21, (4, 10), device=cpu_device)
        output = model(sequences)

        target = torch.randn(4, device=cpu_device)
        loss = nn.functional.mse_loss(output, target)
        loss.backward()

        # Check all parameters have gradients
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"


@pytest.mark.slow
class TestTransformerTrainingLoop:
    """Test transformer training loops."""

    def test_transformer_training_reduces_loss(self, cpu_device: str) -> None:
        """Test that training actually reduces loss."""
        model = SumTransformer(
            alphabet_size=21,
            emb_size=16,
            max_sequence_size=10,
            head_mlp_hidden_size=32,
            n_transformers=1,
            n_heads=2,
            block_mlp_dropout=0.0,
            block_mha_dropout=0.0,
        ).to(cpu_device)

        # Create synthetic data
        torch.manual_seed(42)
        X = torch.randint(0, 21, (50, 10), device=cpu_device)
        # Target: sum of first few positions (learnable pattern)
        y = X[:, :3].float().mean(dim=-1)

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

    def test_transformer_overfits_small_dataset(self, cpu_device: str) -> None:
        """Test that transformer can overfit a small dataset (sanity check)."""
        model = SumTransformer(
            alphabet_size=21,
            emb_size=16,
            max_sequence_size=8,
            head_mlp_hidden_size=32,
            n_transformers=2,
            n_heads=2,
            block_mlp_dropout=0.0,
            block_mha_dropout=0.0,
        ).to(cpu_device)

        # Very small dataset
        torch.manual_seed(42)
        X = torch.randint(0, 21, (5, 8), device=cpu_device)
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

        # Should achieve low loss on training data
        model.eval()
        with torch.no_grad():
            final_loss = criterion(model(X), y).item()

        assert final_loss < 0.1, f"Model failed to overfit small dataset: loss={final_loss:.4f}"

    def test_transformer_with_final_layers_trains(self, cpu_device: str) -> None:
        """Test that transformer with final layers can train."""
        model = SumTransformer(
            alphabet_size=21,
            emb_size=16,
            max_sequence_size=10,
            head_mlp_hidden_size=32,
            n_transformers=1,
            n_heads=2,
            n_final_layers=2,
            final_dropout=0.0,
            block_mlp_dropout=0.0,
            block_mha_dropout=0.0,
        ).to(cpu_device)

        torch.manual_seed(42)
        X = torch.randint(0, 21, (20, 10), device=cpu_device)
        y = torch.randn(20, device=cpu_device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        model.eval()
        with torch.no_grad():
            initial_loss = criterion(model(X), y).item()

        model.train()
        for _ in range(100):
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            final_loss = criterion(model(X), y).item()

        assert final_loss < initial_loss, (
            f"Training did not reduce loss: {initial_loss:.4f} -> {final_loss:.4f}"
        )


@pytest.mark.slow
class TestTransformerWithOptimizers:
    """Test transformer with different optimizer configurations."""

    def test_adamw_with_weight_decay(self, cpu_device: str) -> None:
        """Test training with AdamW and weight decay (as used in examples)."""
        model = SumTransformer(
            alphabet_size=21,
            emb_size=16,
            max_sequence_size=10,
            head_mlp_hidden_size=32,
            n_transformers=1,
            n_heads=2,
            block_mlp_dropout=0.0,
            block_mha_dropout=0.0,
        ).to(cpu_device)

        torch.manual_seed(42)
        X = torch.randint(0, 21, (30, 10), device=cpu_device)
        y = torch.randn(30, device=cpu_device)

        # AdamW with weight decay as used in examples
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=3e-4,
            weight_decay=0.005,
        )
        criterion = nn.MSELoss()

        model.eval()
        with torch.no_grad():
            initial_loss = criterion(model(X), y).item()

        model.train()
        for _ in range(200):
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            final_loss = criterion(model(X), y).item()

        assert final_loss < initial_loss

    def test_gradient_clipping(self, cpu_device: str) -> None:
        """Test that gradient clipping works with transformer."""
        model = SumTransformer(
            alphabet_size=21,
            emb_size=16,
            max_sequence_size=10,
            head_mlp_hidden_size=32,
            n_transformers=1,
            n_heads=2,
            block_mlp_dropout=0.0,
            block_mha_dropout=0.0,
        ).to(cpu_device)

        torch.manual_seed(42)
        X = torch.randint(0, 21, (10, 10), device=cpu_device)
        y = torch.randn(10, device=cpu_device) * 100  # Large targets to create large gradients

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()

        # Clip gradients (as done in examples with grad_clip=300)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=300)  # type: ignore[attr-defined]

        # Check gradients are clipped
        total_norm = torch.sqrt(
            torch.stack([p.grad.norm() ** 2 for p in model.parameters() if p.grad is not None]).sum()
        )
        assert total_norm.item() <= 300.01  # Small tolerance for floating point

        optimizer.step()


@pytest.mark.integration
class TestTransformerBatchProcessing:
    """Test transformer batch processing behavior."""

    def test_variable_batch_sizes(self, cpu_device: str) -> None:
        """Test that transformer handles variable batch sizes correctly."""
        model = SumTransformer(
            alphabet_size=21,
            emb_size=16,
            max_sequence_size=10,
            head_mlp_hidden_size=32,
            n_transformers=1,
            n_heads=2,
            block_mlp_dropout=0.0,
            block_mha_dropout=0.0,
        ).to(cpu_device)
        model.eval()

        # Test different batch sizes
        for batch_size in [1, 4, 16, 32]:
            X = torch.randint(0, 21, (batch_size, 10), device=cpu_device)
            output = model(X)
            expected_shape = () if batch_size == 1 else (batch_size,)
            assert output.shape == expected_shape

    def test_train_eval_mode_difference(self, cpu_device: str) -> None:
        """Test that train and eval modes produce different results with dropout."""
        model = SumTransformer(
            alphabet_size=21,
            emb_size=16,
            max_sequence_size=10,
            head_mlp_hidden_size=32,
            n_transformers=1,
            n_heads=2,
            block_mlp_dropout=0.5,
            block_mha_dropout=0.5,
        ).to(cpu_device)

        torch.manual_seed(42)
        X = torch.randint(0, 21, (4, 10), device=cpu_device)

        # Eval mode should be deterministic
        model.eval()
        out_eval1 = model(X)
        out_eval2 = model(X)
        assert torch.allclose(out_eval1, out_eval2)

        # Train mode with dropout should vary (usually)
        model.train()
        outputs = [model(X) for _ in range(5)]
        # At least some outputs should differ due to dropout
        all_same = all(torch.allclose(outputs[0], o) for o in outputs[1:])
        assert not all_same, "Dropout should cause variation in train mode"
