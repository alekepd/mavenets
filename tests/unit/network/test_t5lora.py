"""Tests for T5LoRAModel."""

from unittest.mock import patch, MagicMock
from types import SimpleNamespace

import torch
from torch import nn, Tensor

from mavenets.network.t5lora import T5LoRAModel
from mavenets.network.tune import NullTuner
from mavenets.network.base import MLP
from mavenets.data.featurize.core import get_default_int_encoder


BATCH_SIZE = 4
SEQ_LEN = 10
EMBED_DIM = 32  # smaller than real 1024 for testing


def _make_mock_t5(embed_dim: int = EMBED_DIM):
    """Create a mock T5 encoder that returns random embeddings with gradients."""

    class FakeT5(nn.Module):
        """Minimal T5-like module that produces differentiable output."""

        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(1, embed_dim)

        def forward(
            self, input_ids: Tensor, attention_mask: Tensor
        ) -> SimpleNamespace:
            batch, seq = input_ids.shape
            # Create differentiable output via a learnable projection
            dummy = torch.ones(batch, seq, 1, device=input_ids.device)
            hidden = self.proj(dummy)
            return SimpleNamespace(last_hidden_state=hidden)

    return FakeT5()


def _make_mock_peft_model(base_model: nn.Module, *args: object) -> nn.Module:
    """Wrap a base model to simulate get_peft_model behavior.

    Adds fake lora_ parameters so that parameter-group filtering works.
    Accepts extra args to match get_peft_model(model, config) signature.
    """
    # Add a fake LoRA parameter attribute
    base_model.lora_dummy = nn.Parameter(torch.zeros(1))  # type: ignore[attr-defined]
    base_model.enable_input_require_grads = MagicMock()  # type: ignore[attr-defined]
    base_model.gradient_checkpointing_enable = MagicMock()  # type: ignore[attr-defined]
    return base_model


@patch("mavenets.network.t5lora.get_peft_model")
@patch("mavenets.network.t5lora.T5EncoderModel")
@patch("mavenets.network.t5lora.T5Tokenizer")
class TestT5LoRAModel:
    """Tests for T5LoRAModel with mocked T5 and PEFT."""

    def _build_model(
        self,
        mock_tokenizer_cls: MagicMock,
        mock_t5_cls: MagicMock,
        mock_get_peft: MagicMock,
        embed_dim: int = EMBED_DIM,
    ) -> T5LoRAModel:
        """Helper to construct a T5LoRAModel with mocks."""
        enc = get_default_int_encoder()

        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.side_effect = None
        mock_tokenizer.return_value = {
            "input_ids": [[1, 2, 3]] * BATCH_SIZE,
            "attention_mask": [[1, 1, 1]] * BATCH_SIZE,
        }
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        # Mock T5 model loading
        fake_t5 = _make_mock_t5(embed_dim)
        mock_t5_cls.from_pretrained.return_value = fake_t5

        # Mock PEFT wrapping — return the model with fake lora params
        mock_get_peft.side_effect = _make_mock_peft_model

        downstream = MLP(
            in_size=embed_dim, out_size=1, hidden_sizes=[16], post_squeeze=True
        )

        model = T5LoRAModel(
            downstream=downstream,
            lora_rank=4,
            integer_encoder=enc,
        )
        return model

    def test_output_shape(
        self,
        mock_tokenizer_cls: MagicMock,
        mock_t5_cls: MagicMock,
        mock_get_peft: MagicMock,
    ) -> None:
        """Forward pass produces correct output shape."""
        model = self._build_model(mock_tokenizer_cls, mock_t5_cls, mock_get_peft)
        enc = get_default_int_encoder()
        int_encoded = enc.batch_encode(["ACDEF"] * BATCH_SIZE)
        output = model(int_encoded)
        assert output.shape == (BATCH_SIZE,)

    def test_gradient_flows(
        self,
        mock_tokenizer_cls: MagicMock,
        mock_t5_cls: MagicMock,
        mock_get_peft: MagicMock,
    ) -> None:
        """Gradients flow through the model after backward pass."""
        model = self._build_model(mock_tokenizer_cls, mock_t5_cls, mock_get_peft)
        enc = get_default_int_encoder()
        int_encoded = enc.batch_encode(["ACDEF"] * BATCH_SIZE)
        output = model(int_encoded)
        loss = output.sum()
        loss.backward()

        # Check that at least some parameters have gradients
        has_grad = False
        for param in model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, "No parameter received a gradient"

    def test_downstream_params_trainable(
        self,
        mock_tokenizer_cls: MagicMock,
        mock_t5_cls: MagicMock,
        mock_get_peft: MagicMock,
    ) -> None:
        """Downstream MLP parameters are trainable."""
        model = self._build_model(mock_tokenizer_cls, mock_t5_cls, mock_get_peft)
        for param in model.downstream.parameters():
            assert param.requires_grad

    def test_with_nulltuner(
        self,
        mock_tokenizer_cls: MagicMock,
        mock_t5_cls: MagicMock,
        mock_get_peft: MagicMock,
    ) -> None:
        """T5LoRAModel works as base_model in NullTuner."""
        t5_lora = self._build_model(mock_tokenizer_cls, mock_t5_cls, mock_get_peft)
        model = NullTuner(t5_lora)
        enc = get_default_int_encoder()
        int_encoded = enc.batch_encode(["ACDEF"] * BATCH_SIZE)
        head_index = torch.zeros(BATCH_SIZE, dtype=torch.int64)

        output = model(int_encoded, head_index, return_raw=False)
        assert output.shape == (BATCH_SIZE,)

    def test_with_nulltuner_return_raw(
        self,
        mock_tokenizer_cls: MagicMock,
        mock_t5_cls: MagicMock,
        mock_get_peft: MagicMock,
    ) -> None:
        """NullTuner with return_raw=True returns both tuned and raw output."""
        t5_lora = self._build_model(mock_tokenizer_cls, mock_t5_cls, mock_get_peft)
        model = NullTuner(t5_lora)
        enc = get_default_int_encoder()
        int_encoded = enc.batch_encode(["ACDEF"] * BATCH_SIZE)
        head_index = torch.zeros(BATCH_SIZE, dtype=torch.int64)

        result = model(int_encoded, head_index, return_raw=True)
        assert isinstance(result, tuple)
        tuned, raw = result
        assert tuned.shape == (BATCH_SIZE,)
        assert raw.shape == (BATCH_SIZE,)

    def test_unsupported_pooling_raises(
        self,
        mock_tokenizer_cls: MagicMock,
        mock_t5_cls: MagicMock,
        mock_get_peft: MagicMock,
    ) -> None:
        """Unsupported pooling strategy raises ValueError."""
        enc = get_default_int_encoder()
        mock_tokenizer_cls.from_pretrained.return_value = MagicMock()
        mock_t5_cls.from_pretrained.return_value = _make_mock_t5()
        mock_get_peft.side_effect = _make_mock_peft_model

        downstream = MLP(in_size=EMBED_DIM, out_size=1, hidden_sizes=[16])
        try:
            T5LoRAModel(
                downstream=downstream,
                pooling="max",
                integer_encoder=enc,
            )
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "max" in str(e)

    def test_peft_model_called(
        self,
        mock_tokenizer_cls: MagicMock,
        mock_t5_cls: MagicMock,
        mock_get_peft: MagicMock,
    ) -> None:
        """get_peft_model is called with the T5 model and a LoraConfig."""
        self._build_model(mock_tokenizer_cls, mock_t5_cls, mock_get_peft)
        mock_get_peft.assert_called_once()
        args = mock_get_peft.call_args
        # First positional arg should be the T5 model
        assert isinstance(args[0][0], nn.Module)

    def test_gradient_checkpointing(
        self,
        mock_tokenizer_cls: MagicMock,
        mock_t5_cls: MagicMock,
        mock_get_peft: MagicMock,
    ) -> None:
        """Gradient checkpointing methods are called when enabled."""
        enc = get_default_int_encoder()

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": [[1, 2, 3]],
            "attention_mask": [[1, 1, 1]],
        }
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        fake_t5 = _make_mock_t5()
        mock_t5_cls.from_pretrained.return_value = fake_t5
        mock_get_peft.side_effect = _make_mock_peft_model

        downstream = MLP(in_size=EMBED_DIM, out_size=1, hidden_sizes=[16])
        model = T5LoRAModel(
            downstream=downstream,
            gradient_checkpointing=True,
            integer_encoder=enc,
        )
        model.t5.enable_input_require_grads.assert_called_once()  # type: ignore[attr-defined]
        model.t5.gradient_checkpointing_enable.assert_called_once()  # type: ignore[attr-defined]
