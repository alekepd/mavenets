"""Provides a T5 encoder with LoRA adapters for fine-tuning on regression tasks.

This module wraps a pretrained T5 encoder model with LoRA (Low-Rank Adaptation)
adapters from the HuggingFace PEFT library, followed by pooling and a downstream
network. The resulting module can be used as the ``base_model`` argument to any
MHTuner instance.
"""

from typing import Final, List, Optional

import torch
from torch import Tensor, nn

from transformers import T5EncoderModel, T5Tokenizer  # type: ignore
from peft import LoraConfig, TaskType, get_peft_model  # type: ignore

from ..data.featurize.core import IntEncoder, get_default_int_encoder


class T5LoRAModel(nn.Module):
    """T5 encoder with LoRA adapters followed by pooling and a downstream network.

    This module is designed to be used as the ``base_model`` argument to any
    :class:`~mavenets.network.tune.MHTuner` subclass. During the forward pass,
    integer-encoded protein sequences are decoded, tokenized, passed through the
    T5 encoder (with LoRA adapters), mean-pooled over residues, and then passed
    through the downstream network.

    Example:
    -------
    ```
    from mavenets.network import MLP, NullTuner
    from mavenets.network.t5lora import T5LoRAModel

    mlp = MLP(in_size=1024, out_size=1, hidden_sizes=[512], post_squeeze=True)
    t5_lora = T5LoRAModel(downstream=mlp, lora_rank=8)
    model = NullTuner(t5_lora).to("cuda")
    ```

    """

    T5_HUGGINGFACE_NAME: Final = "Rostlab/prot_t5_xl_half_uniref50-enc"
    T5_EMBED_DIM: Final = 1024

    def __init__(
        self,
        downstream: nn.Module,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[List[str]] = None,
        t5_model_name: str = "Rostlab/prot_t5_xl_half_uniref50-enc",
        pooling: str = "mean",
        gradient_checkpointing: bool = False,
        integer_encoder: Optional[IntEncoder] = None,
    ) -> None:
        """Initialize T5 with LoRA adapters and downstream network.

        Arguments:
        ---------
        downstream:
            Network applied after pooling T5 embeddings. Typically an MLP with
            ``in_size=1024``. Its output is the output of this module.
        lora_rank:
            Rank of the low-rank LoRA decomposition.
        lora_alpha:
            Scaling factor for LoRA. Higher values make LoRA updates larger
            relative to the pretrained weights.
        lora_dropout:
            Dropout probability applied to LoRA layers.
        lora_target_modules:
            List of T5 submodule names to apply LoRA to. Defaults to
            ``["q", "v"]`` (query and value projections in attention).
        t5_model_name:
            HuggingFace model identifier for the T5 encoder.
        pooling:
            Pooling strategy over residue embeddings. Currently only ``"mean"``
            is supported.
        gradient_checkpointing:
            If True, enable gradient checkpointing on the T5 encoder to reduce
            memory usage at the cost of recomputation during backward pass.
        integer_encoder:
            IntEncoder for decoding integer-encoded sequences to strings. If
            None, the default encoder is used.

        """
        super().__init__()

        if pooling != "mean":
            raise ValueError(
                f"Unsupported pooling strategy {pooling!r}. Only 'mean' is supported."
            )
        self._pooling = pooling

        if lora_target_modules is None:
            lora_target_modules = ["q", "v"]

        # Load T5 encoder and apply LoRA
        t5_base: T5EncoderModel = T5EncoderModel.from_pretrained(t5_model_name)  # type: ignore[no-any-return]

        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=lora_target_modules,
        )

        self.t5 = get_peft_model(t5_base, lora_config)  # type: ignore[no-any-return]

        if gradient_checkpointing:
            self.t5.enable_input_require_grads()
            self.t5.gradient_checkpointing_enable()  # type: ignore[reportUnknownMemberType]

        self.downstream = downstream

        # Non-parameter attributes for sequence handling
        if integer_encoder is None:
            integer_encoder = get_default_int_encoder()
        self._integer_encoder = integer_encoder
        self._tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(  # type: ignore[reportUnknownMemberType]
            t5_model_name, do_lower_case=False
        )

    def forward(self, int_encoded: Tensor) -> Tensor:
        """Encode sequences with T5+LoRA, pool, and apply downstream network.

        Arguments:
        ---------
        int_encoded:
            Integer-encoded sequences of shape ``(batch, seq_len)``. Each row is
            a protein sequence encoded using the IntEncoder alphabet.

        Returns:
        -------
        Output of the downstream network, typically shape ``(batch,)`` or
        ``(batch, out_size)``.

        """
        # Decode integers to string sequences
        str_sequences = self._integer_encoder.batch_decode(int_encoded)
        formatted = [" ".join(s) for s in str_sequences]

        # Tokenize
        ids = self._tokenizer(formatted, add_special_tokens=True)
        device = next(self.t5.parameters()).device
        input_ids = torch.tensor(ids["input_ids"], device=device)
        attention_mask = torch.tensor(ids["attention_mask"], device=device)

        # T5 forward with gradients (LoRA adapters are trainable)
        embedding_repr = self.t5(
            input_ids=input_ids, attention_mask=attention_mask
        )
        embeddings: Tensor = embedding_repr.last_hidden_state

        # Pool over residue dimension
        pooled = embeddings.mean(dim=1)

        # Downstream network
        return self.downstream(pooled)
