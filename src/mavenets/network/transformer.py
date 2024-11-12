"""Provides tools for creating transformer models."""

from typing import Tuple, Union, Callable
from torch import nn
import torch
from .base import FFLayer


class SumTransformer(nn.Module):
    """Transformer prediction model with summed contributions.

    Input is treated as sequence of tokens, with each token corresponding
    to an amino acid. Positional and amino-acid-type encodings are summed
    and then passed through transformer blocks for for adjustment. After
    this adjustment, a shared linear layer is applied to all tokens
    to predict a scalar energy contribution, which is then summed over
    the entire sequence.

    This roughly corresponds to the following diagram:

                     <sequence input>
                            |
      <position encoding>---+---<amino encoding>
                 |                     |
                 +--------[sum]--------+
                            |
                        [Block()] * N
                            |
               [ [token 1], ... [token n] ]
                     |      |      |
                 [ final ]...   [ final ]
                     |      |      |
               [    E_1,   ...    E_n    ]
                     |      |      |
                     +----<sum>----+
                            |
                         <output>

    where [final] is either a single shared linear transformation predicting a scalar,
    or a series or of FFN networks with residual connections and dropout.

    Input is expected to be of the form (batch_size, n_token, n_feature). In the case of
    amino acids, the feature is probably best a integer corresponding to the amino acid
    identity. All such integers should be non-negative.
    """

    def __init__(
        self,
        alphabet_size: int,
        emb_size: int = 32,
        max_sequence_size: int = 201,
        n_transformers: int = 1,
        n_heads: int = 1,
        head_mlp_hidden_size: int = 512,
        block_mlp_dropout: float = 0.2,
        block_mha_dropout: float = 0.2,
        block_activation_class: Callable[[], nn.Module] = nn.SiLU,
        n_final_layers: int = 0,
        final_dropout: float = 0,
    ) -> None:
        """Initialize layers and data.

        Arguments:
        ---------
        alphabet_size:
            Maximum integer found in the feature column of the input.
        emb_size:
            Dimensionality of embeddings used.
        max_sequence_size:
            Largest input sequence (n_tokens_ considered. Here set to the SARS-COV-1
            considered sequence length.
        n_transformers:
            Number of transformer adjustments to apply.
        n_heads:
            Number of transformer heads. See Block.
        head_mlp_hidden_size:
            Size of hidden layer in block networks. See Block.
        block_mlp_dropout:
            Dropout in block networks. See Block.
        block_mha_dropout:
            Dropout in block attention. See Block.
        block_activation_class:
            Callable that returns activation functions for created networks; passed to
        n_final_layers:
            Number of final layers to used when mapping transformed embeddings
            to scalar contributions. If zero, a linear layer is used to map
            each embedding to a sum contribution. If positive, an FFN with
            residual connections refines the embedding before the final linear
            step. Transformation is _shared_ across all tokens.
        final_dropout:
            Dropout in final layers; only active if n_final_layers positive as it is
            applied to residual-based refinements.

        """
        # hack to round up common alphabet size to a power of two.
        super().__init__()
        alphabet_size = max(alphabet_size, 32)
        self.max_sequence_size = max_sequence_size
        self.embedder = nn.Embedding(
            num_embeddings=alphabet_size, embedding_dim=emb_size
        )
        self.pos_embedder = nn.Embedding(
            # use a power of two bounding size in the actual case
            num_embeddings=max(max_sequence_size, 256),
            embedding_dim=emb_size,
        )

        self.refiners = nn.ModuleList(
            [
                Block(
                    emb_size=emb_size,
                    hidden_mlp_size=head_mlp_hidden_size,
                    dropout=block_mlp_dropout,
                    mha_dropout=block_mha_dropout,
                    num_heads=n_heads,
                    activation_class=block_activation_class,
                )
                for x in range(n_transformers)
            ]
        )
        self.final_lnorm = nn.LayerNorm(emb_size)
        if n_final_layers >= 0:
            flayers = [
                FFLayer(
                    in_size=emb_size,
                    out_size=emb_size,
                    residual_connection=True,
                    pre_layer_norm=True,
                    dropout=final_dropout,
                )
                for _ in range(n_final_layers)
            ]
            self.final = nn.Sequential(*flayers, nn.Linear(emb_size, 1, bias=True))
        else:
            raise ValueError("n_final_layers must be positive.")

    def forward(
        self,
        encoded_seq: torch.Tensor,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Predict scalar for int-encoded data."""
        _pos = torch.arange(self.max_sequence_size, device=encoded_seq.device)
        _pos_emb = self.pos_embedder(_pos)
        emb = self.embedder(encoded_seq) + _pos_emb
        for tr in self.refiners:
            emb = tr(emb)
        contrib = self.final(self.final_lnorm(emb))

        # get uncalibrated prediction
        underlying = contrib.sum(dim=(-2,))

        return underlying.squeeze()


class Block(nn.Module):
    """Transformer with self attention.

    Evaluates full (non-causal) attention, processes via MLP, and adjusts input via
    residual connections.
    """

    def __init__(
        self,
        emb_size: int = 32,
        hidden_mlp_size: int = 512,
        dropout: float = 0.2,
        mha_dropout: float = 0.2,
        num_heads: int = 1,
        init_scale: float = 0.7,
        activation_class: Callable[[], nn.Module] = nn.SiLU,
    ) -> None:
        """Initialize layers and data.

        Input to forward have shape (batch_size, n_token, emb_size).

        Arguments:
        ---------
        emb_size:
            Size of last dimension in input.
        hidden_mlp_size:
            Size of hidden layer in network.
        dropout:
            Dropout in post-attention network.
        mha_dropout:
            Dropout in attention block (see nn.MultiHeadAttention).
        num_heads:
            Number of heads in multihead attention.
        init_scale:
            Used to scale initial values. May be useful to modify when many blocks are
            stacked.
        activation_class:
            Callable that returns an activation function (not an activation function
            itself, likely a class).

        """
        super().__init__()
        self.emb_size = emb_size
        self.mha = nn.MultiheadAttention(
            embed_dim=emb_size,
            num_heads=num_heads,
            dropout=mha_dropout,
            kdim=emb_size,
            vdim=emb_size,
            batch_first=True,
        )
        self.mlp = nn.Sequential(
            nn.Linear(emb_size, hidden_mlp_size),
            activation_class(),
            nn.Linear(hidden_mlp_size, emb_size),
            nn.Dropout(dropout),
        )
        self.pre_head_norm = nn.LayerNorm(emb_size)
        self.pre_mlp_norm = nn.LayerNorm(emb_size)

        with torch.no_grad():
            for p in self.parameters():
                p *= init_scale  # type: ignore

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """Transform batch of tokens.

        Maintains dimensionality.
        """
        normed = self.pre_head_norm(inp)
        attended, _ = self.mha(
            query=normed, key=normed, value=normed, need_weights=False
        )
        out = inp + attended
        out = out + self.mlp(self.pre_mlp_norm(out))
        return out
