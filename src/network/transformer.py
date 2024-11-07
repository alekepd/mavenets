"""Provides tools for creating transformer models."""

from typing import Tuple, Union, Optional
from torch import nn
import torch


class TrPredictor(nn.Module):
    """Transformer-based prediction model."""

    def __init__(
        self,
        alphabet_size: int,
        emb_size: int = 32,
        block_size: int = 32,
        full_size: int = 201,
        n_transformers: int = 1,
        num_heads: int = 1,
    ) -> None:
        """Initialize layers and data."""
        # hack to round up common alphabet size to a power of two.
        super().__init__()
        alphabet_size = max(alphabet_size, 32)
        self.full_size = full_size
        self.embedder = nn.Embedding(
            num_embeddings=alphabet_size, embedding_dim=emb_size
        )
        self.pos_embedder = nn.Embedding(
            # use a power of two bounding size in the actual case
            num_embeddings=max(full_size, 256),
            embedding_dim=emb_size,
        )

        self.refiners = nn.ModuleList(
            [
                Block(input_emb_size=emb_size, emb_size=block_size, num_heads=num_heads)
                for x in range(n_transformers)
            ]
        )
        self.penfinal_lnorm = nn.LayerNorm(emb_size)
        self.penfinal = nn.Linear(emb_size, 1, bias=True)

    def forward(
        self,
        encoded_seq: torch.Tensor,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Predict scalar for int-encoded data."""
        _pos = torch.arange(self.full_size, device=encoded_seq.device)
        _pos_emb = self.pos_embedder(_pos)
        emb = self.embedder(encoded_seq) + _pos_emb
        for tr in self.refiners:
            emb = tr(emb)
        contrib = self.penfinal(self.penfinal_lnorm(emb))

        # get uncalibrated prediction
        underlying = contrib.sum(dim=(-2,))

        return underlying.squeeze()


class Block(nn.Module):
    """Transformer with self attention."""

    def __init__(
        self,
        emb_size: int = 32,
        input_emb_size: Optional[int] = None,
        hidden_mlp_size: int = 512,
        dropout: float = 0.2,
        mha_dropout: float = 0.2,
        num_heads: int = 1,
        init_scale: float = 0.7,
    ) -> None:
        """Initialize layers and data."""
        super().__init__()
        if input_emb_size is None:
            input_emb_size = emb_size
        self.emb_size = emb_size
        self.mha = nn.MultiheadAttention(
            embed_dim=emb_size,
            num_heads=num_heads,
            dropout=mha_dropout,
            kdim=input_emb_size,
            vdim=input_emb_size,
            batch_first=True,
        )
        self.mlp = nn.Sequential(
            nn.Linear(input_emb_size, hidden_mlp_size),
            nn.SiLU(),
            nn.Linear(hidden_mlp_size, input_emb_size),
            nn.Dropout(dropout),
        )
        self.pre_head_norm = nn.LayerNorm(input_emb_size)
        self.pre_mlp_norm = nn.LayerNorm(input_emb_size)

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
