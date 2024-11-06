"""Basic implementation of a transformer on Terra's dataset.

Some of this class re-implements standard classes for pedagogical purposes.
"""

from typing import (
    List,
    Optional,
    overload,
    Union,
    Literal,
    Callable,
    Tuple,
)
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class ELULinear(nn.Module):
    """Linear layer with positive-only linear transformation.

    Positivity is enforced via exp.  Bias is not constraint to be positive.
    """

    def __init__(
        self, in_size: int, out_size: int, bias: bool = True, leak: float = 0.01
    ) -> None:
        """Intialize weights."""
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        if leak < 0.0:
            raise ValueError("leak must be positive.")
        self.leak = leak
        self.weights = Parameter(torch.empty((in_size, out_size)))
        with torch.no_grad():
            torch.nn.init.uniform_(self.weights, -0.01, 0.01)
        if bias:
            self.bias: Optional[torch.Tensor] = Parameter(torch.zeros((out_size,)))
        else:
            self.bias = None

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """Filter weights and apply."""
        if self.bias is None:
            return inp @ (torch.nn.functional.elu(self.weights) + 1.0 - self.leak)
        else:
            return (
                inp @ (torch.nn.functional.elu(self.weights) + 1.0 - self.leak)
                + self.bias
            )


class BaseFFN(nn.Module):
    """Simple class for making feed forward networks."""

    def __init__(
        self,
        in_size: int,
        hidden_size: int,
        n_hidden: int,
        out_size: Optional[int] = None,
        activation_class: Callable[[], nn.Module] = nn.ReLU,
        residual_connection: bool = False,
        global_residual_connection: bool = False,
        dropout: float = 0.0,
        positive_linear: bool = False,
        scale: Optional[float] = None,
    ) -> None:
        """Create layers."""
        super().__init__()
        if hidden_size < 1:
            raise ValueError("Only a positive number of hidden layers are supported.")
        if global_residual_connection and out_size != in_size:
            raise ValueError(
                "Global residual connection requires in_size to be equal to out_size."
            )
        self.residual_connection = residual_connection
        self.global_residual_connection = global_residual_connection
        self.scale = scale

        self.positive_linear = positive_linear

        if self.positive_linear:
            linear_constructor: Callable[[int, int], nn.Module] = ELULinear
        else:
            linear_constructor = nn.Linear

        if out_size is None:
            out_size = in_size

        if dropout > 0.0:
            self.initial = nn.Sequential(
                linear_constructor(in_size, hidden_size),
                activation_class(),
                nn.Dropout(dropout),
            )
        else:
            self.initial = nn.Sequential(
                linear_constructor(in_size, hidden_size), activation_class()
            )
        self.final = linear_constructor(hidden_size, out_size)
        hidden_list: List[nn.Module] = []
        for _ in range(n_hidden - 1):
            hidden_list.append(linear_constructor(hidden_size, hidden_size))
            hidden_list.append(activation_class())
            if dropout > 0.0:
                hidden_list.append(nn.Dropout(p=dropout))

        self.hiddens = nn.ModuleList(hidden_list)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """Evaluate."""
        proc = self.initial(inp)
        if self.residual_connection:
            for layer in self.hiddens:
                proc = proc + layer(proc)
        else:
            for layer in self.hiddens:
                proc = layer(proc)
        out = self.final(proc)
        if self.global_residual_connection:
            out = inp + out
        if self.scale is not None:
            out = self.scale * out
        return out


class TrPredictor(nn.Module):
    """Transformer-based prediction model."""

    def __init__(
        self,
        alphabet_size: int,
        n_calibrators: int,
        emb_size: int = 32,
        block_size: int = 32,
        full_size: int = 201,
        n_transformers: int = 1,
        num_heads: int = 1,
        calibrator_width: int = 16,
        calibrator_n_hidden: int = 1,
        calibrator_dropout: float = 0.0,
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
        # self.calibrators = nn.ModuleList(
        #    [
        #        BaseFFN(
        #            in_size=1,
        #            out_size=1,
        #            hidden_size=calibrator_width,
        #            n_hidden=calibrator_n_hidden,
        #            residual_connection=False,
        #            global_residual_connection=False,
        #            dropout=calibrator_dropout,
        #            activation_class=nn.LeakyReLU,
        #            positive_linear=False,
        #            scale=1.0 / calibrator_width,
        #        )
        #        for _ in range(n_calibrators)
        #    ]
        # )

        self.calibrator_fanout = nn.Sequential(nn.Linear(1, calibrator_width), nn.ELU())
        # note that this is n different networks. can be recast
        # into a vector-valued final linear layer.
        self.calibrator_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Dropout(calibrator_dropout),
                    nn.Linear(calibrator_width, 1),
                )
                for _ in range(n_calibrators)
            ]
        )

    @overload
    def forward(
        self,
        encoded_seq: torch.Tensor,
        experiment_id: torch.Tensor,
        include_raw: Literal[True],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ...

    @overload
    def forward(
        self,
        encoded_seq: torch.Tensor,
        experiment_id: torch.Tensor,
        include_raw: Literal[False],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ...

    @overload
    def forward(
        self,
        encoded_seq: torch.Tensor,
        experiment_id: torch.Tensor,
        include_raw: Literal[False] = ...,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ...

    def forward(
        self,
        encoded_seq: torch.Tensor,
        experiment_id: torch.Tensor,
        include_raw: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Predict scalar for int-encoded data."""
        _pos = torch.arange(self.full_size).to(encoded_seq.device)
        _pos_emb = self.pos_embedder(_pos)
        emb = self.embedder(encoded_seq) + _pos_emb
        for tr in self.refiners:
            emb = tr(emb)
        contrib = self.penfinal(self.penfinal_lnorm(emb))

        # get uncalibrated prediction
        underlying = contrib.sum(dim=(-2,))

        # calibrate depending on experiment

        # multiple cal networks

        # contributions = []
        # for exp, cal in enumerate(self.calibrators):
        #    individual_corrected = cal(underlying)
        #    mask = (exp == experiment_id).view(-1, 1) + 0.01
        #    contr = mask * individual_corrected
        #    contributions.append(contr)
        # corrected = sum(contributions)

        # multiple cal heads

        contributions = []
        pre_calhead = self.calibrator_fanout(underlying)
        for exp, cal in enumerate(self.calibrator_heads):
            individual_corrected = cal(pre_calhead)
            mask = (exp == experiment_id).view(-1, 1)  # type: ignore
            contr = mask * individual_corrected
            contributions.append(contr)
        corrected = sum(contributions) + underlying
        if include_raw:
            return (corrected.squeeze(), underlying.squeeze())
        else:
            return corrected.squeeze()


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
