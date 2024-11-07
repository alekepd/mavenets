"""Basic implementation of a transformer on Terra's dataset.

Some of this class re-implements standard classes for pedagogical purposes.
"""

from typing import (
    List,
    Optional,
    Callable,
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
