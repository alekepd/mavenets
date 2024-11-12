"""Provides classes for designing feed forward networks.

Most classes in this module are used to create more complex networks,
although may be used on their own.
"""

from __future__ import annotations
from typing import (
    List,
    Optional,
    Callable,
    Sequence,
)
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


def _identity(inp: torch.Tensor) -> torch.Tensor:
    return inp


class ELULinear(nn.Module):
    r"""Linear layer with positive-only linear transformation.

    For input `x`, the transformation applied is `x f(A) + b`, where `f`
    is a shifted softplus. By default, the softplus is shift to be almost
    always positive. Note that `b` is not constrained to be positive.

    This class allows one to create an affine transformation whose linear component
    may be characterized by a matrix with positive values.

    Note that pre-linear weights are initialized via a kaiming uniform distribution;
    however, this likely does not have merit due to the non-linear transform. Biases
    are initialized at 0.
    """

    def __init__(
        self, in_size: int, out_size: int, bias: bool = True, leak: float = 0.01
    ) -> None:
        """Initialize weights.

        Arguments:
        ---------
        in_size:
            Size of input; broadcasting is performed like a nn.Linear layer.
        out_size:
            Size of output; broadcasting is performed like a nn.Linear layer.
        bias:
            Whether to have a (trainable) offset to the linear transformation.
        leak:
            Amount by which to allow negative values in linear transform. Must be
            non-negative. 0 corresponds to no-negative values.

        """
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        if leak < 0.0:
            raise ValueError("leak must be positive.")
        self.leak = leak
        self.weights = Parameter(torch.empty((in_size, out_size)))
        with torch.no_grad():
            torch.nn.init.kaiming_uniform_(self.weights)
        if bias:
            self.bias: Optional[torch.Tensor] = Parameter(torch.zeros((out_size,)))
        else:
            self.bias = None

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """Transform weights and apply."""
        if self.bias is None:
            return inp @ (torch.nn.functional.elu(self.weights) + 1.0 - self.leak)
        else:
            return (
                inp @ (torch.nn.functional.elu(self.weights) + 1.0 - self.leak)
                + self.bias
            )


class FFLayer(nn.Module):
    """Single feed forward layer.

    Supports dropout, input layer-norm, and residual connections. Layer norm is called
    immediately on input before linear processing. Transformations are applied as:

                  <input>
                     |
                     +-------------+
                     |             |
               [ layer norm ]      |
                     |             |
                 [ linear ]        |
                     |             | (residual)
               [ activation ]      |
                     |             |
                [ dropout ]        |
                     |             |
                     +-------------+
                     |
                 <output>

    """

    def __init__(
        self,
        in_size: int,
        out_size: int,
        bias: bool = True,
        residual_connection: bool = False,
        activation_class: Callable[[], nn.Module] = nn.LeakyReLU,
        pre_layer_norm: bool = False,
        dropout: float = 0.0,
    ) -> None:
        """Store arguments and initialize layers.

        Arguments:
        ---------
        in_size:
            Size of input.
        out_size:
            Size of output.
        bias:
            Whether to include a bias term in the linear transformation.
        residual_connection:
            Whether to use a residual connection. See class description.
        activation_class:
            Callable that returns an activation function (not an activation function
            itself, likely a class).
        pre_layer_norm:
            Whether to apply layer normalization on input as a first step.
        dropout:
            Whether to use dropout either right before the residual connection,
            or if no such connection is used, as a terminal step.

        """
        super().__init__()
        self.affine = nn.Linear(in_size, out_size, bias=bias)
        self.activation = activation_class()
        if residual_connection and in_size != out_size:
            raise ValueError(
                "Cannot create a residual connection when in_size and out_size differ."
            )
        self.residual_connection = residual_connection
        self.dropout = nn.Dropout(dropout)
        if pre_layer_norm:
            self.pre_layer_norm: Callable[[torch.Tensor], torch.Tensor] = nn.LayerNorm(
                normalized_shape=in_size
            )
        else:
            self.pre_layer_norm = _identity

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """Apply layers and perform residual_connection if specified."""
        proc = self.pre_layer_norm(inp)
        proc = self.affine(proc)
        proc = self.activation(proc)
        proc = self.dropout(proc)
        if self.residual_connection:
            proc = inp + proc
        return proc


class BaseFFN(nn.Module):
    """Simple class for making feed forward networks.

    This module allows a variety of options to be applied for create
    modified multilayer perceptrons. Supported operations include
    residual connections, dropout, and ELULinear units.

    Note that the input is assumed be flat (in the sence of nn.Linear
    broadcasting). Networks are roughly of the form

                              <inp>
                                |
                            <hidden 0>
                                |
                              [...]
                                |
                            <hidden n>
                                |
                              <out>

    where a supplied activation function is used inbetween each layer (but not after
    the final layer).

    Two types of residual connections may be inserted. global_residual_connection
    creates a overall transformation of the form x -> f(x) + x; this is only possible
    if the size of input and output are the same. residual_connection instead
    applies a residual connection along each hidden block.

    If specified, dropout is applied after each activation layer.

    Note that this class does not support hidden layers of multiple distinct sizes.
    """

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
        """Validate arguments and create layers.

        Arguments:
        ---------
        in_size:
            Size of input. Multidimensional input is treated according to nn.Linear.
        hidden_size:
            Dimensionality of all hidden layers.
        n_hidden:
            Number of hidden layers. Must be at least 1.
        out_size:
            Size of output. Multidimensional input is treated according to nn.Linear. If
            set to None (the default), assumed to be the same as in_size.
        activation_class:
            Callable that returns an activation function (not an activation function
            itself, likely a class).
        residual_connection:
            Whether to apply a residual connection in each hidden layer. See class
            description.
        global_residual_connection:
            Whether to apply a residual connection directly connecting the input to
            the output of the network. See class description.
        dropout:
            Wheter to apply dropout after every activation layer.
        positive_linear:
            Whether to use ELULinear layers throughout the network.
        scale:
            Static amount (not tensor) to multiply the output of the network by.

        """
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

        # we store multiple layers and then apply them following stored rules
        # and activations in the forward pass

        # first create starting layer
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
        # create final layer
        self.final = linear_constructor(hidden_size, out_size)
        # create hidden layers. Note that the first layer already implicitly
        # defines a hidden layer, so we create n_hidden-1 additional transforms.
        hidden_list: List[nn.Module] = []
        for _ in range(n_hidden - 1):
            hidden_list.append(linear_constructor(hidden_size, hidden_size))
            hidden_list.append(activation_class())
            if dropout > 0.0:
                hidden_list.append(nn.Dropout(p=dropout))

        self.hiddens = nn.ModuleList(hidden_list)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """Evaluate stored transforms."""
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


class MLP(nn.Module):
    """Fully connected feed-forward network.

    Unlike BaseFFN, this class allows the size of each layer to be specified.
    Very few other options are supported.
    """

    def __init__(
        self,
        in_size: int,
        out_size: int,
        hidden_sizes: Sequence[int],
        activation_class: Callable[[], nn.Module] = nn.ReLU,
    ) -> None:
        """Create layers.

        Arguments:
        ---------
        in_size:
            Size of input. Multidimensional input is treated according to nn.Linear.
        out_size:
            Size of output. Multidimensional input is treated according to nn.Linear.
        hidden_sizes:
            Sizes of all hidden layers.
        activation_class:
            Callable that returns an activation function (not an activation function
            itself, likely a class).

        """
        super().__init__()
        transforms: List[nn.Module] = []
        current_size = in_size
        # each iteration of this loop creates a linear layer and activation
        # linking between two sizes.
        for next_size in hidden_sizes:
            transforms.append(nn.Linear(current_size, next_size))
            transforms.append(activation_class())
            current_size = next_size
        transforms.append(nn.Linear(current_size, out_size))
        self.network = nn.Sequential(*transforms)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """Evaluate layers."""
        return self.network(inp)

    @classmethod
    def triangle_network(
        cls,
        in_size: int,
        out_size: int,
        hidden_size: int,
        n_hidden: int,
        activation_class: Callable[[], nn.Module] = nn.ReLU,
    ) -> MLP:
        """Create a multilayer perceptron with hidden layers of decreasing size.

        Arguments:
        ---------
        in_size:
            Size of input. Multidimensional input is treated according to nn.Linear.
        out_size:
            Size of output. Multidimensional input is treated according to nn.Linear.
        hidden_size:
            Size of first hidden layer. Each subsequent hidden layer has half the size
            of the previous layer (as defined by //2). If some layers would have
            non-positive size, a ValueError is raised.
        n_hidden:
            Number of hidden layers. See hidden_size.
        activation_class:
            Callable that returns an activation function (not an activation function
            itself, likely a class).

        Returns:
        -------
        MLP with described layers.

        """
        hidden_layer_sizes = []
        for _ in range(n_hidden):
            hidden_layer_sizes.append(hidden_size)
            hidden_size = hidden_size // 2
        if any(x < 1 for x in hidden_layer_sizes):
            raise ValueError(
                "Proposed sizes include some with non-positive sizes: {}".format(
                    hidden_layer_sizes
                )
            )
        return cls(
            in_size=in_size,
            out_size=out_size,
            hidden_sizes=hidden_layer_sizes,
            activation_class=activation_class,
        )
