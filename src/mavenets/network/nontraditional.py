"""Defines non-traditional neural network models."""

from typing import Callable, List
import torch
from torch import nn


class _Ident(nn.Module):
    """Module that does nothing."""

    def __init__(self) -> None:
        pass

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        return inp


class LRMLP(nn.Module):
    """Linear model with input features augmented by MLP.

    Data is first processed by a network to produce a vector feature. This
    is then concatenated with the input features and mixed using a linear layer.

    This roughly corresponds to the following diagram.

             <sequence input>
                     |
                     +--------------+
                     |              |
                     |            [mlp]
                     |              |
                     +---[concat]---+
                             |
                         [linear]
                             |
                         <output>

    where [mlp] is a single layer mlp that processes the input sequence. The mlp
    output dimension is given by augment_channel_size and is limited to a simple
    perceptron with no hidden layer.

    """

    def __init__(
        self,
        in_size: int,
        out_size: int,
        augment_channel_size: int,
        activation_class: Callable[[], nn.Module] = nn.LeakyReLU,
        pre_flatten: bool = False,
        post_squeeze: bool = False,
    ) -> None:
        """Create layers.

        Arguments:
        ---------
        in_size:
            Size of input. Multidimensional input is treated according to nn.Linear.
        out_size:
            Size of output.
        augment_channel_size:
            Size of MLP output used to augment input features.
        activation_class:
            Callable that returns an activation function (not an activation function
            itself, likely a class).
        pre_flatten:
            If True, a nn.Flatten method (with default arguments) is applied before 
            calculations.
        post_squeeze:
            If True, squeeze is called on the network output. May cause dimensional 
            differences on size-1 batches.

        """
        super().__init__()
        if pre_flatten:
            self.preprocess: nn.Module = nn.Flatten()
        else:
            self.preprocess = _Ident()
        side_transforms: List[nn.Module] = []
        side_transforms.append(nn.Linear(in_size, augment_channel_size))
        side_transforms.append(activation_class())
        self.side_transform = nn.Sequential(*side_transforms)
        self.mix_transform = nn.Linear(in_size + augment_channel_size, out_size)
        self.post_squeeze = post_squeeze

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """Process input."""
        procced = self.preprocess(inp)
        # get mlp-added content
        side_signal = self.side_transform(procced)
        # combine with input data
        combined = torch.concatenate([procced, side_signal], dim=1)
        # linearly mix data
        mixed = self.mix_transform(combined)
        if self.post_squeeze:
            return torch.squeeze(mixed)
        else:
            return mixed
