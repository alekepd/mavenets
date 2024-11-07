"""Provides tools to adjust the output of an existing callable."""

from typing import Callable, overload, Union, Tuple, Literal
from .base import BaseFFN
from torch import nn, Tensor

_model_T = Callable[[Tensor], Tensor]


class MHTuner(nn.Module):
    """Abstract class of a multihead tuner.

    Tuners take the output of an exist class and adjust it. Multiple adjustments
    are possible are selected based on a (likely integer) tensor supplied to
    the forward method.

    After creation, this Module may be directly used in place of the original model.

    This class provides the structure to return both the adjusted and raw
    input.  Child classes should inherent this class and override tune. They
    should _not_ override forward.

    """

    def __init__(self, base_model: _model_T) -> None:
        """Initialize module structure and store base model."""
        super().__init__()
        self.base_model = base_model

    @overload
    def forward(
        self, inp: Tensor, head_index: Tensor, return_raw: Literal[False]
    ) -> Tensor:
        ...

    @overload
    def forward(
        self, inp: Tensor, head_index: Tensor, return_raw: Literal[True]
    ) -> Union[Tensor, Tensor]:
        ...

    @overload
    def forward(
        self, inp: Tensor, head_index: Tensor, return_raw: Literal[False] = ...
    ) -> Tensor:
        ...

    def forward(
        self, inp: Tensor, head_index: Tensor, return_raw: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Apply underlying model and tune.

        Optionally returns both raw and tuned output.

        DO NOT OVERRIDE THIS METHOD. Instead, override tuner.

        Arguments:
        ---------
        inp:
            Passed as input to base_model.
        head_index:
            Passed along with model output to self.tuner.
        return_raw:
            If true, we return both the

        Returns:
        -------
        Either a Tensor or a 2-Tuple of Tensors, first element being tuned output
        and second element the raw output. See return_raw.

        """
        raw_output = self.base_model(inp)
        tuned_output = self.tune(raw_output, head_index)
        if return_raw:
            return tuned_output, raw_output
        else:
            return tuned_output

    def tune(self, signal: Tensor, head_index: Tensor) -> Tensor:
        """Tune input signal using one of several tuners.

        Child classes must implement this method. Do not mutate the arguments.

        Arguments:
        ---------
        signal:
            Input signal to tune.
        head_index:
            Determines how to calibate the data; likely an integer tensor determining
            which tuner head to use.

        Returns:
        -------
        A Tensor contained the tuned output. Note that this method should not return
        both the raw and tuned data, just the tuned data.

        """
        raise NotImplementedError(
            "Trying to use an undefined tune method. "
            "You are probably trying to use MHTuner directly, which is a child class. "
            "Inherit the class and override the tune method."
        )


class NullTuner(MHTuner):
    """Tuner that does nothing.

    Provided to allow non-tuned models to function in tuned pipelines.
    """

    def __init__(self, base_model: _model_T) -> None:
        """Call parent initializer."""
        super().__init__(base_model=base_model)

    def tune(self, signal: Tensor, head_index: Tensor) -> Tensor:  # noqa: ARG002
        """Return signal unaltered.

        Arguments:
        ---------
        signal:
            Input signal to return unaltered.
        head_index:
            Ignored.

        Returns:
        -------
        returns input signal (torch.Tensor).

        """
        return signal


class SharedFanTuner(MHTuner):
    """Applies a simple shared head-based linear network tuning.

    This class is built to apply to one-dimensional input (not including batch
    dimension).

    Input is fist "featurized" by applying a linear layer that increases the
    dimensionality (fans it out) and then a shared function. Second, a linear layer
    is applied to linearly combine these features into an output signal. The first
    step is shared among the different tuning heads, but the second is not.

    This class does not support negative numbers in the head_index argument.
    """

    def __init__(
        self,
        base_model: _model_T,
        n_heads: int,
        dropout: float = 0.0,
        fan_size: int = 16,
        fan_activation: Callable[[], nn.Module] = nn.ELU,
        residual_connection: bool = True,
    ) -> None:
        """Store options and initialize layers.

        Note that n_heads must be larger or the same as the largest integer
        found during runtime in the head_index argument (e.g., if the biggest
        later found integer is 8, n_heads must be 9).

        Arguments:
        ---------
        base_model:
            Callable to adjust the output of.
        n_heads:
            Number of tuning heads to create.
        dropout:
            Whether to use dropout right before final linear layer.
        fan_size:
            Size of fanned input data (not including batch dimension).
        fan_activation:
            No-argument callable (usually a class) returning function to apply to fanned
            input.
        residual_connection:
            Whether to apply adjustment in the form f(x) = correction + x. Typically
            a good idea for stability of gradient based training.

        """
        super().__init__(base_model)
        self.residual_connection = residual_connection
        self.fanout = nn.Sequential(nn.Linear(1, fan_size), fan_activation())
        # note that this is n different networks. can be recast
        # into a vector-valued final linear layer.
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(fan_size, 1),
                )
                for _ in range(n_heads)
            ]
        )

    def tune(self, signal: Tensor, head_index: Tensor) -> Tensor:
        """Tune input signal.

        Arguments:
        ---------
        signal:
            Input tensor of shape (n_batch,) or (n_batch,1) (one dimensional input). The
            returned calibrated data will be of the same size.
        head_index:
            Tensor of shape (n_batch,) of integers. Specifies the tuning head to apply
            to that sample.

        Returns:
        -------
        Tuned output (torch.tensor).

        """
        signal_shape = signal.shape
        fanned = self.fanout(signal.view(-1, 1))
        contributions = []
        for exp, cal in enumerate(self.heads):
            individual_corrected = cal(fanned)
            # create mask that as zeros
            # might need a view on head_index if we want to be shape robust.
            mask = (exp == head_index).view(-1, 1)  # type: ignore
            contr = mask * individual_corrected
            contributions.append(contr)
        if self.residual_connection:
            corrected = sum(contributions) + signal
        else:
            corrected = sum(contributions)
        # accounts for whether the input was shape (batch,) or (batch, 1)
        return corrected.view(signal_shape)


class FFNTuner(MHTuner):
    """Tunes input using multiple fully connected networks.

    This class does not support negative numbers in the head_index argument.
    """

    def __init__(
        self,
        base_model: _model_T,
        n_heads: int,
        hidden_size: int,
        n_hidden: int,
        positive_linear: bool = False,
        dropout: float = 0.0,
        activation_class: Callable[[], nn.Module] = nn.LeakyReLU,
    ) -> None:
        """Create tuning networks.

        Note that n_heads must be larger or the same as the largest integer
        found during runtime in the head_index argument (e.g., if the biggest
        later found integer is 8, n_heads must be 9).

        Arguments:
        ---------
        base_model:
            Callable to adjust the output of.
        n_heads:
            Number of tuning networks to create.
        hidden_size:
            Size of the hidden layers.
        n_hidden:
            Number of hidden layers. Must be at least one.
        positive_linear:
            Whether to use ELULinear transformations. This forces the scaling to be
            positive, but may be harder to train.
        dropout:
            Dropout value to use in underlying networks.
        activation_class:
            Callable that returns activation functions for created networks.

        """
        super().__init__(base_model)
        self.heads = nn.ModuleList(
            [
                BaseFFN(
                    in_size=1,
                    out_size=1,
                    hidden_size=hidden_size,
                    n_hidden=n_hidden,
                    residual_connection=False,
                    global_residual_connection=False,
                    dropout=dropout,
                    activation_class=activation_class,
                    positive_linear=positive_linear,
                    scale=1.0 / hidden_size,
                )
                for _ in range(n_heads)
            ]
        )

    def tune(self, signal: Tensor, head_index: Tensor) -> Tensor:
        """Tune input signal.

        Arguments:
        ---------
        signal:
            Input tensor of shape (n_batch,) or (n_batch,1) (one dimensional input). The
            returned calibrated data will be of the same size.
        head_index:
            Tensor of shape (n_batch,) of integers. Specifies the tuning head to apply
            to that sample.

        Returns:
        -------
        Tuned output (torch.tensor).

        """
        signal_shape = signal.shape
        contributions = []
        for exp, cal in enumerate(self.heads):
            individual_corrected = cal(signal.view(-1, 1))
            # create mask that as zeros
            # might need a view on head_index if we want to be shape robust.
            mask = (exp == head_index).view(-1, 1)  # type: ignore
            contr = mask * individual_corrected
            contributions.append(contr)
        corrected = sum(contributions)
        # accounts for whether the input was shape (batch,) or (batch, 1)
        return corrected.view(signal_shape)
