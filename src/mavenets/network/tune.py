"""Provides tools to tune (adjust) the output of an existing callable.

Adjustments are performed by nn.Modules which take as existing callable as input during
definition. During invocation, the adjustment module calls the underlying source
callable and adjusts the output. If the underlying callable is a nn.Module, the
adjustment object functions as a composite module containing all tuning and source
parameters.

For example, in the case of
```
source_model = BaseFFN(...)
adjusted_model =  SharedFanTuner(source_model, ...)
```
`adjusted_model` now is `source_model` with a calibrator.

Importantly, all classes in this module are "multi-head" calibrations which change
the input signature of a model which maps single tensor to a single tensor to one
which maps a pair of tensors (as distinct arguments) to a single tensor. The first
item in the input pair is passed to the underlying function, while the second
is passed to the adjustement/tuning function. Typically, the second tensor is a
single int32/64 for each batch item indexing one of several "heads", each of which
contains a different tuning function.

The adjusted/tuned output is returned, and so can the original output of the
wrapped model.  This corresponds to the following flow of information for a
model wrapped in an adjustment/tuning Module:

                     <input-1>   <input-2>
                         |           |
               <underlying model>    |
                         |           |
            +------------+           |
            |            |           |
            |         <tuner>------- +
            |            |
            |            |
      <raw output>  <tuned output>

This behavior is defined in the abstract MHTuner class. This behavior is then
implemented in different child classes which define the tuning procedure.


"""

from typing import Callable, overload, Union, Tuple, Literal, TypeVar, Generic
from .base import BaseFFN
from torch import nn, Tensor, full, atleast_2d

_model_T = Callable[[Tensor], Tensor]
_T = TypeVar("_T")


class MHTuner(nn.Module, Generic[_T]):
    """Abstract class of a multihead tuner.

    Tuners take the output of an exist class and adjust it. Multiple adjustments
    are possible are selected based on a (likely integer) tensor supplied to
    the forward method.

    After creation, this Module may be directly used in place of the original callable.
    If the original callable is a nn.Module, the new Module parameters include that of
    the original nn.Module.

    This class provides the structure to return both the adjusted and raw
    input.  Child classes should inherent this class and override tune. They
    should _not_ override forward.

    """

    def __init__(self, base_model: Callable[[_T], Tensor]) -> None:
        """Initialize module structure and store base model."""
        super().__init__()
        self.base_model = base_model

    @overload
    def forward(
        self, inp: _T, head_index: Tensor, return_raw: Literal[False]
    ) -> Tensor:
        ...

    @overload
    def forward(
        self, inp: _T, head_index: Tensor, return_raw: Literal[True]
    ) -> Union[Tensor, Tensor]:
        ...

    @overload
    def forward(
        self, inp: _T, head_index: Tensor, return_raw: Literal[False] = ...
    ) -> Tensor:
        ...

    def forward(
        self, inp: _T, head_index: Tensor, return_raw: bool = False
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

    def create_singlehead_model(self, head_index: int) -> "HeadLock":
        """Create wrapped version of self that fixes the selected tuning head.

        This method will only work if the underlying module beneath the tuner
        accepts tensor input for the inp argument.

        Arguments:
        ---------
        head_index:
            Index determining which head to use.

        Returns:
        -------
        HeadLock instance that sets the model.

        """
        # technically not type safe since HeadLock only works if the
        # underlying model uses a tensor-based input.
        return HeadLock(self, head_index=head_index)  # type: ignore


class HeadLock(nn.Module):
    """Wrapper which fixed tuner head used for evaluation.

    Useful to transform a multi-head model to a single-head model.

    """

    def __init__(self, model: MHTuner[Tensor], head_index: int) -> None:
        """Store model and head index."""
        super().__init__()
        self.model = model
        self.head_index = head_index

    def forward(self, inp: Tensor) -> Tensor:
        """Evaluate model with locked head index value."""
        derived_heads = full(
            size=inp.shape[0:1], fill_value=self.head_index, device=inp.device
        )
        return self.model.forward(inp=inp, head_index=derived_heads, return_raw=False)


class NullTuner(MHTuner[_T]):
    """Tuner that does nothing.

    Provided to allow non-tuned models to function in tuned pipelines.
    """

    def __init__(self, base_model: Callable[[_T], Tensor]) -> None:
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


class LinearTuner(MHTuner[_T]):
    """Applies a simple linear network tuning.

    This class is built to apply to one-dimensional input (not including batch
    dimension).

    This model adjusts the output of the core model using a linear model that
    is specific to each unique integer of input-2.  This corresponds to the
    following effective structure on each input.

                         <input-1>                      <input-2>
                             |                              |
                      <wrapped model>          <select linear tuning head>
                             |                              |
                    +--------+                              |
                    |        |                              |
                    |        +--<apply linear tuning head>--+
                    |                       |
                    |                       |
               <raw output>           <tuned output>

    `input-2` is interpreted as an index selecting which head to use; as a
    result, n_heads must represent the length of a list that can be indexed by
    all integers that might be seen.
    """

    def __init__(
        self,
        base_model: Callable[[_T], Tensor],
        n_heads: int,
        residual_connection: bool = True,
        bias: bool = True,
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
        residual_connection:
            Whether to apply adjustment in the form f(x) = correction + x. Typically
            a good idea for stability of gradient based training.
        bias:
            Whether each linear head should have a bias.

        """
        super().__init__(base_model)
        self.residual_connection = residual_connection
        # note that this is n different networks. can be recast
        # into a vector-valued final linear layer.
        self.heads = nn.ModuleList([nn.Linear(1, 1, bias=bias) for _ in range(n_heads)])

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
        reshaped = atleast_2d(signal)
        contributions = []
        for exp, cal in enumerate(self.heads):
            individual_corrected = cal(reshaped)
            # create mask that as zeros
            # might need a view on head_index if we want to be shape robust.
            mask = (exp == head_index).view(-1, 1)  # type: ignore
            contr = mask * individual_corrected
            contributions.append(contr)
        if self.residual_connection:
            corrected = sum(contributions) + signal.view(-1, 1)
        else:
            corrected = sum(contributions)
        # accounts for whether the input was shape (batch,) or (batch, 1)
        return corrected.view(signal_shape)


class SharedFanTuner(MHTuner[_T]):
    """Applies a simple shared head-based linear network tuning.

    This class is built to apply to one-dimensional input (not including batch
    dimension).

    Input is fist "featurized" by applying a linear layer that increases the
    dimensionality (fans it out) and then a shared function. Second, a linear layer
    is applied to linearly combine these features into an output signal. The first
    step is shared among the different tuning heads, but the second is not.

    This corresponds to the following effective structure on each input.

                            <input-1>               <input-2>
                                |                       |
                          <wrapped model>               |
                                |                       |
                    +-----------+                       |
                    |           |                       |
                    |   <fan featurization>  <select tuning head>
                    |           |                       |
                    |           +--<apply tuning head>--+
                    |                      |
               <raw output>           <tuned output>

    Here, each tuning head is a simple linear layer. `input-2` is interpreted
    as an index selecting which head to use; as a result, n_heads must represent
    the length of a list that can be indexed by all integers that might be seen.
    """

    def __init__(
        self,
        base_model: Callable[[_T], Tensor],
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
            corrected = sum(contributions) + signal.view(-1, 1)
        else:
            corrected = sum(contributions)
        # accounts for whether the input was shape (batch,) or (batch, 1)
        return corrected.view(signal_shape)


class FFNTuner(MHTuner[_T]):
    """Tunes input using multiple fully connected networks.

    This class is built to apply to one-dimensional input (not including batch
    dimension).

    Input is processed by one of multiple feed forward networks. No parameters
    are shared between the networks.

    This corresponds to the following effective structure on each input.

                            <input-1>               <input-2>
                                |                       |
                    +-----------+                       |
                    |           |              <select tuning head>
                    |           |                       |
                    |           +--<apply tuning head>--+
                    |                      |
               <raw output>           <tuned output>

    Here, each tuning head is a simple linear layer. `input-k` is interpreted
    as an index selecting which head to use; as a result, n_heads must represent
    the length of a list that can be indexed by all integers that might be seen.

    """

    def __init__(
        self,
        base_model: Callable[[_T], Tensor],
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
