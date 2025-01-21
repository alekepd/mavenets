"""Simple transformations for cleaning data.

Objects should follow fit/transform interface typical to sklearn, but operate
on torch.Tensors.

These objects are not torch modules and will not be trained via typical torch
procedures.
"""

from typing import Protocol
from torch import Tensor, flatten, no_grad, clamp


class SKT_protocol(Protocol):
    """Protocol representing sklearn-style transform.

    Transforms are fit on sample data and then can be repeatedly applied.

    """

    def fit(
        self,
        data: Tensor,
        /,
    ) -> None:
        """Fit settings of transform on sample data.

        No transformed data is returned.
        """
        ...

    def transform(
        self,
        data: Tensor,
        /,
    ) -> Tensor:
        """Fit settings of transform on sample data."""
        ...


class Whiten:
    """Perform component-wise whitening calculated across a dataset."""

    def __init__(self, min_stdev: float = 1e-7) -> None:
        """Store options.

        Arguments:
        ---------
        min_stdev:
            Calculated standard deviations are clipped to this value from
            below. Avoids divide by zero errors.

        """
        self.already_fit = False
        self.min_stdev = min_stdev

    def fit(self, data: Tensor, /) -> None:
        """Calculate means and standard deviations of input data.

        Arguments:
        ---------
        data:
            data to train trainsform on. Should be at least two dimensions, with first
            dimension indexing examples (i.e., a the batch dimension).

        Returns:
        -------
        None

        """
        with no_grad():
            shape = data.shape
            if len(shape) == 1:
                raise ValueError("Input tensor must have rank of at least 2.")
            reshaped = flatten(data, start_dim=1)
            reshaped_means = reshaped.mean(dim=0)
            reshaped_stdevs = clamp(reshaped.std(dim=0), min=self.min_stdev)
            self.means = reshaped_means.view(shape[1:])[None, ...]
            self.stdevs = reshaped_stdevs.view(shape[1:])[None, ...]

        self.already_fit = True

    def transform(self, data: Tensor, /) -> Tensor:
        """Whiten data using previously calculated means and standard deviations.

        Arguments:
        ---------
        data:
            Data to whiten. Not used to calculate means or standard deviations.

        Returns:
        -------
        Transformed tensor.

        """
        if not self.already_fit:
            raise ValueError("Transform not yet fit.")
        with no_grad():
            to_return = (data - self.means) / self.stdevs
        return to_return


class NullTransform:
    """Dummy transform that does not modify data."""

    def __init__(self, copy: bool = True) -> None:
        """Store options.

        Arguments:
        ---------
        copy:
            Whether to copy input tensor when applying transform. Non-null transforms
            do not operate in place, so True provides more consistent behavior. However,
            it may use up GPU memory.

        """
        self.copy = copy

    def fit(self, data: Tensor, /) -> None:
        """Do nothing. Provided for interface/protocol compatibility.

        Arguments:
        ---------
        data:
            ignored

        Returns:
        -------
        None

        """

    def transform(self, data: Tensor, /) -> Tensor:
        """Return data unchanged.

        Data may be copied or not; see __init__ docstring.

        Arguments:
        ---------
        data:
            Data that is returned. May be copied; see __init__ options.

        Returns:
        -------
        torch.Tensor

        """
        with no_grad():
            if self.copy:
                to_return = data.detach().clone()
            else:
                to_return = data
        return to_return
