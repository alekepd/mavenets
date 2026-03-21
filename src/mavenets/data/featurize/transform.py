"""Simple transformations for cleaning data.

Objects should follow fit/transform interface typical to sklearn, but operate
on torch.Tensors.

These objects are not torch modules and will not be trained via typical torch
procedures.
"""

from typing import Optional, Protocol

import numpy as np
from sklearn.decomposition import IncrementalPCA  # type: ignore[import-untyped]
from torch import Tensor, flatten, no_grad, clamp
import torch


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


class IncrementalPCATransform:
    """Incremental PCA for dimensionality reduction of per-residue embeddings.

    Supports two modes controlled by `per_residue`:

    - **Per-residue** (`per_residue=True`): Input is 3D `(n, seq_len, embed_dim)`.
      PCA is fitted on the embedding dimension by reshaping to
      `(n*seq_len, embed_dim)`. Output is `(n, seq_len * n_components)`.

    - **Global** (`per_residue=False`): Input is 3D `(n, seq_len, embed_dim)`.
      PCA is fitted on the flattened representation `(n, seq_len*embed_dim)`.
      Output is `(n, n_components)`.

    Designed for memory-constrained settings: call `partial_fit_chunk` repeatedly
    on small batches, then `transform` on each batch independently.

    """

    def __init__(self, n_components: int, per_residue: bool = True) -> None:
        """Store options.

        Arguments:
        ---------
        n_components:
            Number of principal components to retain.
        per_residue:
            If True, PCA operates on the embedding dimension across all residue
            positions. If False, PCA operates on the full flattened representation.

        """
        self.n_components = n_components
        self.per_residue = per_residue
        self._pca = IncrementalPCA(n_components=n_components)
        self.already_fit = False
        self._seq_len: Optional[int] = None
        self._fit_buffer: Optional["np.ndarray[object, np.dtype[np.float64]]"] = None

    def _prepare(self, data: Tensor) -> "np.ndarray[object, np.dtype[np.float64]]":
        """Reshape and convert a 3D tensor for PCA.

        Arguments:
        ---------
        data:
            3D tensor of shape `(n, seq_len, embed_dim)`.

        Returns:
        -------
        2D numpy array ready for PCA fitting or transformation.

        """
        if data.ndim != 3:
            raise ValueError(
                f"Expected 3D tensor (n, seq_len, embed_dim), got {data.ndim}D."
            )
        n, seq_len, embed_dim = data.shape

        if self._seq_len is None:
            self._seq_len = seq_len
        elif self._seq_len != seq_len:
            raise ValueError(
                f"Sequence length mismatch: expected {self._seq_len}, got {seq_len}."
            )

        arr: np.ndarray[object, np.dtype[np.float64]] = data.numpy(force=True).astype(np.float64)
        if self.per_residue:
            return arr.reshape(n * seq_len, embed_dim)
        else:
            return arr.reshape(n, seq_len * embed_dim)

    def partial_fit_chunk(self, chunk: Tensor) -> None:
        """Incrementally fit PCA on a batch of embeddings.

        Rows are buffered internally and ``partial_fit`` is called whenever the
        buffer accumulates at least ``n_components`` rows.  Call
        :meth:`flush_partial_fit` after the last chunk to fit on any remaining
        buffered rows.

        Arguments:
        ---------
        chunk:
            3D tensor of shape `(batch, seq_len, embed_dim)`.

        """
        prepared = self._prepare(chunk)

        if self._fit_buffer is not None:
            self._fit_buffer = np.concatenate([self._fit_buffer, prepared], axis=0)
        else:
            self._fit_buffer = prepared

        # Flush complete blocks of size >= n_components
        while self._fit_buffer.shape[0] >= self.n_components:
            batch = self._fit_buffer[: self.n_components]
            self._fit_buffer = self._fit_buffer[self.n_components :]
            self._pca.partial_fit(batch)
            self.already_fit = True

    def flush_partial_fit(self) -> None:
        """Fit on any remaining buffered rows from :meth:`partial_fit_chunk`.

        Must be called after the last ``partial_fit_chunk`` call to ensure all
        data is used for fitting.  If the buffer contains fewer than
        ``n_components`` rows and PCA has already been partially fitted, the
        remaining rows are used for a final ``partial_fit`` call.  If PCA has
        never been fitted (i.e., total data < ``n_components``), a
        ``ValueError`` is raised.

        """
        if self._fit_buffer is not None and self._fit_buffer.shape[0] > 0:
            if not self.already_fit:
                raise ValueError(
                    f"Total number of samples ({self._fit_buffer.shape[0]}) is less "
                    f"than n_components ({self.n_components}). Cannot fit PCA."
                )
            self._pca.partial_fit(self._fit_buffer)
            self._fit_buffer = None

    def fit(self, data: Tensor, /) -> None:
        """Fit PCA on the full dataset in one call.

        Arguments:
        ---------
        data:
            3D tensor of shape `(n, seq_len, embed_dim)`.

        """
        prepared = self._prepare(data)
        self._pca.fit(prepared)
        self.already_fit = True

    def transform(self, data: Tensor, /) -> Tensor:
        """Transform data using the fitted PCA.

        Arguments:
        ---------
        data:
            3D tensor of shape `(n, seq_len, embed_dim)`.

        Returns:
        -------
        2D tensor of shape `(n, seq_len * n_components)` for per-residue mode,
        or `(n, n_components)` for global mode.

        """
        if not self.already_fit:
            raise ValueError("Transform not yet fit.")

        n = data.shape[0]
        prepared = self._prepare(data)
        transformed: np.ndarray[object, np.dtype[np.float64]] = self._pca.transform(prepared)

        if self.per_residue:
            assert self._seq_len is not None
            # (n*seq_len, k) -> (n, seq_len*k)
            result = transformed.reshape(n, self._seq_len * self.n_components)
        else:
            # already (n, k)
            result = transformed

        return torch.tensor(result, dtype=torch.float32)
