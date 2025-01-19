"""Provides compatible for some tools from the bioembeddings package."""

from typing import Optional

from bio_embeddings.embed import (  # type: ignore
    EmbedderInterface,
    SeqVecEmbedder,
    ProtTransBertBFDEmbedder,
    ESM1bEmbedder,
)

from .core import IntEncoder, get_default_int_encoder
from torch import Tensor, as_tensor, stack

# maps names to to all servable encoders.
_CATALOG = {
    "seqvec": SeqVecEmbedder,
    "protbert": ProtTransBertBFDEmbedder,
    "esm1b": ESM1bEmbedder,
}


class BioembeddingEncoderWrapper:
    """Featurizer acting on integer encoded sequences.

    Note that this wrapping is not optimized, and may be slow for each encoding
    invocation.

    """

    def __init__(
        self, base_encoder: EmbedderInterface, integer_encoder: IntEncoder
    ) -> None:
        """Store options.

        Arguments:
        ---------
        base_encoder:
            Embedder from the bio_embeddings library. Must support an `embed` method
            that maps a string (fasta) - encoded sequence to a numpy 1-array.
        integer_encoder:
            IntEncoder instance whose batch_decode method will be used to create
            string-represented sequences from integer encoded sequences. Must correspond
            to how sequences later passed to batch_encode were originally
            integer-encoded.

        """
        self.base_encoder = base_encoder
        self.integer_encoder = integer_encoder

    def batch_encode(self, int_encoded: Tensor) -> Tensor:
        """Encode batch of sequences.

        Arguments:
        ---------
        int_encoded:
            Tensor of shape (b, d): b specifies the sample (i.e., batch dimension). Each
            element is an integer encoded (not one-hot) encoded element. Should
            be encoded using the integer_encoder created an initialization.

        Returns:
        -------
        Returns Tensor with first dimension the batch dimension.

        """
        str_features = self.integer_encoder.batch_decode(int_encoded)
        featurized = [self.base_encoder.embed(seq) for seq in str_features]
        stacked = stack([as_tensor(x) for x in featurized], dim=0)
        return stacked


def get_wrapped_bioembedding_encoder(
    name: str, int_encoder: Optional[IntEncoder]
) -> BioembeddingEncoderWrapper:
    """Create integer-compatible encoder from bioembeddings package.

    Note that the returned encoding routines may only be _based_ on the corresponding
    encoder.

    Arguments:
    ---------
    name:
        Name of encoder to obtain. Currently, we support: "seqvec", "protbert", and
        "esm1b".
    int_encoder:
        Integer encoder used to wrap the loaded bioemedding encoder. If None,
        the output of get_default_int_encoder. Using None should be sufficient in most
        cases.

    Returns:
    -------
    Wrapped encoder object. Can be called directly on a batch of integer
    encoded sequences in torch tensor form.

    """
    if int_encoder is None:
        int_encoder = get_default_int_encoder()
    try:
        base_encoder = _CATALOG[name]()
    except KeyError as e:
        raise ValueError(
            "Unknown bioembedding encoder name. "
            "Known encoders: {}".format(list(_CATALOG.keys()))
        ) from e
    wrapped = BioembeddingEncoderWrapper(
        base_encoder=base_encoder, integer_encoder=int_encoder
    )
    return wrapped
