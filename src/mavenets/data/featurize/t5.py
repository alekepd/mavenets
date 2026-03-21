"""Provides compatible for some tools from the bioembeddings package."""

from typing import Final, List, Optional, Sequence, Iterable, TypeVar
from .core import IntEncoder, get_default_int_encoder
from .transform import IncrementalPCATransform
from torch import Tensor

from transformers import T5Tokenizer, T5EncoderModel  # type:ignore
import torch

T = TypeVar("T")


def chunks(inp: Sequence[T], n: int) -> Iterable[Sequence[T]]:
    """Yield successive n-sized chunks from inp.

    From stack overflow:
    https://stackoverflow.com/questions/312443/
    how-do-i-split-a-list-into-equally-sized-chunks
    """
    for i in range(0, len(inp), n):
        yield inp[i:(i + n)]


class T5EncoderWrapper:
    """Featurizer acting on integer encoded sequences.

    Note that this wrapping is not optimized, and may be slow for each encoding
    invocation.

    """

    T5_huggingface_name: Final = "Rostlab/prot_t5_xl_half_uniref50-enc"

    def __init__(
        self,
        integer_encoder: Optional[IntEncoder],
        device: str,
        flatten: bool = True,
        per_protein: bool = True,
        batch_size: int = 32,
    ) -> None:
        """Store options.

        Arguments:
        ---------
        integer_encoder:
            IntEncoder instance whose batch_decode method will be used to create
            string-represented sequences from integer encoded sequences. Must correspond
            to how sequences later passed to batch_encode were originally
            integer-encoded.
        device:
            torch device identifier (e.g., "cuda")
        flatten:
            If per_protein is True, ignored. Else, if true, we return a matrix
            of shape (b, f) for batched input. Otherwise, the returned values
            are organized by sequence (with an additional entry), shape (b,
            n_res+1, 1024)
        per_protein:
            If True, we average over all residues in each sample to produce a single
            1024-sized vector. Setting this to False keeps embeddings separate per
            amino acid. False is more expressive but may have a massive memory
            footprint.
        batch_size:
            Number of examples to process at once. Bigger values use more memory
            and probably don't help performance.

        """
        if integer_encoder is None:
            integer_encoder = get_default_int_encoder()
        self.device = device
        self.flatten = flatten
        self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(  # type: ignore[reportUnknownMemberType]
            self.T5_huggingface_name, do_lower_case=False
        )

        # Load the model
        self.t5: T5EncoderModel = T5EncoderModel.from_pretrained(self.T5_huggingface_name).to(device)  # type: ignore[no-any-return]
        self.integer_encoder = integer_encoder
        self.per_protein = per_protein
        self.batch_size = batch_size

    def vectorized_encode(self, int_encoded: Tensor) -> Tensor:
        """Encode iterable of sequences.

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
        formatted = [" ".join(x) for x in str_features]
        ids = self.tokenizer(formatted, add_special_tokens=True)
        input_ids = torch.tensor(ids["input_ids"]).to(self.device)
        attention_mask = torch.tensor(ids["attention_mask"]).to(self.device)

        # generate embeddings
        with torch.no_grad():
            embedding_repr = self.t5(input_ids=input_ids, attention_mask=attention_mask)

        emb: Tensor = embedding_repr.last_hidden_state

        if self.per_protein:
            # average over residues, not batch examples
            return emb.mean(dim=1)
        elif self.flatten:
            return emb.flatten(start_dim=1)
        else:
            return emb

    def batch_encode(self, int_encoded: Tensor) -> Tensor:
        """Encode iterable of sequences.

        Unlike vectorized_encode, input is processed in chunks.

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
        processed: List[Tensor] = []
        for piece in chunks(int_encoded, self.batch_size):
            processed.append(self.vectorized_encode(piece))
        return torch.concatenate(processed, axis=0)  # type: ignore[arg-type]


def t5_pca_encode(
    int_encoded: Tensor,
    pca_transform: IncrementalPCATransform,
    integer_encoder: IntEncoder,
    device: str,
    fit_pca: bool,
    batch_size: int = 32,
) -> Tensor:
    """Encode sequences with T5 and apply PCA dimensionality reduction.

    When `fit_pca` is True (training data), this performs two passes through the
    T5 model: the first incrementally fits the PCA transform on per-residue
    embeddings, and the second transforms and collects the reduced features.
    This avoids holding all raw T5 embeddings in memory simultaneously.

    When `fit_pca` is False (validation/test data), only a single pass is needed
    using the already-fitted PCA transform.

    Arguments:
    ---------
    int_encoded:
        Integer-encoded sequences of shape `(n, seq_len)`.
    pca_transform:
        IncrementalPCATransform instance. If `fit_pca` is True, this will be
        fitted incrementally on the training embeddings. If False, it must
        already be fitted.
    integer_encoder:
        IntEncoder used to decode integer sequences back to strings for T5.
    device:
        Torch device for T5 inference (e.g., "cuda" or "cpu").
    fit_pca:
        If True, incrementally fit the PCA on the embeddings before transforming.
        If False, use the already-fitted PCA to transform directly.
    batch_size:
        Number of sequences to process per T5 forward pass.

    Returns:
    -------
    Tensor of shape `(n, seq_len_out * n_components)` for per-residue PCA or
    `(n, n_components)` for global PCA, where `seq_len_out` includes the T5
    EOS token.

    """
    enc = T5EncoderWrapper(
        integer_encoder=integer_encoder,
        device=device,
        per_protein=False,
        flatten=False,
        batch_size=batch_size,
    )

    n = int_encoded.shape[0]

    if fit_pca:
        # Pass 1: Incrementally fit PCA on T5 embeddings
        for start in range(0, n, batch_size):
            piece = int_encoded[start : start + batch_size]
            embeddings = enc.vectorized_encode(piece).cpu()
            pca_transform.partial_fit_chunk(embeddings)
        pca_transform.flush_partial_fit()

    # Transform pass: encode with T5 and project with PCA
    transformed: List[Tensor] = []
    for start in range(0, n, batch_size):
        piece = int_encoded[start : start + batch_size]
        embeddings = enc.vectorized_encode(piece).cpu()
        projected = pca_transform.transform(embeddings)
        transformed.append(projected)

    return torch.concatenate(transformed, dim=0)
