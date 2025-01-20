"""Provides compatible for some tools from the bioembeddings package."""

from typing import Final, Sequence, Iterable
from .core import IntEncoder, get_default_int_encoder
from torch import Tensor

from transformers import T5Tokenizer, T5EncoderModel  # type:ignore
import torch

def chunks(inp: Sequence, n: int) -> Iterable[Sequence]:
    """Yield successive n-sized chunks from inp.

    From stack overflow:
    https://stackoverflow.com/questions/312443/
    how-do-i-split-a-list-into-equally-sized-chunks
    """
    for i in range(0, len(inp), n):
        yield inp[i:(i + n)]


def _example(device: str = "cuda") -> Tensor:
    tokenizer = T5Tokenizer.from_pretrained(
        "Rostlab/prot_t5_xl_half_uniref50-enc", do_lower_case=False
    )

    # Load the model
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc").to(
        device
    )

    # prepare your protein sequences as a list
    sequence_examples = [" ".join(x) for x in ["PRTEINN", "SEQWENC"]]

    ids = tokenizer(sequence_examples, add_special_tokens=True)

    input_ids = torch.tensor(ids["input_ids"]).to(device)
    attention_mask = torch.tensor(ids["attention_mask"]).to(device)

    # generate embeddings
    with torch.no_grad():
        embedding_repr = model(input_ids=input_ids, attention_mask=attention_mask)

    emb = embedding_repr.last_hidden_state

    return emb


class T5EncoderWrapper:
    """Featurizer acting on integer encoded sequences.

    Note that this wrapping is not optimized, and may be slow for each encoding
    invocation.

    """

    T5_huggingface_name: Final = "Rostlab/prot_t5_xl_half_uniref50-enc"

    def __init__(
        self,
        integer_encoder: IntEncoder,
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
        self.tokenizer = T5Tokenizer.from_pretrained(
            self.T5_huggingface_name, do_lower_case=False
        )

        # Load the model
        self.t5 = T5EncoderModel.from_pretrained(self.T5_huggingface_name).to(device)
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

        emb = embedding_repr.last_hidden_state

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
        processed = []
        for piece in chunks(int_encoded,self.batch_size):
            processed.append(self.vectorized_encode(piece))
        return torch.concatenate(processed,axis=0) # type: ignore
