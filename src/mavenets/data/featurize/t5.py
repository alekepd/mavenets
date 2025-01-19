"""Provides compatible for some tools from the bioembeddings package."""

from typing import Final
from .core import IntEncoder, get_default_int_encoder
from torch import Tensor

from transformers import T5Tokenizer, T5EncoderModel  # type:ignore
import torch


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

    def __init__(self, integer_encoder: IntEncoder, device: str) -> None:
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

        """
        if integer_encoder is None:
            integer_encoder = get_default_int_encoder()
        self.device = device
        self.tokenizer = T5Tokenizer.from_pretrained(
            self.T5_huggingface_name, do_lower_case=False
        )

        # Load the model
        self.t5 = T5EncoderModel.from_pretrained(self.T5_huggingface_name).to(device)
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
        # need
        formatted = [" ".join(x) for x in str_features]
        ids = self.tokenizer(formatted, add_special_tokens=True)
        input_ids = torch.tensor(ids["input_ids"]).to(self.device)
        attention_mask = torch.tensor(ids["attention_mask"]).to(self.device)

        # generate embeddings
        with torch.no_grad():
            embedding_repr = self.t5(input_ids=input_ids, attention_mask=attention_mask)

        emb = embedding_repr.last_hidden_state

        return emb
