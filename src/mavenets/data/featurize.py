"""Routines to featurize sequence data."""

from typing import List, Dict, overload, Literal, Union, Iterable
import pandas as pd  # type: ignore
import torch


def get_alphabet(frame: pd.DataFrame, column_name: str) -> List[str]:
    """Return all possible letters."""
    return sorted(set("".join(frame.loc[:, column_name])))


def encoder_dict(alphabet: List[str]) -> Dict[str, int]:
    """Create encoder dictionary.

    Arguments:
    ---------
    alphabet:
        list of letters in dictionary. Assumed have unique elements.

    Returns:
    -------
    Dictionary mapping alphabet members to integers.

    """
    return {letter: ind for ind, letter in enumerate(alphabet)}


def decoder_dict(alphabet: List[str]) -> Dict[int, str]:
    """Create encoder dictionary.

    Arguments:
    ---------
    alphabet:
        list of letters in dictionary. Assumed have unique elements.

    Returns:
    -------
    Dictionary mapping integers to alphabet members.

    """
    return dict(enumerate(alphabet))


def _sanity_check() -> None:
    letters = ["A", "D", "E"]
    enc = encoder_dict(letters)
    dec = decoder_dict(letters)
    for let in letters:
        encoded = enc[let]
        if dec[encoded] != let:
            raise ValueError(
                "Encoder does not satisfy round-trip equivalence in sanity check."
            )


_sanity_check()


class IntEncoder:
    """Encodes strings as integer lists or tensors."""

    def __init__(self, alphabet: List[str]) -> None:
        """Create backing dictionaries.

        Arguments:
        ---------
        alphabet:
            list of letters in dictionary. Must have unqiue elements.

        """
        if len(set(alphabet)) != len(alphabet):
            raise ValueError("Alphabet does not comprise unique elements.")
        self.encode_map = encoder_dict(alphabet)
        self.decode_map = decoder_dict(alphabet)

    @overload
    def encode(self, target: str, tensor: Literal[True]) -> torch.Tensor:
        ...

    @overload
    def encode(self, target: str, tensor: Literal[False]) -> List[int]:
        ...

    def encode(self, target: str, tensor: bool) -> Union[List[int], torch.Tensor]:
        """Encode single example.

        Arguments:
        ---------
        target:
            Single string to encode.
        tensor:
            If true, encoded content is returned as a Tensor. Else, returned as a
            list of ints.

        Returns:
        -------
        List of integers or a Tensor.

        """
        encoded = [self.encode_map[x] for x in target]
        if tensor:
            return torch.tensor(encoded, dtype=torch.int32)
        else:
            return encoded

    def batch_encode(self, targets: Iterable[str]) -> torch.Tensor:
        """Encode multiple examples into single tensor.

        Only supports encoding into a Tensor.
        """
        encoded = [self.encode(x, tensor=True) for x in targets]
        return torch.stack(encoded, dim=0)

    def __len__(self) -> int:
        """Return number of known symbols."""
        return len(self.encode_map)
