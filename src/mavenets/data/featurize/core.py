"""Routines to featurize sequence data."""

from typing import (
    List,
    Dict,
    overload,
    Literal,
    Union,
    Iterable,
    Optional,
    Protocol,
    Final,
)
from functools import lru_cache
import pandas as pd  # type: ignore
import torch
from torch.nn.functional import one_hot
from torch import float32, int64
from ..spec import SARS_COV2_SEQ

# known amino acid codes. Defining them statically here allows
# reproduciblity if models are training on datasets lacking chemical coverage.
BASE_ALPHA: Final = [
    "A",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "V",
    "W",
    "Y",
    "X",
]


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


def _dict_sanity_check() -> None:
    letters = list(SARS_COV2_SEQ)
    enc = encoder_dict(BASE_ALPHA)
    dec = decoder_dict(BASE_ALPHA)
    for let in letters:
        encoded = enc[let]
        if dec[encoded] != let:
            raise ValueError(
                "Encoder does not satisfy round-trip equivalence in sanity check."
            )


_dict_sanity_check()


class _p_encode(Protocol):
    @overload
    def __call__(
        self, target: str, tensor: Literal[True], device: Optional[str] = ...
    ) -> torch.Tensor:
        ...

    @overload
    def __call__(
        self, target: str, tensor: Literal[False], device: Optional[str] = ...
    ) -> List[int]:
        ...

    def __call__(
        self, target: str, tensor: bool, device: Optional[str] = None
    ) -> Union[List[int], torch.Tensor]:
        ...


class _p_decode(Protocol):
    def __call__(self, target: Iterable[int], cast: bool = True) -> str:
        ...


class IntEncoder:
    """Encodes strings as integer lists or tensors."""

    def __init__(
        self, alphabet: List[str], lru_cache_size: Optional[int] = None
    ) -> None:
        """Create backing dictionaries.

        Arguments:
        ---------
        alphabet:
            list of letters in dictionary. Must have unique elements.
        lru_cache_size:
            If a positive integer, then we wrap encoding in an LRU cache of this size.
            Note that when serving mutable encodings (e.g., tensors), this option
            can share memory/objects if two equal but distinct arguments are encoded.
            When using batch encoding this likely will not happen since there is an
            additional stacking step.

        """
        if len(set(alphabet)) != len(alphabet):
            raise ValueError("Alphabet does not comprise unique elements.")
        self.alphabet: Final = alphabet
        self.encode_map: Final = encoder_dict(alphabet)
        self.decode_map: Final = decoder_dict(alphabet)
        if lru_cache_size:
            # it seems like lru_cache is picking up on only the first overloaded type
            # definition, so we override the type.
            self.encode: _p_encode = lru_cache(maxsize=lru_cache_size)(self._encode)  # type: ignore
            self.decode: _p_decode = lru_cache(maxsize=lru_cache_size)(self._decode)  # type: ignore
        else:
            self.encode = self._encode
            self.decode = self._decode

    @overload
    def _encode(
        self, target: str, tensor: Literal[True], device: Optional[str] = ...
    ) -> torch.Tensor:
        ...

    @overload
    def _encode(
        self, target: str, tensor: Literal[False], device: Optional[str] = ...
    ) -> List[int]:
        ...

    def _encode(
        self, target: str, tensor: bool, device: Optional[str] = None
    ) -> Union[List[int], torch.Tensor]:
        """Encode single example.

        Arguments:
        ---------
        target:
            Single string to encode.
        tensor:
            If true, encoded content is returned as a Tensor. Else, returned as a
            list of ints.
        device:
            Device to make tensor on. Likely "cpu" or "cuda".

        Returns:
        -------
        List of integers or a Tensor.

        """
        encoded = [self.encode_map[x] for x in target]
        if tensor:
            return torch.tensor(encoded, dtype=torch.int32, device=device)
        else:
            return encoded

    def _decode(self, target: Iterable[int], cast: bool = True) -> str:
        """Decode single example.

        Translates from integers to a string.

        Arguments:
        ---------
        target:
           Iterable of integers to transform into a string.
        cast:
            If True, we cal int() on each symbol for attempting lookup.

        Returns:
        -------
        string

        """
        if cast:
            return "".join([self.decode_map[int(x)] for x in target])
        else:
            return "".join([self.decode_map[x] for x in target])

    def batch_decode(self, target: Iterable[Iterable[int]]) -> List[str]:
        """Decode iterable of examples.

        Translates from integers to strings.

        Arguments:
        ---------
        target:
           Iterable of integers to transform into a string.

        Returns:
        -------
        List of strings.

        """
        return [self.decode(x) for x in target]

    def batch_encode(
        self, targets: Iterable[str], device: Optional[str] = None
    ) -> torch.Tensor:
        """Encode multiple examples into single tensor.

        Only supports encoding into a Tensor.
        """
        encoded = [self.encode(x, tensor=True, device=device) for x in targets]
        return torch.stack(encoded, dim=0)

    def __len__(self) -> int:
        """Return number of known symbols."""
        return len(self.encode_map)


def get_default_int_encoder(cache_size: Optional[int] = None) -> IntEncoder:
    """Return default integer encoder."""
    return IntEncoder(BASE_ALPHA, lru_cache_size=cache_size)


def _default_encoder_sanity_check() -> None:
    enc = get_default_int_encoder()
    sample = "".join(enc.alphabet)
    if sample != enc.decode(enc.encode(sample, tensor=False)):
        raise ValueError(
            "Default encoder does not satisfy round-trip equivalence in sanity check."
        )


_default_encoder_sanity_check()


def int_to_floatonehot(int_form: torch.Tensor, num_classes: int = -1) -> torch.Tensor:
    """Transform 32-bit integer encoding to a float32 one-hot encoding.

    Arguments:
    ---------
    int_form:
        Tensor to Transform. Cast to int64 internally.
    num_classes:
        Number of classes (size) to use in one hot encoding. If -1, the number of
        classes present in the data is used. See one_hot.

    Returns:
    -------
    float32 one hot encoding.

    """
    encoded = one_hot(
        int_form.to(int64),
        num_classes=num_classes,
    ).to(float32)
    return encoded
