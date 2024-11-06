"""Routines to featurize data."""

from typing import List, Dict, overload, Literal, Union
import pandas as pd  # type: ignore
import torch


def get_alphabet(frame: pd.DataFrame, column_name: str) -> List[str]:
    """Return all possible letters."""
    return sorted(set("".join(frame.loc[:, column_name])))


def encoder_dict(alphabet: List[str]) -> Dict[str, int]:
    """Create encoder dictionary."""
    return {letter: ind for ind, letter in enumerate(alphabet)}


def decoder_dict(alphabet: List[str]) -> Dict[int, str]:
    """Create encoder dictionary."""
    return dict(enumerate(alphabet))


class IntEncoder:
    """Encodes strings as integer lists or tensors."""

    def __init__(self, alphabet: List[str]) -> None:
        """Create backing dictionaries."""
        self.encode_map = encoder_dict(alphabet)
        self.decode_map = decoder_dict(alphabet)

    @overload
    def encode(self, target: str, tensor: Literal[True]) -> torch.Tensor:
        ...

    @overload
    def encode(self, target: str, tensor: Literal[False]) -> List[int]:
        ...

    def encode(self, target: str, tensor: bool) -> Union[List[int], torch.Tensor]:
        """Encode single example."""
        encoded = [self.encode_map[x] for x in target]
        if tensor:
            return torch.tensor(encoded, dtype=torch.int32)
        else:
            return encoded

    def batch_encode(self, targets: List[str]) -> torch.Tensor:
        """Encode multiple examples into single tensor."""
        encoded = [self.encode(x, tensor=True) for x in targets]
        return torch.stack(encoded, dim=0)

    def __len__(self) -> int:
        """Return number of known symbols."""
        return len(self.encode_map)
