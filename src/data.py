"""Routines to load and encode data."""

from typing import List, Dict, overload, Literal, Union, Tuple, Final
import pandas as pd  # type: ignore
import torch
from torch.utils.data import TensorDataset
from dataclasses import dataclass


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


DATA_FILENAMES: Final = [
    ("train_data.csv", "valid_data.csv"),
    ("not_norm_train_data_B1351.csv", "not_norm_valid_data_B1351.csv"),
    ("not_norm_train_data_E484K.csv", "not_norm_valid_data_E484K.csv"),
    ("not_norm_train_data_N501Y.csv", "not_norm_valid_data_N501Y.csv"),
    ("not_norm_train_data_omicron_BA1.csv", "not_norm_valid_data_omicron_BA1.csv"),
    ("not_norm_train_data_omicron_BA2.csv", "not_norm_valid_data_omicron_BA2.csv"),
    (
        "not_norm_train_data_omicron_Wuhan_Hu_1.csv",
        "not_norm_valid_data_omicron_Wuhan_Hu_1.csv",
    ),
    ("not_norm_train_data_Wuhan_Hu_1.csv", "not_norm_valid_data_Wuhan_Hu_1.csv"),
]


@dataclass(order=True)
class DataSpec:
    """Class for keeping track of an item in inventory."""

    train_filename: str
    valid_filename: str


DATA_SPECS: Final = [
    DataSpec(train_filename=x[0], valid_filename=x[1]) for x in DATA_FILENAMES
]

CSV_RID_CNAME: Final = "seq_id"
SEQ_CNAME: Final = "sequence"
SIGNAL_CNAME: Final = "signal"
EXPERIMENT_CNAME: Final = "experiment_index"


def _csv_read(f: str) -> pd.DataFrame:
    frame = pd.read_csv(f, header=None)
    frame.columns = [CSV_RID_CNAME, SEQ_CNAME, SIGNAL_CNAME]
    return frame


def get_data(
    device: str,
) -> Tuple[TensorDataset, TensorDataset]:
    """Load and process data."""
    train_frames = []
    valid_frames = []

    # this sorted is a good idea to make things stable.
    for id_num, d in enumerate(sorted(DATA_SPECS)):
        tf = _csv_read(d.train_filename)
        vf = _csv_read(d.valid_filename)
        tf[EXPERIMENT_CNAME] = id_num
        vf[EXPERIMENT_CNAME] = id_num
        train_frames.append(tf)
        valid_frames.append(vf)

    train_frame = pd.concat(train_frames)
    valid_frame = pd.concat(valid_frames)

    enc = IntEncoder(get_alphabet(train_frame, SEQ_CNAME))

    train_encoded = enc.batch_encode(train_frame.loc[:, SEQ_CNAME])
    train_signal = torch.tensor(
        train_frame.loc[:, SIGNAL_CNAME].to_numpy(), dtype=torch.float32
    )
    train_dset_id = torch.tensor(
        train_frame.loc[:, EXPERIMENT_CNAME].to_numpy(), dtype=torch.int32
    )

    valid_encoded = enc.batch_encode(valid_frame.loc[:, SEQ_CNAME])
    valid_signal = torch.tensor(
        valid_frame.loc[:, SIGNAL_CNAME].to_numpy(), dtype=torch.float32
    )
    valid_dset_id = torch.tensor(
        valid_frame.loc[:, EXPERIMENT_CNAME].to_numpy(), dtype=torch.int32
    )

    train_dataset = TensorDataset(
        train_encoded.to(device), train_signal.to(device), train_dset_id.to(device)
    )
    valid_dataset = TensorDataset(
        valid_encoded.to(device), valid_signal.to(device), valid_dset_id.to(device)
    )

    return train_dataset, valid_dataset
