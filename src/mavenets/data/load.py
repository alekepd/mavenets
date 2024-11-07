"""Callables to load datasets."""

from typing import Final, Tuple, Iterable, Literal, Union
import pandas as pd  # type: ignore
from torch.utils.data import TensorDataset
from torch import tensor, float32, int32, int64
from torch.nn.functional import one_hot
from pathlib import Path
from .spec import DATA_SPECS, DataSpec, resolve_dataspec
from .featurize import IntEncoder, get_alphabet

# column names for labeling loaded csvs.
CSV_RID_CNAME: Final = "seq_id"
SEQ_CNAME: Final = "sequence"
SIGNAL_CNAME: Final = "signal"
EXPERIMENT_CNAME: Final = "experiment_index"

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
]


# applies column names solely based on column order.
def _csv_read(f: Path) -> pd.DataFrame:
    frame = pd.read_csv(f, header=None)
    frame.columns = [CSV_RID_CNAME, SEQ_CNAME, SIGNAL_CNAME]
    return frame


def get_datasets(
    device: str,
    train_specs: Union[None, Iterable[DataSpec], Iterable[int], Iterable[str]] = None,
    val_specs: Union[None, Iterable[DataSpec], Iterable[int], Iterable[str]] = None,
    feat_type: Literal["integer", "onehot"] = "integer",
    parent_path: Path = Path(),
) -> Tuple[TensorDataset, TensorDataset]:
    """Load and process data.

    Arguments:
    ---------
    device:
        Torch device on which to place the entire datasets. Likely "cuda" or "cpu".
    train_specs:
        None or Iterable of ints or strings used to specify what datasets to place in
        the training dataset. If integers, compared against the index of the specs; if
        a string, compared against the names. If None, all data sets are used.
    val_specs:
        None or Iterable of ints or strings used to specify what datasets to place in
        the validation dataset. If integerss, compared against the index of the specs;
        if a string, compared against the names. If None, all data sets are used.
    feat_type:
        Featurization used; only "integer" and "onehot" are accepted. "integer" 
        corresponds to a vector with one integer entry per amino acid determining
        the residue type. "onehot" creates a 0-1 vector that is longer with the same
        information (see torch.nn.functional.one_hot). Note that the one hot is
        converted to the float32 dtype.
    parent_path:
        Path object specifying where to look for csv files.

    Returns:
    -------
    2-Tuple of TensorDatasets on the specified device.

    TensorDatasets return (feat, signal, dataset_index) during iteration.

    """
    if feat_type not in ("integer", "onehot"):
        raise ValueError("Only integer featurization is supported.")

    if train_specs is None:
        train_specs = DATA_SPECS

    if val_specs is None:
        val_specs = DATA_SPECS

    train_frames = []

    # load training datasets, record index
    for sp in (resolve_dataspec(x) for x in train_specs):
        tf = _csv_read(parent_path / sp.train_filename)
        tf[EXPERIMENT_CNAME] = sp.index
        train_frames.append(tf)

    train_frame = pd.concat(train_frames)

    valid_frames = []

    # load validation datasets, record index
    for sp in (resolve_dataspec(x) for x in val_specs):
        vf = _csv_read(parent_path / sp.valid_filename)
        vf[EXPERIMENT_CNAME] = sp.index
        valid_frames.append(vf)

    valid_frame = pd.concat(valid_frames)

    # make sure that there are no amino acids in the data not in our standard
    # alphabet. We use a standard alphabet to maintain featurization stability
    # across possibly smaller input datasets.
    alpha = get_alphabet(pd.concat([train_frame,valid_frame]), SEQ_CNAME)
    if not set(alpha).issubset(set(BASE_ALPHA)):
        raise ValueError("Data contains residues not represented fixed alphabet.")

    enc = IntEncoder(BASE_ALPHA)

    train_int_encoded = enc.batch_encode(train_frame.loc[:, SEQ_CNAME])
    if feat_type == "onehot":
        # we must cast to int64 to make torch happy
        train_encoded = one_hot(
            train_int_encoded.to(int64), num_classes=len(BASE_ALPHA)
        ).to(float32)
    else:
        train_encoded = train_int_encoded
    train_signal = tensor(train_frame.loc[:, SIGNAL_CNAME].to_numpy(), dtype=float32)
    train_dset_id = tensor(train_frame.loc[:, EXPERIMENT_CNAME].to_numpy(), dtype=int32)

    valid_int_encoded = enc.batch_encode(valid_frame.loc[:, SEQ_CNAME])
    if feat_type == "onehot":
        # we must cast to int64 to make torch happy
        valid_encoded = one_hot(
            valid_int_encoded.to(int64), num_classes=len(BASE_ALPHA)
        ).to(float32)
    else:
        valid_encoded = valid_int_encoded
    valid_signal = tensor(valid_frame.loc[:, SIGNAL_CNAME].to_numpy(), dtype=float32)
    valid_dset_id = tensor(valid_frame.loc[:, EXPERIMENT_CNAME].to_numpy(), dtype=int32)

    train_dataset = TensorDataset(
        train_encoded.to(device), train_signal.to(device), train_dset_id.to(device)
    )
    valid_dataset = TensorDataset(
        valid_encoded.to(device), valid_signal.to(device), valid_dset_id.to(device)
    )

    return train_dataset, valid_dataset
