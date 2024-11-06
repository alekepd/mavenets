"""Callables to load datasets."""

from typing import Final, Tuple, Iterable, Literal, Union
import pandas as pd  # type: ignore
from torch.utils.data import TensorDataset
from torch import tensor, float32, int32
from pathlib import Path
from .spec import DATA_SPECS, DataSpec, resolve_dataspec
from .featurize import IntEncoder, get_alphabet

# column names for labeling loaded csvs.
CSV_RID_CNAME: Final = "seq_id"
SEQ_CNAME: Final = "sequence"
SIGNAL_CNAME: Final = "signal"
EXPERIMENT_CNAME: Final = "experiment_index"


# applies column names solely based on column order.
def _csv_read(f: Path) -> pd.DataFrame:
    frame = pd.read_csv(f, header=None)
    frame.columns = [CSV_RID_CNAME, SEQ_CNAME, SIGNAL_CNAME]
    return frame


def get_datasets(
    device: str,
    train_specs: Union[None, Iterable[DataSpec],Iterable[int],Iterable[str]] = None,
    val_specs: Union[None, Iterable[DataSpec],Iterable[int],Iterable[str]] = None,
    feat_type: Literal["integer"] = "integer",
    parent_path: Path = Path(),
) -> Tuple[TensorDataset, TensorDataset]:
    """Load and process data."""
    if feat_type != "integer":
        raise ValueError("Only integer featurization is supported.")

    if train_specs is None:
        train_specs = DATA_SPECS

    if val_specs is None:
        val_specs = DATA_SPECS

    train_frames = []

    for sp in (resolve_dataspec(x) for x in train_specs):
        print(sp)
        tf = _csv_read(parent_path / sp.train_filename)
        tf[EXPERIMENT_CNAME] = sp.index
        train_frames.append(tf)

    train_frame = pd.concat(train_frames)

    valid_frames = []

    # this sorted is a good idea to make things stable.
    for sp in (resolve_dataspec(x) for x in val_specs):
        print(sp)
        vf = _csv_read(parent_path / sp.valid_filename)
        vf[EXPERIMENT_CNAME] = sp.index
        valid_frames.append(vf)

    valid_frame = pd.concat(valid_frames)

    enc = IntEncoder(get_alphabet(train_frame, SEQ_CNAME))

    train_encoded = enc.batch_encode(train_frame.loc[:, SEQ_CNAME])
    train_signal = tensor(train_frame.loc[:, SIGNAL_CNAME].to_numpy(), dtype=float32)
    train_dset_id = tensor(train_frame.loc[:, EXPERIMENT_CNAME].to_numpy(), dtype=int32)

    valid_encoded = enc.batch_encode(valid_frame.loc[:, SEQ_CNAME])
    valid_signal = tensor(valid_frame.loc[:, SIGNAL_CNAME].to_numpy(), dtype=float32)
    valid_dset_id = tensor(valid_frame.loc[:, EXPERIMENT_CNAME].to_numpy(), dtype=int32)

    train_dataset = TensorDataset(
        train_encoded.to(device), train_signal.to(device), train_dset_id.to(device)
    )
    valid_dataset = TensorDataset(
        valid_encoded.to(device), valid_signal.to(device), valid_dset_id.to(device)
    )

    return train_dataset, valid_dataset
