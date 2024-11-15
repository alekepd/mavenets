"""Contains descriptions of all considered datasets.

Objects here allow data files to be associated with experiment IDs
and other data attributes. Routines are provided to conveniently 
query available files.
"""

from typing import Final, Union
from dataclasses import dataclass, asdict
from itertools import chain
from pathlib import Path


# this file only contains structure for residues 5-187 inclusive
# this is not yet correctly taken into account in some places.
LEGACY_SARSCOV2_FILENAME: Final = Path("RBD_amaro.pdb")

# this has all residues
SARSCOV2_FILENAME: Final = Path("RBD_7DX5.pdb")

SARS_COV2_SEQ: Final = (
    "NITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKL"
    "NDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSK"
    "VGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTN"
    "GVGYQPYRVVVLSFELLHAPATVCGPKKST"
)


@dataclass(order=True, frozen=True)
class DataSpec:
    """Describes all files associated with a given experiment."""

    name: str  # convienient name labeling the experiment.
    train_filename: Path  # local path of train csv
    valid_filename: Path  # local path of valid csv
    index: int  # positive unique labeling experiment.


DATA_SPECS: Final = [
    DataSpec(
        name="base",
        train_filename=Path("train_data.csv"),
        valid_filename=Path("valid_data.csv"),
        index=0,
    ),
    DataSpec(
        name="B1351",
        train_filename=Path("not_norm_train_data_B1351.csv"),
        valid_filename=Path("not_norm_valid_data_B1351.csv"),
        index=1,
    ),
    DataSpec(
        name="E484K",
        train_filename=Path("not_norm_train_data_E484K.csv"),
        valid_filename=Path("not_norm_valid_data_E484K.csv"),
        index=2,
    ),
    DataSpec(
        name="N501Y",
        train_filename=Path("not_norm_train_data_N501Y.csv"),
        valid_filename=Path("not_norm_valid_data_N501Y.csv"),
        index=3,
    ),
    DataSpec(
        name="BA1",
        train_filename=Path("not_norm_train_data_omicron_BA1.csv"),
        valid_filename=Path("not_norm_valid_data_omicron_BA1.csv"),
        index=4,
    ),
    DataSpec(
        name="BA2",
        train_filename=Path("not_norm_train_data_omicron_BA2.csv"),
        valid_filename=Path("not_norm_valid_data_omicron_BA2.csv"),
        index=5,
    ),
    DataSpec(
        name="wuhan_omicron",
        train_filename=Path("not_norm_train_data_omicron_Wuhan_Hu_1.csv"),
        valid_filename=Path("not_norm_valid_data_omicron_Wuhan_Hu_1.csv"),
        index=6,
    ),
    DataSpec(
        name="wuhan",
        train_filename=Path("not_norm_train_data_Wuhan_Hu_1.csv"),
        valid_filename=Path("not_norm_valid_data_Wuhan_Hu_1.csv"),
        index=7,
    ),
]

MAX_DATASPEC_INDEX: Final = max(x.index for x in DATA_SPECS)


def resolve_dataspec(identifier: Union[str, int, DataSpec]) -> DataSpec:
    """Return data specification matching name or index.

    Arguments:
    ---------
    identifier:
        Either a string, integer, or Dataspec. If a Dataspec, directly
        returned. Otherwise, matched against all possible index and
        name fields of stored experiments.

    Returns:
    -------
    Dataspec

    """
    if isinstance(identifier, DataSpec):
        return identifier
    for x in DATA_SPECS:
        if identifier == x.name or identifier == x.index:
            return x
    else:
        raise ValueError("No matching DataSpec found.")


def _sanity_check() -> None:
    """Perform basic checks to avoid hard-to-find typographical errors."""
    # check to make sure no fields are duplicated anywhere
    _lists = [list(asdict(x).values()) for x in DATA_SPECS]
    _all = list(chain.from_iterable(_lists))
    assert len(_all) == len(set(_all))
    del _lists
    del _all

    # check to make sure that all index ints are >= 0
    assert all(x.index >= 0 for x in DATA_SPECS)


_sanity_check()
