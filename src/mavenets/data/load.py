"""Tools for loading data."""

from typing import Final, Tuple, Iterable, Literal, Union, Dict, overload, Optional
import pandas as pd  # type: ignore
from torch.utils.data import TensorDataset
from torch import Tensor, tensor, float32, int32
from torch.utils.data import Dataset
from pathlib import Path
from torch_geometric.data import Data  # type: ignore
from .spec import DATA_SPECS, DataSpec, resolve_dataspec, SARSCOV2_FILENAME
from .featurize import (
    get_alphabet,
    get_default_int_encoder,
    int_to_floatonehot,
    IntEncoder,
)
from .featurize.t5 import T5EncoderWrapper
from .graph import get_graph

# column names for labeling loaded MAVE experiment csvs.
CSV_RID_CNAME: Final = "seq_id"
SEQ_CNAME: Final = "sequence"
SIGNAL_CNAME: Final = "signal"
EXPERIMENT_CNAME: Final = "experiment_index"


class DNSEDataset(Dataset):
    """Graph dataset with identical edges but varying labels.

    This creates a dataset with Dynamic Node embeddings and Static Edges. At
    initialization, connectivity and edge features are given which apply to
    all served examples. Other tensors are also provided; these tensors
    are indexed along their first axis to create examples.

    Example:
    -------
    ```
    edge_ind, edge_feat = get_graph(...) # get SARS-COV2 structural graph
    node_feats = torch.randn(5,201) # generate face node labels for 5 structures
    d = DNSEDataset(edge_attr=edge_feat,edge_index=edge_ind,x=node_feats)
    ```

    `d` is now a ataset of 5 examples, each with the same graph but different node
    features. Note feature information is placed under the attribute name `x`. The
    name of the attribute is defined by the name of the argument used; multiple
    tensors may be specified as multiple arguments.

    """

    def __init__(
        self,
        /,
        edge_attr: Tensor,
        edge_index: Tensor,
        **kwargs,
    ) -> None:
        """Store data.

        Arguments are keyword-only.

        Arguments:
        ---------
        edge_attr:
            Tensor of shape (n_edges, n_edge_Feats) coining the features of each edge.
        edge_index:
            Tensor of shape (n_edges, 2) containing the start and end of each edge.
        **kwargs:
            Tensors which are indexed along their leading axis when serving examples.

        """
        super().__init__()
        self.edge_feat = edge_attr
        self.edge_index = edge_index
        if len(kwargs) == 0:
            raise ValueError("Must provide at least one field tensor.")
        self.tensor_dict: Dict[str, Tensor] = kwargs

    def __len__(self) -> int:
        """Return number of data pairs."""
        first_item = next(iter(self.tensor_dict.values()))
        return first_item.shape[0]

    def __getitem__(self, idx: int) -> Data:
        """Slice each underlying tensor, assemble Data object, serve."""
        additional_fields = {}
        for key, value in self.tensor_dict.items():
            # Views
            additional_fields[key] = value[idx]
        # note that batch_based indexing may be affected by the names
        # of attributes
        data = Data(
            **additional_fields,
            edge_attr=self.edge_feat,
            edge_index=self.edge_index,
        )

        return data


# applies column names solely based on column order.
def _mave_csv_read(f: Path) -> pd.DataFrame:
    """Read csv files of a given format from disk and label columns.

    Format is assumed to be:
        first column is an identitication number, second column is sequence
        information, third column is a signal to fit.

    """
    frame = pd.read_csv(f, header=None)
    frame.columns = [CSV_RID_CNAME, SEQ_CNAME, SIGNAL_CNAME]
    return frame


def _get_aggregate_mave_csv(
    specs: Union[Iterable[int], Iterable[str], Iterable[DataSpec]],
    identifier: str,
    directory: Path,
) -> pd.DataFrame:
    """Read selection of csv files based on a given identifier.

    Arguments:
    ---------
    specs:
        Iterable of identifiers to pass to resolve_dataset.
    identifier:
        Used to access each data spec via getattr. Likely "train_filename",
        "valid_filename", or "test_filename".
    directory:
        Where to look for the specified files obtained via the getattr call.

    Returns:
    -------
    DataFrame containing loaded sequences and the corresponding integer
    experiment identifiers under EXPERIMENT_CNAME.

    """
    frames = []

    # load datasets, record index
    for sp in (resolve_dataspec(x) for x in specs):
        tf = _mave_csv_read(directory / getattr(sp, identifier))
        tf[EXPERIMENT_CNAME] = sp.index
        frames.append(tf)

    return pd.concat(frames)


def _process_table(
    frame: pd.DataFrame, feat_type: str, int_encoder: IntEncoder, device: str
) -> Tuple[Tensor, Tensor, Tensor]:
    """Transform data frame into processed tensors.

    Arguments:
    ---------
    frame:
        Data frame to process. Should have columns corresponding to SEQ_CNAME,
        SIGNA_CNAME, and EXPERIMENT_CNAME variables. Likely from _mave_csv_read or
        _get_aggregate_mave_csv.
    feat_type:
        How to featurize data. "onehot" corresponds to one-hot features
        (float32), "integer" corresponds to integer encoding, "t5" uses
        encodings from a pretrained transformer.
    int_encoder:
        IntEncoder instance to perform integer encoding.
    device:
        torch device specifier. Only used for t5 inference.

    Returns:
    -------
    Three tensors: First is the featurized sequences, second is the signal to
    fit against, third contains dataset ids.

    """
    int_encoded = int_encoder.batch_encode(frame.loc[:, SEQ_CNAME])
    if feat_type == "onehot":
        encoded = int_to_floatonehot(int_encoded, num_classes=len(int_encoder.alphabet))
    elif feat_type == "integer":
        encoded = int_encoded
    elif feat_type == "t5":
        enc = T5EncoderWrapper(
            integer_encoder=int_encoder, device=device, per_protein=True
        )
        encoded = enc.batch_encode(int_encoded).cpu()
    else:
        raise ValueError("Unknown featurization type: {}".format(feat_type))

    signal = tensor(frame.loc[:, SIGNAL_CNAME].to_numpy(), dtype=float32)
    dset_id = tensor(frame.loc[:, EXPERIMENT_CNAME].to_numpy(), dtype=int32)
    return encoded, signal, dset_id


@overload
def get_datasets(
    *,
    device: str,
    train_specs: Union[None, Iterable[DataSpec], Iterable[int], Iterable[str]] = ...,
    val_specs: Union[None, Iterable[DataSpec], Iterable[int], Iterable[str]] = ...,
    test_specs: Union[None, Iterable[DataSpec], Iterable[int], Iterable[str]] = ...,
    feat_type: Literal["integer", "onehot", "t5"] = ...,
    graph: bool = ...,
    graph_sequence_window_size: int = ...,
    graph_n_distance_feats: int = ...,
    graph_distance_cutoff: float = ...,
    parent_path: Path = ...,
    include_test: Literal[False],
) -> Tuple[Dataset, Dataset]:
    ...


@overload
def get_datasets(
    *,
    device: str,
    train_specs: Union[None, Iterable[DataSpec], Iterable[int], Iterable[str]] = ...,
    val_specs: Union[None, Iterable[DataSpec], Iterable[int], Iterable[str]] = ...,
    test_specs: Union[None, Iterable[DataSpec], Iterable[int], Iterable[str]] = ...,
    feat_type: Literal["integer", "onehot", "t5"] = ...,
    graph: bool = ...,
    graph_sequence_window_size: int = ...,
    graph_n_distance_feats: int = ...,
    graph_distance_cutoff: float = ...,
    parent_path: Path = ...,
    include_test: Literal[True],
) -> Tuple[Dataset, Dataset, Dataset]:
    ...


@overload
def get_datasets(
    *,
    device: str,
    train_specs: Union[None, Iterable[DataSpec], Iterable[int], Iterable[str]] = ...,
    val_specs: Union[None, Iterable[DataSpec], Iterable[int], Iterable[str]] = ...,
    test_specs: Union[None, Iterable[DataSpec], Iterable[int], Iterable[str]] = ...,
    feat_type: Literal["integer", "onehot", "t5"] = ...,
    graph: bool = ...,
    graph_sequence_window_size: int = ...,
    graph_n_distance_feats: int = ...,
    graph_distance_cutoff: float = ...,
    parent_path: Path = ...,
    include_test: Literal[False] = ...,
) -> Tuple[Dataset, Dataset]:
    ...


def get_datasets(  # noqa: C901
    *,
    device: str,
    train_specs: Union[None, Iterable[DataSpec], Iterable[int], Iterable[str]] = None,
    val_specs: Union[None, Iterable[DataSpec], Iterable[int], Iterable[str]] = None,
    test_specs: Union[None, Iterable[DataSpec], Iterable[int], Iterable[str]] = None,
    feat_type: Literal["integer", "onehot", "t5"] = "integer",
    graph: bool = False,
    graph_sequence_window_size: int = 10,
    graph_n_distance_feats: int = 10,
    graph_distance_cutoff: float = 2.5,
    parent_path: Path = Path(),
    include_test: bool = False,
) -> Union[Tuple[Dataset, Dataset], Tuple[Dataset, Dataset, Dataset]]:
    """Load, featurize, and return SARSCOV2 data for training and evaluation.

    Loads target signal and sequences from disk, and if graph is specified reads
    a file describing the 3d structure of the protein. If graph is True, a Tuple of
    two DNSEDataset is returned; else, two TensorDatasets are returned, first being
    the train data and second the validation data.

    If include_test is True, 3 datasets are returned: train, validation, and test.
    If False, only train and validation are returned.

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
        the validation dataset. If integers, compared against the index of the specs;
        if a string, compared against the names. If None, all data sets are used.
    test_specs:
        None or Iterable of ints or strings used to specify what datasets to place in
        the test dataset. If integers, compared against the index of the specs;
        if a string, compared against the names. If None, all data sets are used. This
        information is only used if include_test is True.
    feat_type:
        Featurization used; only "integer", "onehot", and "t5" are accepted. "integer"
        corresponds to a vector with one integer entry per amino acid determining
        the residue type. "onehot" creates a 0-1 vector that is longer with the same
        information (see torch.nn.functional.one_hot). Note that the one hot is
        converted to the float32 dtype. "t5" uses embeddings from a pretrained
        T5 model from hugging face. Note that t5 may trigger the download
        of the model which is approximately 10GB.
    graph:
        If True, returned datasets are DNSEDataset instances based on a
        structure/sequence graph. Edges are directed and featurized; see graph_* and
        ..graph.get_graph
    graph_sequence_window_size:
        When graph is True, passed to ..graph.get_graph. Defines how close two
        residues must be in primary sequence to gain a sequence-based
        connection.
    graph_n_distance_feats:
        When graph is True, passed to ..graph.get_graph.  Distance in sequence
        and 3D space are expanded using a set of radial basis functions. This
        argument controls the number of basis functions.
    graph_distance_cutoff:
        When graph is True, passed to ..graph.get_graph. When two residues are
        within this distance (in Angstroms), they are connected via a distance-based
        contact.
    parent_path:
        Path object specifying where to look for csv (and if specified, structure)
        files.
    include_test:
        If True, we return 3 datasets: train, validation, and test. If false,
        only train and validation are returned.

    Returns:
    -------
    If graph is False, (2 or 3)-Tuple of TensorDatasets (train, val) on the specified
    device.  TensorDatasets return (feat, signal, dataset_index) during iteration. Else,
    2-3 DMSE datasets of the same data coupled with a structure graph. See include_test.

    """
    if feat_type not in ("integer", "onehot", "t5"):
        raise ValueError("Only integer, onehot, or t5 featurization is supported.")

    if train_specs is None:
        train_specs = DATA_SPECS

    if val_specs is None:
        val_specs = DATA_SPECS

    if test_specs is None:
        test_specs = DATA_SPECS

    train_frame = _get_aggregate_mave_csv(
        specs=train_specs, identifier="train_filename", directory=parent_path
    )

    valid_frame = _get_aggregate_mave_csv(
        specs=val_specs, identifier="valid_filename", directory=parent_path
    )

    all_frames = [train_frame, valid_frame]

    if include_test:
        test_frame: Optional[pd.DataFrame] = _get_aggregate_mave_csv(
            specs=test_specs, identifier="test_filename", directory=parent_path
        )
        all_frames.append(test_frame)

    enc = get_default_int_encoder()

    # make sure that there are no amino acids in the data not in our standard
    # alphabet. We use a standard alphabet to maintain featurization stability
    # across possibly smaller input datasets.
    alpha = get_alphabet(pd.concat(all_frames), SEQ_CNAME)
    if not set(alpha).issubset(set(enc.alphabet)):
        raise ValueError("Data contains residues not represented fixed alphabet.")

    train_encoded, train_signal, train_dset_id = _process_table(
        train_frame,
        feat_type=feat_type,
        int_encoder=enc,
        device=device,
    )

    valid_encoded, valid_signal, valid_dset_id = _process_table(
        valid_frame,
        feat_type=feat_type,
        int_encoder=enc,
        device=device,
    )

    if include_test:
        test_encoded, test_signal, test_dset_id = _process_table(
            test_frame,
            feat_type=feat_type,
            int_encoder=enc,
            device=device,
        )

    if graph:
        edge_labels, edge_features = get_graph(
            structure=str(parent_path / SARSCOV2_FILENAME),
            max_cutoff=graph_distance_cutoff,
            min_cutoff=0.0,
            num_distance_features=graph_n_distance_feats,
            window_size=graph_sequence_window_size,
            node_offset=0,
        )

        train_dataset: Dataset = DNSEDataset(
            edge_attr=edge_features.to(device),
            edge_index=edge_labels.to(device),
            x=train_encoded.to(device),
            y=train_signal.to(device),
            experiment=train_dset_id.to(device),
        )
        valid_dataset: Dataset = DNSEDataset(
            edge_attr=edge_features.to(device),
            edge_index=edge_labels.to(device),
            x=valid_encoded.to(device),
            y=valid_signal.to(device),
            experiment=valid_dset_id.to(device),
        )

        if include_test:
            test_dataset: Dataset = DNSEDataset(
                edge_attr=edge_features.to(device),
                edge_index=edge_labels.to(device),
                x=test_encoded.to(device),
                y=test_signal.to(device),
                experiment=test_dset_id.to(device),
            )
    else:
        train_dataset = TensorDataset(
            train_encoded.to(device), train_signal.to(device), train_dset_id.to(device)
        )
        valid_dataset = TensorDataset(
            valid_encoded.to(device), valid_signal.to(device), valid_dset_id.to(device)
        )
        if include_test:
            test_dataset = TensorDataset(
                test_encoded.to(device), test_signal.to(device), test_dset_id.to(device)
            )

    if include_test:
        return train_dataset, valid_dataset, test_dataset
    else:
        return train_dataset, valid_dataset
