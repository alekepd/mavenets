"""Tools for loading data."""

from typing import Final, Tuple, Iterable, Literal, Union, Dict
import pandas as pd  # type: ignore
from torch.utils.data import TensorDataset
from torch import Tensor, tensor, float32, int32, int64
from torch.nn.functional import one_hot
from torch.utils.data import Dataset
from pathlib import Path
from torch_geometric.data import Data  # type: ignore
from .spec import DATA_SPECS, DataSpec, resolve_dataspec, SARSCOV2_FILENAME
from .featurize import get_alphabet, get_default_int_encoder
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


def get_datasets(
    device: str,
    train_specs: Union[None, Iterable[DataSpec], Iterable[int], Iterable[str]] = None,
    val_specs: Union[None, Iterable[DataSpec], Iterable[int], Iterable[str]] = None,
    feat_type: Literal["integer", "onehot"] = "integer",
    graph: bool = False,
    graph_sequence_window_size: int = 10,
    graph_n_distance_feats: int = 10,
    graph_distance_cutoff: float = 2.5,
    parent_path: Path = Path(),
) -> Tuple[Dataset, Dataset]:
    """Load, featurize, and return SARSCOV2 data for training and evaluation.

    Loads target signal and sequences from disk, and if graph is specified reads
    a file describing the 3d structure of the protein. If graph is True, a Tuple of 
    two DNSEDataset is returned; else, two TensorDatasets are returned, first being
    the train data and second the validation data.

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

    Returns:
    -------
    If graph is False, 2-Tuple of TensorDatasets (train, val) on the specified device.
    TensorDatasets return (feat, signal, dataset_index) during iteration. Else,
    pair of DMSE datasets of the same data coupled with a structure graph.

    """
    if feat_type not in ("integer", "onehot"):
        raise ValueError("Only integer or onehot featurization is supported.")

    if train_specs is None:
        train_specs = DATA_SPECS

    if val_specs is None:
        val_specs = DATA_SPECS

    train_frames = []

    # load training datasets, record index
    for sp in (resolve_dataspec(x) for x in train_specs):
        tf = _mave_csv_read(parent_path / sp.train_filename)
        tf[EXPERIMENT_CNAME] = sp.index
        train_frames.append(tf)

    train_frame = pd.concat(train_frames)

    valid_frames = []

    # load validation datasets, record index
    for sp in (resolve_dataspec(x) for x in val_specs):
        vf = _mave_csv_read(parent_path / sp.valid_filename)
        vf[EXPERIMENT_CNAME] = sp.index
        valid_frames.append(vf)

    valid_frame = pd.concat(valid_frames)

    enc = get_default_int_encoder()

    # make sure that there are no amino acids in the data not in our standard
    # alphabet. We use a standard alphabet to maintain featurization stability
    # across possibly smaller input datasets.
    alpha = get_alphabet(pd.concat([train_frame, valid_frame]), SEQ_CNAME)
    if not set(alpha).issubset(set(enc.alphabet)):
        raise ValueError("Data contains residues not represented fixed alphabet.")

    train_int_encoded = enc.batch_encode(train_frame.loc[:, SEQ_CNAME])
    if feat_type == "onehot":
        # we must cast to int64 to make torch happy
        train_encoded = one_hot(
            train_int_encoded.to(int64), num_classes=len(enc.alphabet)
        ).to(float32)
    else:
        train_encoded = train_int_encoded

    train_signal = tensor(train_frame.loc[:, SIGNAL_CNAME].to_numpy(), dtype=float32)
    train_dset_id = tensor(train_frame.loc[:, EXPERIMENT_CNAME].to_numpy(), dtype=int32)

    valid_int_encoded = enc.batch_encode(valid_frame.loc[:, SEQ_CNAME])
    if feat_type == "onehot":
        # we must cast to int64 to make torch happy
        valid_encoded = one_hot(
            valid_int_encoded.to(int64), num_classes=len(enc.alphabet)
        ).to(float32)
    else:
        valid_encoded = valid_int_encoded
    valid_signal = tensor(valid_frame.loc[:, SIGNAL_CNAME].to_numpy(), dtype=float32)
    valid_dset_id = tensor(valid_frame.loc[:, EXPERIMENT_CNAME].to_numpy(), dtype=int32)

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
    else:
        train_dataset = TensorDataset(
            train_encoded.to(device), train_signal.to(device), train_dset_id.to(device)
        )
        valid_dataset = TensorDataset(
            valid_encoded.to(device), valid_signal.to(device), valid_dset_id.to(device)
        )

    return train_dataset, valid_dataset
