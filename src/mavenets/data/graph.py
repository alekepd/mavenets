"""Specialzed tools for data loading for graph networks."""
from typing import Final, Literal, Tuple, List
from warnings import warn
import numpy as np
import itertools
from torch.utils.data import Dataset
from torch_geometric.data import Data  # type: ignore
import torch
import pandas as pd  # type: ignore
import mdtraj as md  # type: ignore

AMINO_ACIDS: Final = np.asarray(
    [
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
)

DEFAULT_CUT: Final = 1.5
DEFAULT_GAIN: Final = 75.0


def onehot_embed(sequence: np.ndarray) -> np.ndarray:
    """Featurize letter-sequence into one-hot representation.

    If input array is of shape (length,)Returned array is of shape (length,onehot_size).
    """
    return np.array([AMINO_ACIDS == s for s in sequence]).astype(np.float32)


def featurize_distances(
    distances: torch.Tensor,
    min_dist: float,
    max_dist: float,
    num_features: int,
    gain: float = 1.0,
) -> torch.Tensor:
    """Featurize distances using a Gaussian grid.

    Each distance is evaluated as input to multiple (unnormalized) Gaussian
    functions. The output of all of these functions is returned as a vector feature.
    The centers of each exponential are defined as a regularly spaced grid between
    min_dist and max_dist with num_features points.

    If num_features is set to 1, the input distances are returned unchanged.

    Arguments:
    ---------
    distances:
        tensor to featurize into distances. Can be any shape.
    min_dist:
        Start of grid.
    max_dist:
        End of grid.
    num_features:
        Number of grid points.
    gain:
        Width of each Gaussian function. Higher values correspond to
        smaller standard deviations.

    Returns:
    -------
    If num_features is greater than 0, a tensor with one additional dimension
    than the input tensor. New index is places on the right and indexes the
    added features. If 0, returns the input tensor.

    """
    if num_features == 0:
        return distances
    else:
        mean = torch.linspace(min_dist, max_dist, num_features)
        return torch.exp(-gain * (distances[..., None] - mean).pow(2))


def get_graph(
    structure: str,
    max_cutoff: float,
    min_cutoff: float,
    num_distance_features: int,
    window_size: int,
    scheme: Literal[
        "ca", "closest", "closest-heavy", "sidechain", "sidechain-heavy"
    ] = "ca",
    gain: float = DEFAULT_GAIN,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create a distance- and sequence- based graph from a molecular structure.

    Each residue defines a node, and edges between nodes are determined
    by both the distance between residues and the number of residues
    that separates them along the protein backbone.

    The edge feature is structured as: [ backbone_features : distance_features ].

    Both backbone and distance features are created by transforming the
    distance or displacement through a set of radial basis functions.

    Arguments:
    ---------
    structure:
        File descriptor readable by mdtraj containing an atomistic molecular
        configuration. Only the first frame is used.
    scheme:
        Passed to mdtraj via .compute_contacts to determine the distance
        between each residues.
    max_cutoff:
        If two residues are separated by a distance between min_cutoff
        and max_cutoff, they are given a distance feature. The *_cutoff
        variables also determine the radial basis function grid used to
        for distance featurization.
    min_cutoff:
        See max_cutoff.
    num_distance_features:
        Distance are featurized with the corresponding distance transformed
        into a vector feature with this many elements.
    window_size:
        An edge is given a backbone feature if the two residues are within
        this many residues. Implicitly controls the size of the feature
        vector through the size of the backbone feature.
    gain:
        Sharpness of the radial basis functions used in featurization.
        See featurize_distances.


    Returns:
    -------
    Tuple of two tensors. First contains pairs of nodes that have an edge
    (n_edges, 2) and second contains the features corresponding to that
    edge (n_edges, feature_size).

    Note:
    ----
    The topology of the graph is non-directed, but the featurization is directed
    due to the backbone features.

    """
    pdb = md.load_pdb(structure)
    reslist = list(range(pdb.n_residues))
    pairs = list(itertools.product(reslist, reslist))
    # contacts holds a (1, n_pairs) array of distances
    contacts, _ = md.compute_contacts(pdb, pairs, scheme=scheme)
    # distances holds a (1,n_res,n_res distance) matrix
    distances = torch.tensor(md.geometry.squareform(contacts, pairs))

    num_window_features = 2 * window_size + 1

    edge_feats: List[torch.Tensor] = []
    edges: List[Tuple[int, int]] = []

    # go through all pairs of residues and calculate backbone and
    # distance features. If both are outside a preset range,
    # we define the pair of residues to not have an edge.
    for i in range(pdb.n_residues):
        for j in range(pdb.n_residues):
            if i == j:
                continue
            use_edge = False
            feat = torch.zeros(num_distance_features + num_window_features)
            # We define sequence features if we are close enough
            # along primary sequence
            if np.abs(i - j) <= window_size:
                use_edge = True
                # this is not symmetric with respect to index swap
                feat[:num_window_features] = featurize_distances(
                    torch.tensor(j - i),
                    -window_size,
                    window_size,
                    num_window_features,
                    gain=gain,
                )

            # We define distance features if we are close enough
            # in 3D space
            dist_ij = distances[0][i][j]
            if dist_ij <= max_cutoff:
                use_edge = True
                feat[num_window_features:] = featurize_distances(
                    dist_ij, min_cutoff, max_cutoff, num_distance_features, gain=gain
                )
            # if neither class of feature is present, dont include the edge.
            if use_edge:
                edges.append((i, j))
                edge_feats.append(feat)
    tensor_edges = torch.tensor(np.transpose(edges))
    tensor_edge_feats = torch.stack(edge_feats, dim=0)
    return tensor_edges, tensor_edge_feats


class LegacyGraphDataReader(Dataset):
    """Dataset for serving structure SARS-COV molecule graphs.

    Serves one-hot versions of a subset of residues (positions 5-187 inclusive). This
    class contains hardcoded behavior and inconsistent usage of arguments. It is
    retained here to support legacy code.

    A single graph topology is created and served for all items. This graph is created
    using both sequence and distance information. See get_graph for details.
    """

    def __init__(
        self,
        base_dir: str,
        structure_file: str,
        num_distance_features: int,
        window_size: int,
        dataset: Literal["train", "test", "valid"],
        flattened: bool = False,
        graph_cutoff: float = DEFAULT_CUT,
    ) -> None:
        """Read all data from disk, create the structure graph, and store options.

        Arguments:
        ---------
        base_dir:
            CSV files are searched for in a subdirectory called "raw" in this directory.
        structure_file:
            Filename of topology-containing input to mdtraj.load used to define the
            3D distances for graph creation.
        num_distance_features:
            Size of vectorial featurization to use to process 3D distance information.
            Note that this is not the size of the overall edge features, as they contain
            both 3D and sequence distance information.
        window_size:
            Primary sequence distance cutoff used for sequence-based graph creation.
        dataset:
            A string used to define the name of the CSV to read for sequence and
            target value information.
        flattened:
            Stored as an same-named attribute, but unused.
        graph_cutoff:
            3D Distance cutoff used for structure-based graph creation.

        """
        warn("Using deprecated data reader for graphs.", stacklevel=1)
        super().__init__()
        self.x = self._get_data(base_dir, dataset)[0]
        self.y = self._get_data(base_dir, dataset)[1]
        self.flattened = flattened  # this doesn't seem to be used anywhere
        self.dataset = dataset
        # WT graph
        graph = get_graph(
            structure_file,
            max_cutoff=graph_cutoff,
            min_cutoff=0.0,
            num_distance_features=num_distance_features,
            window_size=window_size,
        )
        self.edge_index = graph[0]
        self.edge_feat = graph[1]

    def _get_data(self, base_dir: str, dataset: str) -> Tuple[np.ndarray, np.ndarray]:
        """Read csv data and return as numpy arrays.

        Note:
        ----
        csv location is partially hardcoded as "{base_dir}/raw/{dataset}_data_short.csv"

        Arguments:
        ---------
        base_dir:
            Parent directory for finding csv. CSVs are searched for in the "raw"
            subdirectory.
        dataset:
            string used to define name of csv to read.

        Returns:
        -------
        2-Tuple, first entry is sequences, second is target signal.

        """
        df = pd.read_csv(
            "{}/raw/{}_data_short.csv".format(base_dir, dataset), header=None
        )
        x = np.asarray([df[1].values])[0]
        y = np.asarray([df[2].values]).reshape(-1, 1).astype(np.float32)

        return x, y

    def __len__(self) -> int:
        """Return number of data pairs."""
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> Data:
        """Index array, isolate subsection, one-hot it, serve in Data object."""
        seq = self.x[idx]
        shortened_seq = seq[5:188]
        onehot = onehot_embed(shortened_seq)

        data = Data(
            x=torch.tensor(onehot, dtype=torch.float),
            y=torch.tensor(self.y[idx], dtype=torch.float),
            edge_attr=self.edge_feat,
            edge_index=self.edge_index,
        )

        return data
