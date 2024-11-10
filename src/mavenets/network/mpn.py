"""Message passing neural network."""

from typing import Final, Tuple, Literal, List
import itertools
import torch
import numpy as np
import mdtraj as md  # type: ignore
import torch.nn as nn
from torch_scatter import scatter, scatter_mean  # type: ignore
from einops.layers import torch as einops_torch
from torch_geometric.data import Data  # type: ignore
from torch_geometric.loader import DataLoader  # type: ignore
from torch.utils.data import Dataset
import pandas as pd  # type: ignore

BASE_DIR: Final = "/group/ag_cmb/scratch/tsztain/Bloom/"
STRUCTURE: Final = (
    "{}/structure_prep/TRAJECTORIES_ace2_rbd_prot_glyc_memb_amarolab"
    "/RBD_amaro.pdb".format(BASE_DIR)
)

DEFAULT_CUT: Final = 1.5
DEFAULT_GAIN: Final = 75.0
DEFAULT_NUM_DISTANCE_FEATS: Final = 10
DEFAULT_WINDOW_SIZE: Final = 10
DEFAULT_NUM_EDGE_FEATS: Final = (
    2 * DEFAULT_WINDOW_SIZE + 1
) + DEFAULT_NUM_DISTANCE_FEATS


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
    structure: md.Trajectory,
    scheme: Literal[
        "ca", "closest", "closest-heavy", "sidechain", "sidechain-heavy"
    ] = "ca",
    max_cutoff: float = 1.5,
    min_cutoff: float = 0.0,
    num_distance_features: int = DEFAULT_NUM_DISTANCE_FEATS,
    window_size: int = DEFAULT_WINDOW_SIZE,
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
        mdtraj.Trajectory containing an atomistic molecular configuration.
    scheme:
        Passed to mdtraj via .compute_contacts to determine the distance
        between each residues
    max_cutoff:
        If two residues are separated by a distance between min_cutoff
        and max_cutoff, the are given a distance feature. The *_cutoff
        variables also determine the radial basis function grid used to
        for distance featurization.
    min_cutoff:
        See max_cutoff
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
            feat = torch.zeros(num_distance_features)
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


def onehot_embed(sequence: np.ndarray) -> np.ndarray:
    """Featurize letter-sequence into one-hot representation.

    If input array is of shape (length,)Returned array is of shape (length,onehot_size).
    """
    return np.array([AMINO_ACIDS == s for s in sequence]).astype(np.float32)


class Message(nn.Module):
    """Performs a single message passing step.

    Messages are computing using a MLP with two hidden layers.
    """

    def __init__(
        self, num_nodes: int, num_features: int, num_hidden: int, num_edge_feats: int
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.tot_num_features = num_features * 2 + num_edge_feats
        self.message = nn.Sequential(
            nn.Linear(self.tot_num_features, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_features),
        )

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_feat: torch.Tensor
    ) -> torch.Tensor:
        # nodes -> (num_nodes, num_feats)
        # edges -> (2, num_edges)

        res = x

        source = edge_index[0]
        target = edge_index[1]

        # shape ->  (batch * edge_idx, feat)
        edge = torch.cat([x[source], x[target], edge_feat], dim=-1)

        # concatenate the edge features to `edge`

        # shape -> (batch, num_edges = (num_nodes * num_nodes), num_feats)
        message = self.message(edge)
        update = scatter(
            message, target, reduce="mean", dim=0, dim_size=target.max() + 1
        )

        x = res + update

        return x


class GraphNet(nn.Module):
    """Message passing network.

    Input graphs have node and edge features.
    """

    def __init__(
        self,
        num_nodes: int,
        num_features: int,
        num_hidden: int,
        num_messages: int,
        num_edge_feats: int,
    ) -> None:
        """Initialize layers.

        Arguments:
        ---------
        num_nodes:
            Number of nodes in graphs.
        num_features:
            Dimensionality of node features.
        num_hidden:
            Number of hidden layers in each message passing step (see Message).
        num_messages:
            Number of message passing steps.
        num_edge_feats:
            Dimensionality of edge features.

        """
        super().__init__()
        self.messages = nn.ModuleList(
            [
                Message(num_nodes, num_features, num_hidden, num_edge_feats)
                for _ in range(num_messages)
            ]
        )
        self.reduction = nn.Sequential(
            einops_torch.Rearrange("(b n) f->b (n f)", n=num_nodes),
            nn.Linear(num_nodes * num_features, 1),
        )

    def forward(self, data: Data) -> torch.Tensor:
        """Predict on batch of graphs."""
        x, edge_index, edge_feat = data.x, data.edge_index, data.edge_attr
        for message in self.messages:
            x = message(x, edge_index, edge_feat)
        x = scatter_mean(x, data.batch, dim=0)
        return x.sum(dim=-1)


class DataReader(Dataset):
    """Dataset for serving structure graphs.

    Serves one-hot versions of a subset of residues (positions 5-187 inclusive).
    """

    def __init__(
        self,
        base_dir: str,
        dataset: Literal["train", "test", "valid"],
        flattened: bool = False,
    ) -> None:
        """Read all data from disk, create the structure graph, and store options."""
        super().__init__()
        self.x = self._get_data(base_dir, dataset)[0]
        self.y = self._get_data(base_dir, dataset)[1]
        self.flattened = flattened  # this doesn't seem to be used anywhere
        self.dataset = dataset
        # WT graph
        graph = get_graph(STRUCTURE, max_cutoff=DEFAULT_CUT)
        self.edge_index = graph[0]
        self.edge_feat = graph[1]

    def _get_data(self, base_dir: str, dataset: str) -> Tuple[np.ndarray, np.ndarray]:
        """Read csv data and return as numpy arrays."""
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


def main() -> nn.Module:
    """Create model."""
    NUM_MESSAGES: Final = 7

    BATCH_SIZE: Final = 32
    # we are using a subset of the sequence
    NUM_NODES: Final = 183
    NUM_HIDDEN: Final = 128
    NUM_FEATURES: Final = 20

    # unused -> LR: Final = 1e-4
    # unused -> NUM_EPOCHS: Final = 300

    DEVICE: Final = "cuda" if torch.cuda.is_available() else "cpu"

    train_data, test_data, valid_data = [
        DataReader(BASE_DIR, dataset, flattened=True)  # type: ignore
        for dataset in ("train", "test", "valid")
    ]
    train_dl, test_dl, valid_dl = [
        DataLoader(data, shuffle=True, batch_size=BATCH_SIZE)
        for data in (train_data, test_data, valid_data)
    ]

    model = GraphNet(
        NUM_NODES, NUM_FEATURES, NUM_HIDDEN, NUM_MESSAGES, DEFAULT_NUM_EDGE_FEATS
    ).to(DEVICE)

    return model


if __name__ == "__main":
    main()
