"""Message passing neural network."""

from typing import Final, Any
import torch
import torch.nn as nn
from torch_scatter import scatter, scatter_mean  # type: ignore
from einops.layers import torch as einops_torch
from torch_geometric.loader import DataLoader as pygDataLoader  # type: ignore
from torch_geometric.data import Data  # type: ignore
from ..data import LegacyGraphDataReader


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
            #message, target, reduce="mean", dim=0, dim_size=target.max() + 1
            message, target, reduce="mean", dim=0, dim_size=x.shape[0]
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


def _legacy_example() -> Any:
    """Create model."""
    BASE_DIR: Final = "/group/ag_cmb/scratch/tsztain/Bloom/"
    STRUCTURE_FILE: Final = (
        "{}/structure_prep/TRAJECTORIES_ace2_rbd_prot_glyc_memb_amarolab"
        "/RBD_amaro.pdb".format(BASE_DIR)
    )

    NUM_MESSAGES: Final = 7

    WINDOW_SIZE: Final = 10
    NUM_DISTANCE_FEATS: Final = 10

    BATCH_SIZE: Final = 32
    # we are using a subset of the sequence
    NUM_NODES: Final = 183
    NUM_HIDDEN: Final = 128
    NUM_FEATURES: Final = 20

    NUM_EDGE_FEATS: Final = (2 * WINDOW_SIZE + 1) + NUM_DISTANCE_FEATS

    # unused -> LR: Final = 1e-4
    # unused -> NUM_EPOCHS: Final = 300

    DEVICE: Final = "cuda" if torch.cuda.is_available() else "cpu"

    train_data, test_data, valid_data = [
        LegacyGraphDataReader(
            base_dir=BASE_DIR,
            structure_file=STRUCTURE_FILE,
            num_distance_features=NUM_DISTANCE_FEATS,
            window_size=WINDOW_SIZE,
            dataset=dataset, #type: ignore
            flattened=True,
        )
        for dataset in ("train", "test", "valid")
    ]
    train_dl, test_dl, valid_dl = [
        pygDataLoader(data, shuffle=True, batch_size=BATCH_SIZE)
        for data in (train_data, test_data, valid_data)
    ]

    model = GraphNet(
        201, NUM_FEATURES, NUM_HIDDEN, NUM_MESSAGES, NUM_EDGE_FEATS
    )

    return model, train_data, train_dl
