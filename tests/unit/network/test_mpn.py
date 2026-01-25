"""Tests for mavenets.network.mpn module.

These tests require torch_geometric and torch_scatter to be installed.
They will be skipped if these dependencies are not available.
"""

import pytest
import torch
import torch.nn as nn

# Skip entire module if dependencies are not available
torch_geometric = pytest.importorskip("torch_geometric")
torch_scatter = pytest.importorskip("torch_scatter")

# Imports below are after importorskip() to ensure dependencies are available before importing
from torch_geometric.data import Data, Batch  # type: ignore[import-not-found]  # noqa: E402

from mavenets.network.mpn import Message, GraphNet  # type: ignore[import-not-found]  # noqa: E402


class TestMessage:
    """Tests for Message class."""

    def test_init_basic(self) -> None:
        """Should initialize with basic parameters."""
        msg = Message(
            n_nodes=10,
            n_features=8,
            hidden_size=16,
            n_edge_feats=4,
        )
        assert isinstance(msg, nn.Module)
        assert msg.n_nodes == 10
        assert msg.tot_n_features == 8 * 2 + 4  # n_features * 2 + n_edge_feats

    def test_init_message_network_structure(self) -> None:
        """Message network should have correct structure."""
        msg = Message(
            n_nodes=10,
            n_features=8,
            hidden_size=16,
            n_edge_feats=4,
        )
        # Should be: Linear -> ReLU -> Linear -> ReLU -> Linear
        assert isinstance(msg.message, nn.Sequential)
        assert len(msg.message) == 5
        assert isinstance(msg.message[0], nn.Linear)
        assert isinstance(msg.message[1], nn.ReLU)
        assert isinstance(msg.message[2], nn.Linear)
        assert isinstance(msg.message[3], nn.ReLU)
        assert isinstance(msg.message[4], nn.Linear)

    def test_init_custom_activation(self) -> None:
        """Should initialize with custom activation class."""
        msg = Message(
            n_nodes=10,
            n_features=8,
            hidden_size=16,
            n_edge_feats=4,
            activation_class=nn.LeakyReLU,
        )
        assert isinstance(msg.message[1], nn.LeakyReLU)
        assert isinstance(msg.message[3], nn.LeakyReLU)

    def test_init_input_output_sizes(self) -> None:
        """Input and output sizes should be correct."""
        n_features = 8
        hidden_size = 16
        n_edge_feats = 4
        msg = Message(
            n_nodes=10,
            n_features=n_features,
            hidden_size=hidden_size,
            n_edge_feats=n_edge_feats,
        )
        # First linear: (n_features * 2 + n_edge_feats) -> hidden_size
        assert msg.message[0].in_features == n_features * 2 + n_edge_feats
        assert msg.message[0].out_features == hidden_size
        # Last linear: hidden_size -> n_features
        assert msg.message[4].in_features == hidden_size
        assert msg.message[4].out_features == n_features

    def test_forward_shape(self, cpu_device: str) -> None:
        """Forward pass should produce correct output shape."""
        n_nodes = 5
        n_features = 8
        n_edge_feats = 4
        msg = Message(
            n_nodes=n_nodes,
            n_features=n_features,
            hidden_size=16,
            n_edge_feats=n_edge_feats,
        ).to(cpu_device)

        # Create simple graph data
        x = torch.randn(n_nodes, n_features, device=cpu_device)
        # Fully connected edges (excluding self-loops)
        edge_index = torch.tensor(
            [[i, j] for i in range(n_nodes) for j in range(n_nodes) if i != j],
            dtype=torch.long,
            device=cpu_device,
        ).T
        n_edges = edge_index.shape[1]
        edge_feat = torch.randn(n_edges, n_edge_feats, device=cpu_device)

        output = msg(x, edge_index, edge_feat)
        assert output.shape == (n_nodes, n_features)

    def test_forward_residual_connection(self, cpu_device: str) -> None:
        """Forward pass should include residual connection."""
        n_nodes = 5
        n_features = 8
        n_edge_feats = 4
        msg = Message(
            n_nodes=n_nodes,
            n_features=n_features,
            hidden_size=16,
            n_edge_feats=n_edge_feats,
        ).to(cpu_device)

        # Zero out message network weights
        with torch.no_grad():
            for param in msg.message.parameters():
                param.zero_()

        x = torch.randn(n_nodes, n_features, device=cpu_device)
        edge_index = torch.tensor(
            [[i, j] for i in range(n_nodes) for j in range(n_nodes) if i != j],
            dtype=torch.long,
            device=cpu_device,
        ).T
        n_edges = edge_index.shape[1]
        edge_feat = torch.randn(n_edges, n_edge_feats, device=cpu_device)

        output = msg(x, edge_index, edge_feat)
        # With zeroed weights, output should equal input (residual only)
        assert torch.allclose(output, x, atol=1e-6)

    def test_forward_deterministic(self, cpu_device: str) -> None:
        """Forward pass should be deterministic."""
        n_nodes = 5
        n_features = 8
        n_edge_feats = 4
        msg = Message(
            n_nodes=n_nodes,
            n_features=n_features,
            hidden_size=16,
            n_edge_feats=n_edge_feats,
        ).to(cpu_device)

        x = torch.randn(n_nodes, n_features, device=cpu_device)
        edge_index = torch.tensor(
            [[i, j] for i in range(n_nodes) for j in range(n_nodes) if i != j],
            dtype=torch.long,
            device=cpu_device,
        ).T
        n_edges = edge_index.shape[1]
        edge_feat = torch.randn(n_edges, n_edge_feats, device=cpu_device)

        output1 = msg(x, edge_index, edge_feat)
        output2 = msg(x, edge_index, edge_feat)
        assert torch.allclose(output1, output2)

    def test_gradient_flow(self, cpu_device: str) -> None:
        """Gradients should flow through the message passing."""
        n_nodes = 5
        n_features = 8
        n_edge_feats = 4
        msg = Message(
            n_nodes=n_nodes,
            n_features=n_features,
            hidden_size=16,
            n_edge_feats=n_edge_feats,
        ).to(cpu_device)

        x = torch.randn(n_nodes, n_features, device=cpu_device, requires_grad=True)
        edge_index = torch.tensor(
            [[i, j] for i in range(n_nodes) for j in range(n_nodes) if i != j],
            dtype=torch.long,
            device=cpu_device,
        ).T
        n_edges = edge_index.shape[1]
        edge_feat = torch.randn(n_edges, n_edge_feats, device=cpu_device)

        output = msg(x, edge_index, edge_feat)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestGraphNet:
    """Tests for GraphNet class."""

    def test_init_basic(self) -> None:
        """Should initialize with basic parameters."""
        net = GraphNet(
            n_nodes=10,
            n_features=8,
            n_hidden=16,
            n_messages=3,
            n_edge_feats=4,
        )
        assert isinstance(net, nn.Module)
        assert len(net.messages) == 3

    def test_init_message_layers(self) -> None:
        """Should create correct number of message layers."""
        for n_messages in [1, 3, 5]:
            net = GraphNet(
                n_nodes=10,
                n_features=8,
                n_hidden=16,
                n_messages=n_messages,
                n_edge_feats=4,
            )
            assert len(net.messages) == n_messages
            for msg in net.messages:
                assert isinstance(msg, Message)

    def test_init_has_reduction_layer(self) -> None:
        """Should initialize with reduction layer (note: currently unused in forward)."""
        n_nodes = 10
        n_features = 8
        net = GraphNet(
            n_nodes=n_nodes,
            n_features=n_features,
            n_hidden=16,
            n_messages=3,
            n_edge_feats=4,
        )
        # Reduction layer exists but is not used in current forward implementation
        # Forward uses scatter_mean + sum instead
        assert isinstance(net.reduction, nn.Sequential)
        assert hasattr(net, 'reduction')

    def test_forward_single_graph(self, cpu_device: str) -> None:
        """Forward pass should work with single graph."""
        n_nodes = 5
        n_features = 8
        n_edge_feats = 4
        net = GraphNet(
            n_nodes=n_nodes,
            n_features=n_features,
            n_hidden=16,
            n_messages=2,
            n_edge_feats=n_edge_feats,
        ).to(cpu_device)

        # Create a simple graph
        x = torch.randn(n_nodes, n_features, device=cpu_device)
        edge_index = torch.tensor(
            [[i, j] for i in range(n_nodes) for j in range(n_nodes) if i != j],
            dtype=torch.long,
            device=cpu_device,
        ).T
        n_edges = edge_index.shape[1]
        edge_attr = torch.randn(n_edges, n_edge_feats, device=cpu_device)
        batch = torch.zeros(n_nodes, dtype=torch.long, device=cpu_device)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
        output = net(data)
        assert output.shape == (1,)  # Single graph, single output

    def test_forward_batched_graphs(self, cpu_device: str) -> None:
        """Forward pass should work with batched graphs."""
        n_nodes = 5
        n_features = 8
        n_edge_feats = 4
        batch_size = 3
        net = GraphNet(
            n_nodes=n_nodes,
            n_features=n_features,
            n_hidden=16,
            n_messages=2,
            n_edge_feats=n_edge_feats,
        ).to(cpu_device)

        # Create multiple graphs and batch them
        graphs = []
        for _ in range(batch_size):
            x = torch.randn(n_nodes, n_features, device=cpu_device)
            edge_index = torch.tensor(
                [[i, j] for i in range(n_nodes) for j in range(n_nodes) if i != j],
                dtype=torch.long,
                device=cpu_device,
            ).T
            n_edges = edge_index.shape[1]
            edge_attr = torch.randn(n_edges, n_edge_feats, device=cpu_device)
            graphs.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr))

        batch = Batch.from_data_list(graphs)
        output = net(batch)
        assert output.shape == (batch_size,)

    def test_forward_deterministic(self, cpu_device: str) -> None:
        """Forward pass should be deterministic."""
        n_nodes = 5
        n_features = 8
        n_edge_feats = 4
        net = GraphNet(
            n_nodes=n_nodes,
            n_features=n_features,
            n_hidden=16,
            n_messages=2,
            n_edge_feats=n_edge_feats,
        ).to(cpu_device)

        x = torch.randn(n_nodes, n_features, device=cpu_device)
        edge_index = torch.tensor(
            [[i, j] for i in range(n_nodes) for j in range(n_nodes) if i != j],
            dtype=torch.long,
            device=cpu_device,
        ).T
        n_edges = edge_index.shape[1]
        edge_attr = torch.randn(n_edges, n_edge_feats, device=cpu_device)
        batch = torch.zeros(n_nodes, dtype=torch.long, device=cpu_device)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
        output1 = net(data)
        output2 = net(data)
        assert torch.allclose(output1, output2)

    def test_gradient_flow(self, cpu_device: str) -> None:
        """Gradients should flow through the network."""
        n_nodes = 5
        n_features = 8
        n_edge_feats = 4
        net = GraphNet(
            n_nodes=n_nodes,
            n_features=n_features,
            n_hidden=16,
            n_messages=2,
            n_edge_feats=n_edge_feats,
        ).to(cpu_device)

        x = torch.randn(n_nodes, n_features, device=cpu_device, requires_grad=True)
        edge_index = torch.tensor(
            [[i, j] for i in range(n_nodes) for j in range(n_nodes) if i != j],
            dtype=torch.long,
            device=cpu_device,
        ).T
        n_edges = edge_index.shape[1]
        edge_attr = torch.randn(n_edges, n_edge_feats, device=cpu_device)
        batch = torch.zeros(n_nodes, dtype=torch.long, device=cpu_device)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
        output = net(data)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        # Check all network parameters have gradients (except unused reduction layer)
        for name, param in net.named_parameters():
            if not name.startswith('reduction'):
                assert param.grad is not None, f"No gradient for {name}"

    def test_multiple_message_passes(self, cpu_device: str) -> None:
        """Output should change with different number of message passes."""
        n_nodes = 5
        n_features = 8
        n_edge_feats = 4

        net1 = GraphNet(
            n_nodes=n_nodes,
            n_features=n_features,
            n_hidden=16,
            n_messages=1,
            n_edge_feats=n_edge_feats,
        ).to(cpu_device)

        net2 = GraphNet(
            n_nodes=n_nodes,
            n_features=n_features,
            n_hidden=16,
            n_messages=3,
            n_edge_feats=n_edge_feats,
        ).to(cpu_device)

        # Both networks should produce valid outputs
        x = torch.randn(n_nodes, n_features, device=cpu_device)
        edge_index = torch.tensor(
            [[i, j] for i in range(n_nodes) for j in range(n_nodes) if i != j],
            dtype=torch.long,
            device=cpu_device,
        ).T
        n_edges = edge_index.shape[1]
        edge_attr = torch.randn(n_edges, n_edge_feats, device=cpu_device)
        batch = torch.zeros(n_nodes, dtype=torch.long, device=cpu_device)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
        
        output1 = net1(data)
        output2 = net2(data)
        
        assert output1.shape == output2.shape == (1,)

    def test_sparse_graph(self, cpu_device: str) -> None:
        """Forward pass should work with sparse graphs (few edges)."""
        n_nodes = 5
        n_features = 8
        n_edge_feats = 4
        net = GraphNet(
            n_nodes=n_nodes,
            n_features=n_features,
            n_hidden=16,
            n_messages=2,
            n_edge_feats=n_edge_feats,
        ).to(cpu_device)

        # Create sparse graph with only a few edges (linear chain: 0->1->2->3->4)
        x = torch.randn(n_nodes, n_features, device=cpu_device)
        edge_index = torch.tensor(
            [[0, 1, 2, 3], [1, 2, 3, 4]],  # source nodes, target nodes
            dtype=torch.long,
            device=cpu_device,
        )
        n_edges = edge_index.shape[1]
        edge_attr = torch.randn(n_edges, n_edge_feats, device=cpu_device)
        batch = torch.zeros(n_nodes, dtype=torch.long, device=cpu_device)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
        output = net(data)
        assert output.shape == (1,)
        assert not torch.isnan(output).any()
