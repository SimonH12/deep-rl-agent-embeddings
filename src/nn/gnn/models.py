import torch
import torch.nn as nn
import torch.nn.functional as F
from sympy.logic.boolalg import Boolean
from torch_geometric.nn import SAGEConv, GATConv, GATv2Conv
from src.nn.gnn.util import GraphDataLoader
from abc import ABC, abstractmethod

class GNN(nn.Module, ABC):
    """
    Parent class for GNN models.
    """
    def __init__(self, device, num_features, hidden_channels, encode: Boolean = True):
        super(GNN, self).__init__()
        self.device = device
        self.to(self.device)
        self.graph_loader = GraphDataLoader(device)

        if encode:
            self.mlp_in = nn.Sequential(
                nn.Linear(num_features, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels)
            )
        else:
            self.mlp_in = nn.Sequential(
                nn.Linear(num_features, hidden_channels),
            )

    def forward(self, X):
        orig_shape = X.shape
        if len(X.shape) == 4:
            X = X.reshape(-1, orig_shape[-2], orig_shape[-1])

        batch_size, num_vectors, _ = X.shape

        data = self.graph_loader.process_batch(X)
        data = data.to(self.device)

        # data before Data(x=[30000, 16], edge_index=[2, 120000], batch=[30000])
        out = self.forward_graph(data)  # shape: (batch_size * num_vectors, out_dim)
        # out has shape (30000,1)
        out = out.view(batch_size, num_vectors, -1)
        # out has shape (6000, 5, 1)

        if len(orig_shape) == 4:
            out = out.view(orig_shape[0], orig_shape[1], out.shape[-2], out.shape[-1])

        return out

    @abstractmethod
    def forward_graph(self, data):
        """
        Forward pass for a single graph.
        """
        pass


class GSAGE(GNN):
    """
    Basic GraphSAGE model with one SAGE layer.
    Paper: https://arxiv.org/abs/1706.02216
    Maybe look at https://github.com/pyg-team/pytorch_geometric/blob/4e70e5d159ef0ee0f2a5c9129b560ef227e9147f/examples/graph_sage_unsup.py#L24 as a reference implementation.
    """
    def __init__(self, num_features, hidden_channels, out_channels, device, pooling_method, encode):
        super(GSAGE, self).__init__(device, num_features, hidden_channels, encode)
        # add mlp before conv layers to increase feature dimension

        self.sage = SAGEConv(hidden_channels, hidden_channels, aggr=pooling_method)
        self.norm = nn.LayerNorm(hidden_channels)

        self.mlp_out = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels)
        )


        self.to(self.device)

    def forward_graph(self, data):
        x = self.mlp_in(data.x)
        x = F.relu(self.norm(self.sage(x, data.edge_index)))
        x = self.mlp_out(x)
        return x


class GAT(GNN):
    """
    Graph Attention Network with GAT convolutional layers.
    Paper: https://arxiv.org/abs/1710.10903
    """
    def __init__(self, num_features, hidden_channels, out_channels, device, encode, heads=2, v2=False):
        super(GAT, self).__init__(device, num_features, hidden_channels, encode)

        if v2:
            self.gat = GATv2Conv(hidden_channels, hidden_channels, heads=heads, concat=True)
        else:
            self.gat = GATConv(hidden_channels, hidden_channels, heads=heads, concat=True)

        self.mlp_out = nn.Sequential(
            nn.Linear(heads * hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels)
        )

        self.to(self.device)

    def forward_graph(self, data):
        x = self.mlp_in(data.x)
        x = F.relu(self.gat(x, data.edge_index))
        x = self.mlp_out(x)
        return x


if __name__ == "__main__":
    batch_size = 4
    num_vectors = 3  # Can be arbitrary
    num_features = 2
    hidden_channels = 16
    out_channels = 4

    x = torch.rand(batch_size, num_vectors, num_features)  # Random input
    loader = GraphDataLoader(batch_size, num_vectors, num_features)
    data_list = loader.process_batch(x)

    device = torch.device("cpu")
    model = GAT(num_features, hidden_channels, out_channels, device)
    output = model(x)  # Shape: (batch_size, num_vectors, out_channels)
    print(x.shape)
    print(output.shape)
    print("--------------------------------")

    num_vectors = 10
    x = torch.rand(batch_size, num_vectors, num_features)  # Random input
    output = model(x)  # Shape: (batch_size, num_vectors, out_channels)
    print(x.shape)
    print(output.shape)
