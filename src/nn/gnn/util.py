from torch_geometric.data import Data
import torch

class GraphDataLoader:
    def __init__(self, device=torch.device('cpu')):
        self.device = device

    @staticmethod
    def _create_edge_index(num_nodes):
        """
        Creates tensor with shape (2, num_deges):
            - edge_index[0]: source nodes (where the edge starts)
            - edge_index[1]: target nodes (where the edge ends)
        Each node is connected to all others except itself (no self loops)
            - GraphSAGE explicitly does not use self loops
            - GAT and GATv2 create the self loops themselves when using the layer
        """
        row, col = zip(*[(i, j) for i in range(num_nodes) for j in range(num_nodes) if i != j])
        edge_index = torch.tensor([row, col], dtype=torch.long)
        return edge_index

    def process_batch(self, X):
        """
        Converts the input tensor into a batched Data object.

        Args:
            X (Tensor): shape (batch_size, num_nodes, num_features)

        Returns:
            A `Batch` object representing the whole graph batch.
        """
        # assert X.shape == (self.batch_size, self.num_nodes, self.num_features)
        batch_size, num_nodes, num_features = X.shape # (6000, 5, 16)

        X = X
        x = X.reshape(-1, num_features)  # (batch_size * num_nodes, num_features) e.g. (30000, 16)

        # Batch assignment: [0 0 ... 0, 1 1 ... 1, ..., B-1 B-1 ... B-1]
        batch = torch.arange(batch_size, device=self.device).repeat_interleave(num_nodes)

        # Offset edge_index for each graph in the batch
        edge_index = GraphDataLoader._create_edge_index(num_nodes)
        edge_index = edge_index.unsqueeze(0) + (
            torch.arange(batch_size, device=self.device).unsqueeze(1).unsqueeze(2) * num_nodes
        )  # shape (batch_size, 2, num_edges)

        edge_index = edge_index.view(2, -1)  # Merge into one big edge_index of shape (2, total_edges)

        # Return one big Data object — PyG can handle batch-wise operations
        return Data(x=x, edge_index=edge_index, batch=batch)


# main
if __name__ == "__main__":
    # Let's say you have 4 nodes in a single graph
    num_nodes = 4

    # Call the static method directly from the class
    edge_index = GraphDataLoader._create_edge_index(num_nodes)

    # Print the result
    print("Edge Index:")
    print(edge_index)
