import torch
import torch.nn as nn
import torch.nn.functional as F


class AgentTransformer(nn.Module):
    """
    Transformer-based model for processing agent states with support for padding
    to handle variable numbers of agents.

    Args:
        in_features (int): Dimension of input agent states.
        out_features (int): Dimension of output predictions.
        max_num_agents (int): Maximum number of agents the model can handle.
        depth (int): Number of layers in the Transformer encoder.
        model_dim (int): Dimension of the model.
        num_heads (int): Number of attention heads in the Transformer encoder.
        dropout (float): Dropout probability in the Transformer encoder.
        device (torch.device): Device on which the model is to be run.
    """

    def __init__(self, in_features, out_features, max_num_agents=10, model_dim=32, num_heads=2, depth=1, dropout=0.1,
                 device="cpu"):
        super(AgentTransformer, self).__init__()
        self.device = device
        self.max_num_agents = max_num_agents

        # Input projection
        self.input_projection = nn.Linear(in_features, model_dim)


        # --------- Permutation Invariance to the Order of Agents --------- #
        # Transformer Encoder (without positional encoding)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout,
                                                   batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth, enable_nested_tensor=False)
        # ----------------------------------------------------------------- #

        # Output layer
        self.output_layer = nn.Linear(model_dim, out_features)
        self.to(self.device)

    def _make_attention_mask(self, batch_size, num_agents, device):
        mask = torch.zeros((batch_size, self.max_num_agents), dtype=torch.float32, device=device)
        mask[:, :num_agents] = 1.0
        return mask

    def forward(self, x):
        """
        Args:
            x: torch.Tensor, shape (batch_size, num_agents, in_features)
        Returns:
            torch.Tensor: shape (batch_size, out_features)
        """
        batch_size, num_agents, in_features = x.shape

        assert num_agents <= self.max_num_agents, "Number of agents exceeds maximum allowed."


        # --------- ensures that the model can handle variable numbers of agents by padding ---------- #
        # Pad inputs if num_agents < max_num_agents
        if num_agents < self.max_num_agents:
            padded = x.new_zeros((batch_size, self.max_num_agents, in_features), device=x.device)
            padded[:, :num_agents] = x
            x = padded

        # Create attention mask to ignore padded positions
        attention_mask = self._make_attention_mask(batch_size, num_agents, device=x.device)
        attention_mask[:, :num_agents] = 1.0  # True for actual agents, False for padding
        # -------------------------------------------------------------------------------------------- #

        # Each row in x represents a single agent’s state, which is treated as a distinct token in the Transformer
        x = self.input_projection(x)  # Project input to model dimension

        # Apply Transformer encoder with mask
        x = self.transformer_encoder(x, src_key_padding_mask=(attention_mask==0))  # Mask false (padded) positions

        # --------- Permutation Invariance to the Order of Agents --------- #
        # Mean pooling over the agent dimension
        x = (x * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        # ----------------------------------------------------------------- #

        # Output projection
        return self.output_layer(x)



# Example usage
if __name__ == "__main__":
    torch.manual_seed(0)
    batch_size = 32
    num_agents = 5
    max_num_agents = 10
    in_features = 18
    model_dim = 64
    num_heads = 4
    num_layers = 2
    out_features = 18
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model = AgentTransformer(in_features, out_features, max_num_agents, model_dim, num_heads, num_layers,
                                device=device)

    # Example input with fewer agents than max_num_agents
    sample_input = torch.randn(batch_size, num_agents, in_features).to(device)
    output = model(sample_input)

    print("Input shape:", sample_input.shape)  # (batch_size, num_agents, in_features)
    print("Output shape:", output.shape)  # (batch_size, out_features)
