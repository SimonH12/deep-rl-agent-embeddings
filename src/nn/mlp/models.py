import torch
import torch.nn as nn

class MLPLocal(nn.Module):
    def __init__(self, embedding_mlp, pooling_method):
        super().__init__()
        self.embedding_mlp = embedding_mlp
        self.pooling_method = pooling_method

    def forward(self, inputs):
        embeddings = torch.stack([self.embedding_mlp(inputs[:, i, :]) for i in range(inputs.shape[1])], dim=1)
        global_embeddings = torch.stack([
            self.pooling_method(torch.cat([embeddings[:, :i, :], embeddings[:, i+1:, :]], dim=1), dim=1)
            for i in range(inputs.shape[1])
        ], dim=1)
        return torch.cat([inputs, global_embeddings], dim=-1)

class MLPGlobal(nn.Module):
    def __init__(self, embedding_mlp: nn.Module, pooling_method):
        super().__init__()
        self.embedding_mlp = embedding_mlp
        self.pooling_method = pooling_method

    def forward(self, inputs):
        # inputs: (B, L, D)
        embedding = self.embedding_mlp(inputs)  # (B, L, H)
        pooled = self.pooling_method(embedding, dim=1)  # (B, H)
        expanded = pooled.unsqueeze(1).expand(-1, inputs.size(1), -1)  # (B, L, H)
        return torch.cat([inputs, expanded], dim=-1)