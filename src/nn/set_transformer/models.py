# Code originally from SetTransformer paper/GitHub

from src.nn.set_transformer.modules import *

class DeepSet(nn.Module):
    # NOTE: this is basically my mean_local the decoder network is my core mlp
    def __init__(self, dim_input, num_outputs, dim_output, dim_hidden=128):
        super(DeepSet, self).__init__()
        self.num_outputs = num_outputs
        self.dim_output = dim_output
        self.enc = nn.Sequential(
                nn.Linear(dim_input, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden))
        self.dec = nn.Sequential(
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, num_outputs*dim_output))

    def forward(self, X):
        X = self.enc(X).mean(-2)
        X = self.dec(X).reshape(-1, self.num_outputs, self.dim_output)
        return X

class SetTransformer(nn.Module):
    """
    Original SetTransformer implementation.
    """
    def __init__(self, dim_input, num_outputs, dim_output, num_inds=4, model_dim=32, num_heads=4, ln=True,
                 device='cpu', count=2):
        super(SetTransformer, self).__init__()
        self.device = device

        # Encoder (ISAB blocks)
        encoder_blocks = [ISAB(dim_input, model_dim, num_heads, num_inds, ln=ln)]
        for _ in range(1, count):
            encoder_blocks.append(ISAB(model_dim, model_dim, num_heads, num_inds, ln=ln))
        self.enc = nn.Sequential(*encoder_blocks)

        # Pooling (PMA)
        self.pooling = nn.Sequential(
            PMA(model_dim, num_heads, num_outputs, ln=ln)
        )

        # Decoder (SAB blocks + final Linear)
        decoder_blocks = [SAB(model_dim, model_dim, num_heads, ln=ln)]
        for _ in range(1, count):
            decoder_blocks.append(SAB(model_dim, model_dim, num_heads, ln=ln))
        decoder_blocks.append(nn.Linear(model_dim, dim_output))
        self.dec = nn.Sequential(*decoder_blocks)
        self.to(self.device)

    def forward(self, X):
        # Ensure inputs are on the correct device
        shape = X.shape
        if len(X.shape) == 4:
            _, _, agents, features = shape
            X = X.reshape(-1, X.shape[-2], X.shape[-1])

        X = X.to(self.device)
        output = self.dec(self.pooling(self.enc(X)))

        output = output.squeeze(1)

        if len(shape) == 4:
            output = output.reshape(shape[0], shape[1], output.shape[-2], output.shape[-1])
        return output

class SetTransformerOne(nn.Module):
    """
    SetTransformer adapted for single output to be used by the MLP core in my output.
    """
    def __init__(self, dim_input, dim_output, num_inds=32, dim_hidden=32, num_heads=4, ln=True, device='cpu'):
        super(SetTransformerOne, self).__init__()
        self.device = device
        self.enc = nn.Sequential(
                ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
                ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln))
        self.dec = nn.Sequential(
            PMA(dim_hidden, num_heads, num_seeds=1, ln=ln),
            nn.Linear(dim_hidden, dim_output))
        self.to(self.device)

    def forward(self, X):
        X = X.to(self.device)
        forwarded = self.dec(self.enc(X))
        return forwarded.squeeze(1) # Remove second dimension  (batch_size, dim)

class SetTransformerInv(nn.Module):
    """
    Adapted SetTransformer implementation to make it invariant to the number of agents.
    """
    def __init__(self, dim_input, num_agents, dim_output=1, num_inds=4, model_dim=32, num_heads=4, ln=True,
                 device='cpu', count=2):
        super(SetTransformerInv, self).__init__()
        self.device = device
        self.num_agents = num_agents
        self.dim_output = dim_output

        # Encoder (ISAB blocks)
        encoder_blocks = [ISAB(dim_input, model_dim, num_heads, num_inds, ln=ln)]
        for _ in range(1, count):
            encoder_blocks.append(ISAB(model_dim, model_dim, num_heads, num_inds, ln=ln))
        self.enc = nn.Sequential(*encoder_blocks)

        # Pooling (PMA)
        self.pooling = nn.Sequential(
            PMA(model_dim, num_heads, 1, ln=ln)
        )

        # Decoder (SAB blocks + final Linear)
        decoder_blocks = [SAB(model_dim + dim_input, model_dim, num_heads, ln=ln)]
        for _ in range(1, count):
            decoder_blocks.append(SAB(model_dim, model_dim, num_heads, ln=ln))
        decoder_blocks.append(nn.Linear(model_dim, dim_output))
        self.dec = nn.Sequential(*decoder_blocks)

        self.to(self.device)

    def forward(self, X):
        # Ensure inputs are on the correct device
        shape = X.shape
        if len(X.shape) == 4:
            X = X.reshape(-1, X.shape[-2], X.shape[-1])

        X = X.to(self.device)
        output_encoder_pooling = self.pooling(self.enc(X))

        final_output = []
        for agent_idx in range(self.num_agents):
            # add local state of X of the agent to output_encoder_pooling
            local_state = X[:, agent_idx, :].unsqueeze(1)
            concat_encoder = torch.cat([output_encoder_pooling, local_state], dim=2)
            # pass state of the agents concatenated with output_encoder_pooling through decoder
            output_decoder = self.dec(concat_encoder)
            final_output.append(output_decoder)

        final_output = torch.cat(final_output, dim=1)
        if len(shape) == 4:
            final_output = final_output.reshape(shape[0], shape[1], final_output.shape[-2], final_output.shape[-1])
        return final_output

class STransformer(nn.Module):
    def __init__(self, dim_input, dim_output, num_inds=4, model_dim=32, num_heads=4, ln=True, device='cpu', isab=True, count=2):
        super(STransformer, self).__init__()
        self.device = device
        encoder_blocks = []
        if isab:
            encoder_blocks.append(ISAB(dim_input, model_dim, num_heads, num_inds, ln=ln))
            for _ in range(1, count):
                encoder_blocks.append(ISAB(model_dim, model_dim, num_heads, num_inds, ln=ln))
        else:
            encoder_blocks.append(SAB(dim_input, model_dim, num_heads, ln=ln))
            for _ in range(1, count):
                encoder_blocks.append(SAB(model_dim, model_dim, num_heads, ln=ln))

        self.enc = nn.Sequential(*encoder_blocks)
        self.out = nn.Linear(model_dim, dim_output)
        self.to(self.device)

    def forward(self, X):
        # Ensure inputs are on the correct device
        shape = X.shape
        if len(X.shape) == 4:
            _, _, agents, features = shape
            X = X.reshape(-1, X.shape[-2], X.shape[-1])

        X = X.to(self.device)
        output = self.out(self.enc(X))

        if len(shape) == 4:
            output = output.reshape(shape[0], shape[1], output.shape[-2], output.shape[-1])
        return output
