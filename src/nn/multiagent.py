from __future__ import annotations


from typing import Optional, Sequence, Type, Union

import torch
from torch import nn
from torchrl.data.utils import DEVICE_TYPING
from torchrl.modules.models import MLP


class MultiAgentNetworkCore(nn.Module):
    """
    A base class for multi-agent networks, that accepts different embedding strategies.
    """

    def __init__(
        self,
        *,
        use_td_params: bool = True,
        num_cells: Optional[Union[Sequence, int]] = None,
        activation_class: Optional[Type[nn.Module]] = nn.Tanh,
        depth: Optional[int] = None,
        device: Optional[DEVICE_TYPING] = None,
        output_size_embedding_strategy: int,
        embedding_strategy: str,
        output_dim: int = 1,
        mlp: Optional[MLP] = None,
    ):
        super().__init__()

        self.use_td_params = use_td_params
        self.output_dim = output_dim
        self.input_dim = output_size_embedding_strategy
        self.embedding_strategy = embedding_strategy
        self.num_cells = num_cells
        self.activation_class = activation_class
        self.depth = depth

        if mlp:
            self.main_mlp = mlp
        else:
            self.main_mlp = self._build_single_net(device=device)

        if not self.use_td_params:
            self.params.to_module(self.main_mlp)

    def _build_single_net(self, device):
        return MLP(
            in_features=self.input_dim,
            out_features=self.output_dim,
            depth=self.depth,
            num_cells=self.num_cells,
            activation_class=self.activation_class,
            device=device
        )

    def forward(self, inputs) -> torch.Tensor:
        """Forward pass of the network."""
        output = self.main_mlp(inputs)
        return output

    def __repr__(self):
        return "{self.__class__.__name__}({self.__dict__})"