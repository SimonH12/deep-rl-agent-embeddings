from abc import ABC, abstractmethod
from typing import Tuple, Optional, List

from torch.nn.functional import embedding
from torchrl.modules.models import MLP
from torch import nn
import torch
from src.nn.transformer import AgentTransformer
from src.nn.set_transformer.models import SetTransformerOne
from src.config import EmbeddingStrategy
from src.nn.mlp.models import MLPGlobal, MLPLocal

def create_list_num_cells(in_layer, intermediate_layers, depth):

    if intermediate_layers is None or depth is None:
        return in_layer
    list_num_cells = []
    # list_num_cells.append(in_layer)
    for _ in range(depth):
        list_num_cells.append(intermediate_layers)
    return list_num_cells


class FeatureEmbeddingBase(nn.Module, ABC):
    def __init__(self, strategy: EmbeddingStrategy, n_agent_inputs: int,
                 n_agents: Optional[int] = None):
        super().__init__()
        self.name = strategy.name
        self.invariant_to_agent_permutation = strategy.is_invariant
        self.n_agent_inputs = n_agent_inputs
        self.n_agents = n_agents

    @abstractmethod
    def embed_feature(self, inputs: Tuple[torch.Tensor]) -> torch.Tensor:
        pass

    @abstractmethod
    def get_embedding_output_dim(self) -> int:
        pass

    def forward(self, *x):
        return self.embed_feature(x)


class FeatureEmbedding(FeatureEmbeddingBase):
    """
    Handles embedding logic for agents' features based on embedding strategy.
    Notes on naming:
    "_local" => Indicates that the global state that is embedded that an agent receives is made up of all other agents' states,
                with the agent itself excluded. The local state is not passed to the embedding method
                but concatenated to the global state embedding.
    "_global" => Indicates that the global state that is embedded is made up of all agents' states. The local state
                is passed to the embedding method and concatenated (in its original form) to the global state embedding.
    """
    def __init__(self,
                 n_agent_inputs: int,
                 strategy: EmbeddingStrategy,
                 n_agents: Optional[int] = None,
                 embedding_mlp: Optional[MLP] = None,
                 depth: int = 2,
                 num_heads: int = 2,
                 model_dim: int = 32,
                 max_agents: int = 10,
                 activation_class: torch.nn.Module = torch.nn.Tanh,
                 pooling_method: str = "mean",
                 device: torch.device = torch.device("mps"),
                 config_intermediate_layer_size = None,
                 ):


        super().__init__(strategy=strategy, n_agent_inputs=n_agent_inputs, n_agents=n_agents)

        self.strategy = strategy
        self.depth = depth
        self.model_dim = model_dim
        self.activation_class = activation_class
        self.device = device
        self.num_heads = num_heads
        self.max_agents = max_agents
        self._num_cells = create_list_num_cells(self.n_agent_inputs, config_intermediate_layer_size, self.depth)
        if isinstance(self._num_cells, list):
            self.output_size_mlp = self._num_cells[-1]
        else:
            self.output_size_mlp = self._num_cells
        self.embedding_output_dim = self.get_embedding_output_dim()
        self._get_pooling_method(pooling_method)
        self._get_embedding_mlp(embedding_mlp, strategy)

    def _get_embedding_mlp(self, embedding_mlp, strategy):
        if strategy.embedding_type == 'mlp':
            self.embedding_mlp = embedding_mlp if embedding_mlp else self._create_embedding_mlp()
        elif strategy.embedding_type == 'transformer':
            self.embedding_mlp = embedding_mlp if embedding_mlp else self._create_embedding_transformer()
        elif strategy.embedding_type == 'set_transformer':
            self.embedding_mlp = embedding_mlp if embedding_mlp else self._create_embedding_set_transformer()

    def _get_pooling_method(self, pooling_method):
        pooling_functions = {
            "max": lambda t, dim: torch.max(t, dim=dim)[0],  # Extract only values
            "mean": lambda t, dim: torch.mean(t, dim=dim)
        }

        if pooling_method not in pooling_functions:
            raise ValueError(f"Unknown pooling method: {pooling_method}")

        self.pooling_method = pooling_functions[pooling_method]

    def _create_embedding_mlp(self):
        if self.depth is None: # if no depth information, no mlp is performed (used to see how embedding fares if no transformation is applied)
            embedding_mlp = nn.Identity()
        else:
            embedding_mlp = MLP(
                in_features=self.n_agent_inputs,
                out_features=self.output_size_mlp,
                depth=self.depth,
                num_cells=self._num_cells,
                activation_class=self.activation_class,
                device=self.device
            )
        if self.strategy == EmbeddingStrategy.MLP_LOCAL:
            return MLPLocal(
                embedding_mlp=embedding_mlp,
                pooling_method=self.pooling_method,
            )
        elif self.strategy == EmbeddingStrategy.MLP_GLOBAL:
            return MLPGlobal(
                embedding_mlp=embedding_mlp,
                pooling_method=self.pooling_method,
            )
        return embedding_mlp

    def _create_embedding_transformer(self):
        return AgentTransformer(
            in_features=self.n_agent_inputs,
            out_features=self.output_size_mlp,
            depth=self.depth,
            model_dim=self.model_dim,
            num_heads=self.num_heads,
            max_num_agents=self.max_agents,
            device=self.device
        )

    def _create_embedding_set_transformer(self):
        induced_points = self.output_size_mlp // 4
        assert induced_points > 0, "Induced points must be greater than 0. Adjust output size of embedding MLP."
        return SetTransformerOne(
            dim_input=self.n_agent_inputs,
            dim_output=self.output_size_mlp,
            num_inds=induced_points,
            device=self.device
        )

    def embed_feature(self, inputs: Tuple[torch.Tensor]) -> torch.Tensor:
        inputs = inputs[0]
        shape = inputs.shape
        if len(shape) == 4:
            _, _, agents, features = shape
        elif len(shape) == 3:
            _, agents, features = shape
        else:
            raise ValueError(f"Unexpected input shape: {shape}")

        inputs_reshaped = inputs.view(-1, agents, features)
        desired_shape = list(inputs_reshaped.shape)
        desired_shape[-1] = self.output_size_mlp
        inputs_after_mlp = torch.zeros(desired_shape, device=self.device)

        embed_methods = {
            EmbeddingStrategy.CONCAT: self._embed_concat,
            EmbeddingStrategy.MLP: self._embed_mlp,
            EmbeddingStrategy.MLP_LOCAL: self._embed_mlp_local_or_global,
            EmbeddingStrategy.MLP_GLOBAL: self._embed_mlp_local_or_global,
            EmbeddingStrategy.MLP_LOCAL_XY: self._embed_mlp_local_xy,
            EmbeddingStrategy.TRANSFORMER_GLOBAL: self._embed_transformer_global,
            EmbeddingStrategy.TRANSFORMER_LOCAL: self._embed_transformer_local,
            EmbeddingStrategy.SET_TRANSFORMER_GLOBAL: self._embed_transformer_global,
            EmbeddingStrategy.SET_TRANSFORMER_LOCAL: self._embed_transformer_local,
        }
        if self.strategy not in embed_methods:
            raise ValueError(f"Unknown embedding strategy: {self.strategy}")

        if self.strategy is EmbeddingStrategy.MLP_LOCAL_XY:
            return embed_methods[self.strategy](inputs_reshaped, shape, agents)
        elif self.strategy is EmbeddingStrategy.MLP_GLOBAL or self.strategy is EmbeddingStrategy.MLP_LOCAL or self.strategy is EmbeddingStrategy.MLP:
            return embed_methods[self.strategy](inputs_reshaped, shape, inputs_after_mlp)
        else:
            return embed_methods[self.strategy](inputs_reshaped, shape)

    def _embed_concat(self, inputs, shape):
        embeddings = inputs.flatten(-2, -1)
        embeddings = embeddings.unsqueeze(-2).expand(-1, self.n_agents, -1)
        embeddings = embeddings.view(*shape[:-1], -1)
        return embeddings


    def _embed_mlp(self, inputs, shape, inputs_after_mlp):
        embedding = self.embedding_mlp(inputs)
        embedding = self.pooling_method(embedding, dim=-2)
        return embedding.unsqueeze(-2).expand_as(inputs_after_mlp).view(*shape[:-1], -1)

    def _embed_mlp_local_or_global(self, inputs, shape, inputs_after_mlp):
        return self.embedding_mlp(inputs).view(*shape[:-1], -1)

    def _embed_mlp_local_xy(self, inputs, shape, agents):
        embeddings = torch.zeros_like(inputs)

        for agent_idx in range(agents):
            embeddings[:, agent_idx, :] = self.embedding_mlp(inputs[:, agent_idx, :])

        global_embeddings = torch.zeros_like(embeddings)
        for agent_idx in range(agents):
            other_agents = torch.cat([embeddings[:, :agent_idx, :], embeddings[:, agent_idx + 1:, :]], dim=1)
            global_embeddings[:, agent_idx, :] = self.pooling_method(other_agents, dim=1)

        global_xy = inputs[:, :, :2].reshape(-1, self.n_agents * 2).unsqueeze(-2).expand(-1, self.n_agents, -1)
        return torch.cat([inputs, global_embeddings, global_xy], dim=-1).view(*shape[:-1], -1)

    def _embed_transformer_global(self, inputs, shape):
        embeddings = self.embedding_mlp(inputs).unsqueeze(-2).expand_as(inputs)
        return torch.cat([inputs, embeddings], dim=-1).view(*shape[:-1], -1)

    def _embed_transformer_local(self, inputs, shape):
        global_embeddings = torch.stack(
            [self.embedding_mlp(torch.cat([inputs[:, :i, :], inputs[:, i + 1:, :]], dim=1)) for i in
             range(inputs.shape[1])], dim=1)
        return torch.cat([inputs, global_embeddings], dim=-1).view(*shape[:-1], -1)

    def get_embedding_output_dim(self) -> int:
        output_sizes = {
            EmbeddingStrategy.CONCAT: self.n_agents * self.n_agent_inputs,
            EmbeddingStrategy.MLP: self.output_size_mlp,
            EmbeddingStrategy.MLP_LOCAL: self.output_size_mlp + self.n_agent_inputs,
            # EmbeddingStrategy.MLP_GLOBAL: self.output_size_mlp * 2,
            EmbeddingStrategy.MLP_GLOBAL: self.output_size_mlp + self.n_agent_inputs,

            EmbeddingStrategy.MLP_LOCAL_XY: self.output_size_mlp * 2 + self.n_agents * 2,
            EmbeddingStrategy.TRANSFORMER_GLOBAL: self.output_size_mlp * 2,
            EmbeddingStrategy.TRANSFORMER_LOCAL: self.output_size_mlp * 2,
            EmbeddingStrategy.SET_TRANSFORMER_GLOBAL: self.output_size_mlp * 2,
            EmbeddingStrategy.SET_TRANSFORMER_LOCAL: self.output_size_mlp * 2
        }

        if self.strategy not in output_sizes:
            raise ValueError(f"Unknown embedding strategy: {self.strategy}")

        return output_sizes[self.strategy]