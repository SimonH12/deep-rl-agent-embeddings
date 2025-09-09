from dataclasses import dataclass, fields
import torch
from enum import Enum


@dataclass
class KPI:
    """
    Class to store KPIs.
    """
    training_time: float = 0.0
    training_memory_usage: float = 0.0
    inference_time: float = 0.0
    inference_memory_usage: float = 0.0
    mean_rewards: list = None
    loss: list = None
    mmac_params: tuple = None
    flops: tuple = None
    parameters: tuple = None


    @classmethod
    def get_kpi_names(cls):
        # Retrieve all field names of the dataclass, which are the KPIs
        return [field.name for field in fields(cls)]

class EmbeddingStrategy(Enum):
    """
    Enumeration for embedding strategies.

    This class provides a set of embedding strategies that can be used for
    different scenarios, such as concatenation, mean-based aggregation,
    and transformer-based approaches.

    Notes on naming:
    "_local" => Indicates that the global state that is embedded that an agent receives is made up of all other agents' states,
                with the agent itself excluded. The local state is not passed to the embedding method
                but concatenated to the global state embedding.
    "_global" => Indicates that the global state that is embedded is made up of all agents' states. The local state
                is passed to the embedding method and concatenated (in its original form) to the global state embedding.
    """
    CONCAT = "concat"
    MLP = "mlp"
    MLP_LOCAL = "mlp_local"
    MLP_GLOBAL = "mlp_global"
    MLP_LOCAL_XY = "mlp_local_xy"
    TRANSFORMER_GLOBAL = "transformer_global"
    TRANSFORMER_LOCAL = "transformer_local"
    SET_TRANSFORMER_GLOBAL = "set_transformer_global"
    SET_TRANSFORMER_LOCAL = "set_transformer_local"
    SET_TRANSFORMER_OG = "set_transformer_og"
    SET_TRANSFORMER_INV = "set_transformer_inv"
    SAB_TRANSFORMER = "sab_transformer"
    ISAB_TRANSFORMER = "isab_transformer"
    GRAPH_SAGE = "graph_sage"
    GRAPH_GAT = "graph_gat"
    GRAPH_GAT_v2 = "graph_gat_v2"

    @property
    def is_invariant(self) -> bool:
        """Determine if the embedding strategy is invariant to the number and order of agents."""
        return self in {
            EmbeddingStrategy.MLP,
            EmbeddingStrategy.MLP_GLOBAL,
            EmbeddingStrategy.MLP_LOCAL,
            EmbeddingStrategy.TRANSFORMER_GLOBAL,
            EmbeddingStrategy.TRANSFORMER_LOCAL,
            EmbeddingStrategy.SAB_TRANSFORMER,
            EmbeddingStrategy.ISAB_TRANSFORMER,
            EmbeddingStrategy.SET_TRANSFORMER_GLOBAL,
            EmbeddingStrategy.SET_TRANSFORMER_LOCAL,
            EmbeddingStrategy.SET_TRANSFORMER_INV,
            EmbeddingStrategy.GRAPH_SAGE,
            EmbeddingStrategy.GRAPH_GAT,
            EmbeddingStrategy.GRAPH_GAT_v2
        }

    @property
    def embedding_type(self) -> str:
        """Determine the type of embedding based on strategy."""
        embedding_types =  {
            EmbeddingStrategy.CONCAT: "other",
            EmbeddingStrategy.MLP: "mlp",
            EmbeddingStrategy.MLP_LOCAL: "mlp",
            EmbeddingStrategy.MLP_GLOBAL: "mlp",
            EmbeddingStrategy.MLP_LOCAL_XY: "mlp",
            EmbeddingStrategy.TRANSFORMER_GLOBAL: "transformer",
            EmbeddingStrategy.TRANSFORMER_LOCAL: "transformer",
            EmbeddingStrategy.SET_TRANSFORMER_GLOBAL: "set_transformer",
            EmbeddingStrategy.SET_TRANSFORMER_LOCAL: "set_transformer",
            EmbeddingStrategy.SET_TRANSFORMER_OG: "set_transformer",
            EmbeddingStrategy.SET_TRANSFORMER_INV: "set_transformer",
            EmbeddingStrategy.SAB_TRANSFORMER: "transformer",
            EmbeddingStrategy.ISAB_TRANSFORMER: "set_transformer",
            EmbeddingStrategy.GRAPH_SAGE: "graph_nn",
            EmbeddingStrategy.GRAPH_GAT: "graph_nn",
            EmbeddingStrategy.GRAPH_GAT_v2: "graph_nn"
        }
        return embedding_types[self]


@dataclass
class PPOConfig:
    """
    Configuration for the PPO algorithm.
    """
    # Environment settings
    scenario_name: str = "navigation"
    strategy: EmbeddingStrategy = EmbeddingStrategy.CONCAT
    n_agents: int = 5

    # PPO and training settings
    decentralized_execution: bool = True
    frames_per_batch: int = 6000
    n_iters: int = 80
    num_epochs: int = 8
    minibatch_size: int = 400
    lr: float = 3e-4
    max_grad_norm: float = 1.0
    clip_epsilon: float = 0.2
    gamma: float = 0.99
    lmbda: float = 0.9
    entropy_eps: float = 1e-4
    max_steps: int = 200

    # Model settings
    # for MLP approaches (MLP_LOCAL, MLP_GLOBAL, MLP)
    embedding_depth: any = 2
    embedding_num_cells: any = None

    # for GNN approaches (GAT, GAT_v2, GSAGE)
    hidden_layer_width: int = 48
    use_encoder_mlp: bool = True

    # for GAT, GAT_v2, Transformer and SetTransformer
    attention_heads: int = 2

    # for Transformer and SetTransformer
    model_dim: int = 32

    # for Transformer
    max_agents: int = 10

    # for SetTransformer
    inducing_points: int = 4
    norm: bool = False
    count: int = 2

    mlp_core_depth: int = 2
    mlp_core_num_cells: int = 64
    activation_class=torch.nn.Tanh

    # For mlp approaches and gsage
    pooling_method: str = "mean"

    # Add pretrained models if needed
    embedding_mlp_critic: any = None
    embedding_mlp_actor: any = None

    core_mlp_actor: any = None
    core_mlp_critic: any = None

    # Other
    profile: bool = False # NOTE: GAT runs 10 times slower with pytorch profiler

    # Flag to enable strategy-specific defaults
    use_strategy_defaults: bool = False

    # Strategy-specific defaults dictionary
    # These are the hyperparameter for which the EmbeddingStrategy performs best in balance
    STRATEGY_DEFAULTS = {
        EmbeddingStrategy.CONCAT: dict(mlp_core_num_cells=64),

        EmbeddingStrategy.MLP: dict(mlp_core_num_cells=256, embedding_depth=None),
        EmbeddingStrategy.MLP_LOCAL: dict(mlp_core_num_cells=128, embedding_depth=None),
        EmbeddingStrategy.MLP_GLOBAL: dict(mlp_core_num_cells=256, embedding_depth=2, hidden_layer_width=16),

        EmbeddingStrategy.GRAPH_SAGE: dict(     hidden_layer_width=32, use_encoder_mlp=True),
        EmbeddingStrategy.GRAPH_GAT: dict(      hidden_layer_width=64, use_encoder_mlp=True, attention_heads=3),
        EmbeddingStrategy.GRAPH_GAT_v2: dict(   hidden_layer_width=48, use_encoder_mlp=True, attention_heads=1),

        EmbeddingStrategy.TRANSFORMER_GLOBAL: dict(embedding_depth=1, attention_heads=1, model_dim=32), # NOTE: legacy method
        EmbeddingStrategy.SET_TRANSFORMER_INV: dict(    model_dim=48, attention_heads=1, count=1, inducing_points=1),
        EmbeddingStrategy.SET_TRANSFORMER_OG: dict(     model_dim=48, attention_heads=1, count=1, inducing_points=1),
        EmbeddingStrategy.SAB_TRANSFORMER: dict(        model_dim=64, attention_heads=1, count=2, inducing_points=1),
        EmbeddingStrategy.ISAB_TRANSFORMER: dict(       model_dim=64, attention_heads=1, count=2, inducing_points=1),
    }

    def __post_init__(self):
        if self.use_strategy_defaults:
            defaults = PPOConfig.STRATEGY_DEFAULTS.get(self.strategy, {})
            for key, value in defaults.items():
                if key in PPOConfig.__dataclass_fields__:
                    setattr(self, key, value)

    def update(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self

    def __str__(self):
        return f"PPOConfig: {self.__dict__}"
    

if __name__ == "__main__":
    config1 = PPOConfig(strategy=EmbeddingStrategy.MLP)
    print(config1)

    print("With Strategy Defaults Applied")
    config2 = PPOConfig(strategy=EmbeddingStrategy.MLP, use_strategy_defaults=True)
    print(config2)

