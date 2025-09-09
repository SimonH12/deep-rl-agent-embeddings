import torch
from torchrl._utils import logger


from src.nn.gnn.models import GSAGE, GAT
from src.nn.multiagent import MultiAgentNetworkCore
from src.nn.feature_embeddings import FeatureEmbedding
from src.config import PPOConfig, EmbeddingStrategy
from src.nn.set_transformer.models import SetTransformer, SetTransformerInv, STransformer

class BaseCentralizedNetwork:
    def __init__(self, config: PPOConfig, observation_spec, device, output_dim=1, i_am_actor=False):
        self.device = device
        self.output_dim = output_dim
        self.i_am_actor = i_am_actor
        self.network = self._build_network(config, observation_spec)

    def _build_network(self, config, observation_spec):
        if config.strategy == EmbeddingStrategy.SET_TRANSFORMER_OG:
            return self._prepare_set_transformer(observation_spec, config)
        elif config.strategy == EmbeddingStrategy.SET_TRANSFORMER_INV:
            return self._prepare_set_transformer_inv(observation_spec, config)
        elif config.strategy == EmbeddingStrategy.SAB_TRANSFORMER or config.strategy == EmbeddingStrategy.ISAB_TRANSFORMER:
            return self._prepare_stransformer(observation_spec, config)
        elif config.strategy == EmbeddingStrategy.GRAPH_SAGE:
            return self._prepare_gnn_sage(observation_spec, config)
        elif config.strategy == EmbeddingStrategy.GRAPH_GAT:
            return self._prepare_gnn_gat(observation_spec, config)
        elif config.strategy == EmbeddingStrategy.GRAPH_GAT_v2:
            return self._prepare_gnn_gat(observation_spec, config, v2=True)
        else:
            embedding_strategy, mlp_core = self._prepare_standard_embedding_mlp(config, observation_spec)
            return torch.nn.Sequential(embedding_strategy, mlp_core)

    def _prepare_set_transformer(self, observation_spec, config):
        if config.core_mlp_critic and not self.i_am_actor:
            return config.core_mlp_critic
        elif config.core_mlp_critic and self.i_am_actor:
            return config.core_mlp_actor
        n_agents_inputs = observation_spec["agents", "observation"].shape[-1]
        return SetTransformer(
            dim_input=n_agents_inputs,
            num_outputs=config.n_agents,
            dim_output=self.output_dim,
            model_dim=config.model_dim,
            num_inds=config.inducing_points,
            num_heads=config.attention_heads,
            count=config.count,
            ln=config.norm,
            device=self.device
        )

    def _prepare_set_transformer_inv(self, observation_spec, config):
        if config.core_mlp_critic and not self.i_am_actor:
            config.core_mlp_critic.num_agents = config.n_agents
            return config.core_mlp_critic
        elif config.core_mlp_critic and self.i_am_actor:
            config.core_mlp_actor.num_agents = config.n_agents
            return config.core_mlp_actor
        n_agents_inputs = observation_spec["agents", "observation"].shape[-1]
        return SetTransformerInv(
            dim_input=n_agents_inputs,
            num_agents=config.n_agents,
            dim_output=self.output_dim,
            num_inds=config.inducing_points,
            model_dim=config.model_dim,
            num_heads=config.attention_heads,
            count=config.count,
            ln=config.norm,
            device=self.device
        )

    def _prepare_stransformer(self, observation_spec, config):
        if config.core_mlp_critic and not self.i_am_actor:
            config.core_mlp_critic.num_agents = config.n_agents
            return config.core_mlp_critic
        elif config.core_mlp_critic and self.i_am_actor:
            config.core_mlp_actor.num_agents = config.n_agents
            return config.core_mlp_actor
        n_agents_inputs = observation_spec["agents", "observation"].shape[-1]
        return STransformer(
            dim_input=n_agents_inputs,
            dim_output=self.output_dim,
            num_inds=config.inducing_points,
            model_dim=config.model_dim,
            num_heads=config.attention_heads,
            device=self.device,
            count=config.count,
            isab=config.strategy == EmbeddingStrategy.ISAB_TRANSFORMER,
            ln=config.norm
        )

    def _prepare_gnn_sage(self, observation_spec, config):
        if config.core_mlp_critic and not self.i_am_actor:
            config.core_mlp_critic.num_agents = config.n_agents
            return config.core_mlp_critic
        elif config.core_mlp_critic and self.i_am_actor:
            config.core_mlp_actor.num_agents = config.n_agents
            return config.core_mlp_actor
        n_agents_inputs = observation_spec["agents", "observation"].shape[-1]
        return GSAGE(
            num_features=n_agents_inputs,
            out_channels=self.output_dim,
            hidden_channels=config.hidden_layer_width,
            pooling_method=config.pooling_method,
            encode=config.use_encoder_mlp,
            device=self.device
        )

    def _prepare_gnn_gat(self, observation_spec, config, v2=False):
        if config.core_mlp_critic and not self.i_am_actor:
            config.core_mlp_critic.num_agents = config.n_agents
            return config.core_mlp_critic
        elif config.core_mlp_critic and self.i_am_actor:
            config.core_mlp_actor.num_agents = config.n_agents
            return config.core_mlp_actor
        n_agents_inputs = observation_spec["agents", "observation"].shape[-1]
        return GAT(
            num_features=n_agents_inputs,
            out_channels=self.output_dim,
            hidden_channels=config.hidden_layer_width,
            device=self.device,
            heads=config.attention_heads,
            encode=config.use_encoder_mlp,
            v2=v2
        )

    def _prepare_standard_embedding_mlp(self, config, observation_spec):
        """
        For MLP methods and ablations with mlp core network.
        """
        if not self.i_am_actor:
            possible_mlp = config.embedding_mlp_critic
        else:
            possible_mlp = config.embedding_mlp_actor
        embedding_strategy = FeatureEmbedding(
            n_agent_inputs=observation_spec["agents", "observation"].shape[-1],
            strategy=config.strategy,
            n_agents=config.n_agents,
            embedding_mlp=possible_mlp,
            depth=config.embedding_depth,
            model_dim=config.model_dim,
            activation_class=config.activation_class,
            pooling_method=config.pooling_method,
            max_agents=config.max_agents,
            num_heads=config.attention_heads,
            config_intermediate_layer_size=config.embedding_num_cells,
            device=self.device
        )

        if embedding_strategy.invariant_to_agent_permutation and not self.i_am_actor:
            mlp = config.core_mlp_critic
        elif embedding_strategy.invariant_to_agent_permutation and self.i_am_actor:
            mlp = config.core_mlp_actor
        else:
            # NOTE: If concat or mean_local_xy, then the mlp can not be reused as the input size might be different if n_agents is different
            #       In case I want to train a critic further with the same mlp, I need to check if the input size is the same (implement if necessary)
            if config.core_mlp_critic is not None:
                logger.info(f"CriticNetCentralizedLearning: Reusing MLP is not possible because the embedding strategy is not invariant to agent permutation.")
            mlp = None

        mlp_core = MultiAgentNetworkCore(
            device=self.device,
            depth=config.mlp_core_depth,
            num_cells=config.mlp_core_num_cells,
            activation_class=config.activation_class,
            output_size_embedding_strategy=embedding_strategy.embedding_output_dim,
            embedding_strategy=embedding_strategy.name,
            output_dim=self.output_dim,
            mlp=mlp
        )

        assert embedding_strategy.embedding_output_dim == mlp_core.input_dim

        return embedding_strategy, mlp_core