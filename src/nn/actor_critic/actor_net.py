import torch
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal
from src.config import PPOConfig
from src.nn.actor_critic.base_centralized_network import BaseCentralizedNetwork


class ActorNetDecentralizedExecution:
    """
    Policy network with decentralized execution.
    Meaning that each agent has its own policy network with only its own observations as input.
    """
    def __init__(self, config: PPOConfig, n_agents_inputs, n_agents_outputs, action_spec, action_key, device):
        """
        n_agent_inputs (int or None) – number of inputs for each agent
        n_agent_outputs (int) – number of outputs for each agent
        n_agents (int) – number of agents
        """
        if config.core_mlp_actor is None:
            policy_mlp = MultiAgentMLP(
                n_agent_inputs=n_agents_inputs,
                n_agent_outputs= n_agents_outputs,
                n_agents=config.n_agents,
                centralised=False,
                share_params=True,
                device=device,
                depth=config.mlp_core_depth,
                num_cells=config.mlp_core_num_cells,
                activation_class=config.activation_class,
            )
        else:
            policy_mlp = config.core_mlp_actor
            policy_mlp.n_agents = config.n_agents
        

        # Define the policy network
        self.policy_net = torch.nn.Sequential(
            policy_mlp,
            NormalParamExtractor(),  # Extracts loc and scale for the distribution
        )
        
        # Create a TensorDictModule for handling input and output keys
        policy_module = TensorDictModule(
            self.policy_net,
            in_keys=[("agents", "observation")],
            out_keys=[("agents", "loc"), ("agents", "scale")],
        )
        
        # Create a probabilistic actor
        self.actor = ProbabilisticActor(
            module=policy_module,
            spec=action_spec,
            in_keys=[("agents", "loc"), ("agents", "scale")],
            out_keys=[action_key],  # Expected key for actions
            distribution_class=TanhNormal,
            distribution_kwargs={
                "low": action_spec[action_key].space.low,
                "high": action_spec[action_key].space.high,
            },
            return_log_prob=True,
            log_prob_key=("agents", "sample_log_prob"),
        )
    
    def forward(self, observations):
        return self.actor(observations)  # Returns actions given observations

    def move_to_cpu(self):
        self.policy_net.cpu()
        self.actor.cpu()
    
class ActorNetCentralizedExecution(BaseCentralizedNetwork):
    """
    Policy network with centralized execution for CPPO (CTCE).
    Each agent has its own policy network, but all agents' observations are embedded and fed to the network.
    """

    def __init__(self, config: PPOConfig, observation_spec, action_spec, action_key, n_agents_outputs, device):
        super().__init__(config, observation_spec, device, output_dim=n_agents_outputs, i_am_actor=True)

        self.policy_net = torch.nn.Sequential(self.network, NormalParamExtractor())
        policy_module = TensorDictModule(
            self.policy_net,
            in_keys=[("agents", "observation")],
            out_keys=[("agents", "loc"), ("agents", "scale")],
        )

        self.actor = ProbabilisticActor(
            module=policy_module,
            spec=action_spec,
            in_keys=[("agents", "loc"), ("agents", "scale")],
            out_keys=[action_key],
            distribution_class=TanhNormal,
            distribution_kwargs={
                "low": action_spec[action_key].space.low,
                "high": action_spec[action_key].space.high,
            },
            return_log_prob=True,
            log_prob_key=("agents", "sample_log_prob"),
        )

    def forward(self, observations):
        return self.actor(observations)