from tensordict.nn import TensorDictModule
from src.config import PPOConfig
from src.nn.actor_critic.base_centralized_network import BaseCentralizedNetwork


class CriticNetCentralizedLearning(BaseCentralizedNetwork):
    def __init__(self, config: PPOConfig, observation_spec, device):
            super().__init__(config, observation_spec, device)
            self.critic_net = self.network
            self.critic = TensorDictModule(
                module=self.critic_net,
                in_keys=[("agents", "observation")],
                out_keys=[("agents", "state_value")],
            )

    def forward(self, observations):
            return self.critic(observations)
