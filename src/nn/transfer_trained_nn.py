import torch

from src.config import EmbeddingStrategy
from typing import Dict


def _get_mlps(self) -> Dict[str, torch.nn.Module]:
    if (
            self.config.strategy == EmbeddingStrategy.SET_TRANSFORMER_OG or self.config.strategy == EmbeddingStrategy.SET_TRANSFORMER_INV or self.config.strategy == EmbeddingStrategy.GRAPH_SAGE or self.config.strategy == EmbeddingStrategy.GRAPH_GAT or self.config.strategy == EmbeddingStrategy.GRAPH_GAT_v2) \
            and self.config.decentralized_execution:
        policy_main_mlp = self.policy_class.policy_net._modules['0']
        critic_main_mlp = self.critic_class.critic_net

        return {
            "critic_main_mlp": critic_main_mlp,
            "policy_main_mlp": policy_main_mlp
        }
    elif (
            self.config.strategy == EmbeddingStrategy.SET_TRANSFORMER_OG or self.config.strategy == EmbeddingStrategy.SET_TRANSFORMER_INV or self.config.strategy == EmbeddingStrategy.GRAPH_SAGE or self.config.strategy == EmbeddingStrategy.GRAPH_GAT or self.config.strategy == EmbeddingStrategy.GRAPH_GAT_v2) \
            and not self.config.decentralized_execution:
        policy_main_mlp = self.policy_class.policy_net
        critic_main_mlp = self.critic_class.critic_net

        return {
            "critic_main_mlp": critic_main_mlp,
            "policy_main_mlp": policy_main_mlp
        }
    elif self.config.decentralized_execution:
        policy_main_mlp = self.policy_class.policy_net._modules['0']
        critic_main_mlp = self.critic_class.critic_net._modules['1'].main_mlp
        if self.config.strategy.value != 'concat':
            critic_embedding_mlp = self.critic_class.critic_net._modules['0'].embedding_mlp
            return {
                "critic_embedding_mlp": critic_embedding_mlp,
                "critic_main_mlp": critic_main_mlp,
                "policy_main_mlp": policy_main_mlp
            }
        else:
            return {
                "critic_main_mlp": critic_main_mlp,
                "policy_main_mlp": policy_main_mlp
            }
    else:
        policy_main_mlp = self.policy_class.network._modules['1'].main_mlp
        critic_main_mlp = self.critic_class.critic_net._modules['1'].main_mlp
        if self.config.strategy.value != 'concat':
            critic_embedding_mlp = self.critic_class.critic_net._modules['0'].embedding_mlp
            policy_embedding_mlp = self.policy_class.network._modules['0'].embedding_mlp
            return {
                "critic_embedding_mlp": critic_embedding_mlp,
                "critic_main_mlp": critic_main_mlp,
                "policy_embedding_mlp": policy_embedding_mlp,
                "policy_main_mlp": policy_main_mlp
            }
        else:
            return {
                "critic_main_mlp": critic_main_mlp,
                "policy_main_mlp": policy_main_mlp
            }
