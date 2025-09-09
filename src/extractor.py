from typing import Dict
import torch
from src.config import EmbeddingStrategy

def get_policy_and_critic_mlps(config, policy_class, critic_class, decentralized_execution: bool) -> Dict[str, torch.nn.Module]:
    if config.strategy in {
        EmbeddingStrategy.SET_TRANSFORMER_OG,
        EmbeddingStrategy.SET_TRANSFORMER_INV,
        EmbeddingStrategy.SAB_TRANSFORMER,
        EmbeddingStrategy.ISAB_TRANSFORMER,
        EmbeddingStrategy.GRAPH_SAGE,
        EmbeddingStrategy.GRAPH_GAT,
        EmbeddingStrategy.GRAPH_GAT_v2
    }:
        return _get_graph_or_set_transformer_mlps(config, policy_class, critic_class, decentralized_execution)
    else:
        return _get_default_mlps(config, policy_class, critic_class, decentralized_execution)

def _get_graph_or_set_transformer_mlps(config, policy_class, critic_class, decentralized_execution: bool):
    if decentralized_execution:
        policy_main_mlp = policy_class.policy_net._modules['0']
        critic_main_mlp = critic_class.critic_net
    else:
        policy_main_mlp = policy_class.policy_net
        critic_main_mlp = critic_class.critic_net

    return {
        "critic_main_mlp": critic_main_mlp,
        "policy_main_mlp": policy_main_mlp
    }

def _get_default_mlps(config, policy_class, critic_class, decentralized_execution: bool):
    if decentralized_execution:
        policy_main_mlp = policy_class.policy_net._modules['0']
        critic_main_mlp = critic_class.critic_net._modules['1'].main_mlp
        mlps = {
            "critic_main_mlp": critic_main_mlp,
            "policy_main_mlp": policy_main_mlp
        }

        if config.strategy.value != 'concat':
            mlps["critic_embedding_mlp"] = critic_class.critic_net._modules['0'].embedding_mlp

        return mlps
    else:
        policy_main_mlp = policy_class.network._modules['1'].main_mlp
        critic_main_mlp = critic_class.critic_net._modules['1'].main_mlp
        mlps = {
            "critic_main_mlp": critic_main_mlp,
            "policy_main_mlp": policy_main_mlp
        }

        if config.strategy.value != 'concat':
            mlps["critic_embedding_mlp"] = critic_class.critic_net._modules['0'].embedding_mlp
            mlps["policy_embedding_mlp"] = policy_class.network._modules['0'].embedding_mlp

        return mlps
