import torch
import time
import numpy as np
from torch import multiprocessing
from torchrl.collectors import SyncDataCollector
from torchrl.envs import RewardSum, TransformedEnv
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.envs.utils import check_env_specs
from torchrl.objectives import ClipPPOLoss, ValueEstimators
from torchrl._utils import logger
from tqdm import tqdm
from typing import Dict
from ptflops import get_model_complexity_info
from torch.profiler import profile, ProfilerActivity

from src.extractor import get_policy_and_critic_mlps
from src.nn.actor_critic.actor_net import ActorNetDecentralizedExecution, ActorNetCentralizedExecution
from src.nn.actor_critic.critic_net import CriticNetCentralizedLearning
from src.config import PPOConfig, EmbeddingStrategy, KPI
from src.multi_give_way_com import Scenario as MultiGiveWayCom
from src.navigation import Scenario as Navigation


class MultiAgentPPO:
    def __init__(self, config: PPOConfig, device: torch.device, seed: int = 0):
        self.config = config

        torch.manual_seed(seed)
        is_fork = multiprocessing.get_start_method() == "fork"
        self.device = device or (torch.device(0) if torch.mps.is_available() and not is_fork else torch.device("cpu"))

        self.share_parameters_policy = True
        self.total_frames = config.frames_per_batch * config.n_iters

        self.env = self._init_env()
        self.policy_class = self._init_policy()
        self.policy = self.policy_class.actor
        self.critic_class = self._init_critic()
        self.critic = self.critic_class.critic
        self.collector = self._init_collector()
        self.loss_module = self._init_loss()
        self.optimizer = torch.optim.Adam(self.loss_module.parameters(), self.config.lr)
        self.env = self._init_env()
        self.kpis = KPI()

    def _init_policy(self):
        if self.config.decentralized_execution:
            return ActorNetDecentralizedExecution(
                config=self.config,
                n_agents_inputs=self.env.observation_spec["agents", "observation"].shape[-1],
                n_agents_outputs=2 * self.env.action_spec.shape[-1],
                action_spec=self.env.unbatched_action_spec,
                action_key=self.env.action_key,
                device=self.device
            )
        else:
            return ActorNetCentralizedExecution(
                config=self.config,
                observation_spec=self.env.observation_spec,
                action_spec=self.env.unbatched_action_spec,
                action_key=self.env.action_key,
                n_agents_outputs=2 * self.env.action_spec.shape[-1],
                device=self.device
            )

    def _init_critic(self):
        return CriticNetCentralizedLearning(
            config=self.config,
            observation_spec=self.env.observation_spec,
            device=self.device,
        )

    def _init_env(self):
        env_kwargs = {}
        num_vmas_envs = self.config.frames_per_batch // self.config.max_steps
        if self.config.scenario_name == 'multi_give_way_com':
            scen = MultiGiveWayCom()
        elif self.config.scenario_name == 'navigation':
            scen = Navigation()
        else:
            scen = self.config.scenario_name
        env = VmasEnv(
            scenario=scen,
            num_envs=num_vmas_envs,
            continuous_actions=True,
            max_steps=self.config.max_steps,
            device=self.device,
            n_agents=self.config.n_agents,
            **env_kwargs
        )
        env = TransformedEnv(
            env,
            RewardSum(in_keys=[env.reward_key], out_keys=[("agents", "episode_reward")]),
        )
        check_env_specs(env)
        return env

    def _init_collector(self):
        return SyncDataCollector(
            self.env,
            self.policy,
            device=self.device,
            storing_device=self.device,
            frames_per_batch=self.config.frames_per_batch,
            total_frames=self.total_frames,
        )

    def _init_loss(self):
        loss_module = ClipPPOLoss(
            actor_network=self.policy,
            critic_network=self.critic,
            clip_epsilon=self.config.clip_epsilon,
            entropy_coef=self.config.entropy_eps,
            normalize_advantage=False,
        )
        loss_module.set_keys(
            reward=self.env.reward_key,
            action=self.env.action_key,
            sample_log_prob=("agents", "sample_log_prob"),
            value=("agents", "state_value"),
            done=("agents", "done"),
            terminated=("agents", "terminated"),
        )
        loss_module.make_value_estimator(ValueEstimators.GAE, gamma=self.config.gamma, lmbda=self.config.lmbda)
        return loss_module

    def train(self) -> Dict[str, torch.nn.Module]:
        pbar = tqdm(total=self.config.n_iters, desc="episode_reward_mean = 0")
        episode_reward_mean_list = []
        loss_list = []

        training_time = 0

        for tensordict_data in self.collector:  # collect episodes with the current policy (pi_old)
            self._expand_done_terminated(tensordict_data)
            train_start_time = time.time()
            self._compute_gae(tensordict_data) # compute advantage estimates

            data_view = tensordict_data.reshape(-1)
            loss_list_iteration = []

            if self.config.profile: prof = self.run_with_profiler(data_view, loss_list_iteration)
            else: self.run_epochs(data_view, loss_list_iteration)

            self.collector.update_policy_weights_() # pi_old <- pi
            training_time += time.time() - train_start_time
            episode_reward_mean = self._log_metrics(tensordict_data)
            episode_reward_mean_list.append(episode_reward_mean)
            loss_list.append(sum(loss_list_iteration) / len(loss_list_iteration))
            pbar.set_description(f"episode_reward_mean = {episode_reward_mean}", refresh=False)
            pbar.update()

        pbar.close()

        logger.info(f"Training time: {training_time:.2f} seconds")
        self.kpis.training_time = training_time
        if self.config.profile:
            total_memory = sum(
                max(0, item.self_cpu_memory_usage)
                for item in prof.key_averages()
                if hasattr(item, "self_cpu_memory_usage")
            ) / (1024 ** 2)
            self.kpis.training_memory_usage = total_memory  # in mb
        self.kpis.mean_rewards = episode_reward_mean_list
        self.kpis.loss = loss_list
        self.record_flops()
        self.record_inference_kpis()

        return self._get_mlps()

    def _get_mlps(self) -> Dict[str, torch.nn.Module]:
        return get_policy_and_critic_mlps(
            config=self.config,
            policy_class=self.policy_class,
            critic_class=self.critic_class,
            decentralized_execution=self.config.decentralized_execution
        )

    def run_epochs(self, data_view, loss_list_iteration):
        for _ in range(self.config.num_epochs): # K epochs
            for _ in range(self.config.frames_per_batch // self.config.minibatch_size): # Minibatches
                indices = torch.randperm(len(data_view))[:self.config.minibatch_size]
                subdata = data_view[indices]

                # with record_function("loss_module_forward_backward"):
                loss_vals = self.loss_module(subdata)
                loss_value = sum(loss_vals.values())
                loss_value.backward() # computes gradients for all parameters involved in the total loss — critic and actor parameters
                loss_list_iteration.append(loss_value.item())

                torch.nn.utils.clip_grad_norm_(self.loss_module.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()

    def run_with_profiler(self, data_view, loss_list_iteration):
        with profile(
                activities=[ProfilerActivity.CPU],
                profile_memory=True,
                record_shapes=True,
                with_stack=False,
        ) as prof:

            self.run_epochs(data_view, loss_list_iteration)

            prof.step()  # Advance the profiler step
        return prof

    def record_flops(self):
        try:
            input_shape = (self.config.minibatch_size, self.config.n_agents, self.env.observation_spec["agents", "observation"].shape[-1])

            # Get FLOPs and parameters count
            with torch.no_grad():
                macs, params = get_model_complexity_info(self.loss_module.critic_network, input_shape, as_strings=True, print_per_layer_stat=False)

            flops = float(macs.split()[0]) * 2 # mac *2 = flops
            if 'MMac' in macs:
                self.kpis.flops = flops
            elif 'GMac' in macs:
                self.kpis.flops = flops * 1000
            else:
                logger.error(f"Unknown flops: {macs}")
                self.kpis.flops = flops

            logger.info(f"macs: {macs}  Params: {params}")
            self.kpis.parameters = float(params.split()[0]) * 2
        except:
            logger.info("FLOPs could not be calculated")

    def test(self, n_rollouts: int = 10):
        """Run the policy for a number of rollouts and return the mean reward and the standard deviation."""
        rewards = []

        with torch.no_grad():
            for _ in range(n_rollouts):
                tensordict_data = self.env.rollout(
                    max_steps=self.config.max_steps,
                    policy=self.policy,
                    auto_cast_to_device=True,
                    break_when_any_done=False,
                )

                self._expand_done_terminated(tensordict_data)

                done = tensordict_data.get(("next", "agents", "done"))
                reward = (
                    tensordict_data.get(("next", "agents", "episode_reward"))[done].mean().item()
                )
                rewards.append(reward)

        mean_reward = np.mean(rewards)
        logger.info(f"Mean reward over {n_rollouts} rollouts: {mean_reward:.4f}")
        return rewards

    def _expand_done_terminated(self, tensordict_data):
        done_shape = tensordict_data.get_item_shape(("next", self.env.reward_key))
        tensordict_data.set(("next", "agents", "done"),
                            tensordict_data.get(("next", "done")).unsqueeze(-1).expand(done_shape))
        tensordict_data.set(("next", "agents", "terminated"),
                            tensordict_data.get(("next", "terminated")).unsqueeze(-1).expand(done_shape))

    def _compute_gae(self, tensordict_data):
        with torch.no_grad():
            self.loss_module.value_estimator(
                tensordict_data,
                params=self.loss_module.critic_network_params,
                target_params=self.loss_module.target_critic_network_params,
            )

    def _log_metrics(self, tensordict_data):
        done = tensordict_data.get(("next", "agents", "done"))
        episode_reward_mean = (
            tensordict_data.get(("next", "agents", "episode_reward"))[done].mean().item()
        )
        return episode_reward_mean

    def rollout(self):
        with torch.no_grad():
            self.env.rollout(
                max_steps=self.config.max_steps,
                policy=self.policy,
                callback=lambda env, _: env.render(),
                auto_cast_to_device=True,
                break_when_any_done=False,
            )

    def record_inference_kpis(self):
        # Set the device to CPU explicitly (more accurate memory measurements)
        # MPS has no built-in memory measurement
        cpu = torch.device("cpu")
        self.policy.to(cpu)

        with torch.no_grad():
            total_inference_time = 0.0
            num_steps = 0

            # Initialize the memory tracker
            def timing_callback(_, tensordict):
                nonlocal total_inference_time, num_steps
                obs = tensordict.to(cpu)
                start_time = time.time()
                self.policy(obs)  # Model inference
                end_time = time.time()
                inference_time = end_time - start_time
                # Accumulate the results
                total_inference_time += inference_time
                num_steps += 1

            # Run the rollout
            self.env.to(cpu)
            self.env.rollout(
                max_steps=self.config.max_steps,
                policy=self.policy,
                callback=timing_callback,
                auto_cast_to_device=True,
                break_when_any_done=False,
            )

        # Calculate averages
        avg_inference_time = total_inference_time / num_steps if num_steps > 0 else 0

        # logger.info(f"Average inference time: {avg_inference_time:.4f} seconds")
        self.kpis.inference_time = avg_inference_time
        self.policy.to(self.device)
        return avg_inference_time


if __name__ == "__main__":
    device = torch.device("cpu")
    logger.info(f"Device: {device}")

    config = PPOConfig(n_agents=4, num_epochs=8, n_iters=80, decentralized_execution=True, use_strategy_defaults=True,
                       strategy=EmbeddingStrategy.SAB_TRANSFORMER, scenario_name='navigation')

    multi_agent_ppo = MultiAgentPPO(config, device)
    # Train the agent
    multi_agent_ppo.train()

    # Show rollout
    while True:
        logger.info(f"Rollout the trained policy")
        multi_agent_ppo.rollout()
