import torch
from src.config import PPOConfig, KPI, EmbeddingStrategy
from mappo import MultiAgentPPO
import itertools
from dataclasses import replace
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import numpy as np
import scipy.stats as stats

sns.set_theme(
    style="whitegrid",
    font_scale=1.5,
    palette="bright",
    rc={"lines.linewidth": 2}
)

@dataclass
class ExperimentResult:
    kpis: any
    mlps: dict

class Experiment:
    def __init__(self, config: PPOConfig, name: str = None, device: torch.device = None):
        self.config = config
        self.config_dict = {} # will house the generated config with potentially created MLPs
        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = device
        self.name = name or str(config)
        self.results: dict[int, ExperimentResult] = {}
        self.time_taken = None
        self.experiment_mappo = None

    def run(self, update: bool = True, seed: int = 0, merge:bool = False):
        """
        Run the experiment with the given config.
        Updates the kpis and results_mlps attributes.
        merge: dictates whether the result should be merged. (Important when rerunning the experiment)
        """

        if update and (seed in self.config_dict): # see if update and config_dict key exists
            self.experiment_mappo = MultiAgentPPO(self.config_dict[seed], self.device, seed)
        else:
            self.experiment_mappo = MultiAgentPPO(self.config, self.device, seed)
        results_mlps = self.experiment_mappo.train()
        result = ExperimentResult(kpis=self.experiment_mappo.kpis,
                                  mlps=results_mlps)

        if merge:
            self.results[seed] = self._merge_kpis(result, seed)
        else:
            self.results[seed] = result
        if update: self._update_config_mlps(seed)
        return result

    def run_with_profiler(self):
        self.config.update(profile=True)
        old_iter = self.config.n_iters
        self.config.update(n_iters=10)

        self.experiment_mappo = MultiAgentPPO(self.config, device=self.device)
        _ = self.experiment_mappo.train()
        memory_usage = self.experiment_mappo.kpis.training_memory_usage
        self.config.update(profiler=False)
        self.config.update(n_iters=old_iter)
        return memory_usage

    def _merge_kpis(self, new_result: ExperimentResult, seed) -> ExperimentResult:
        # if result is empty list
        if seed not in self.results:
            return new_result
        old_kpis = self.results[seed].kpis

        new_kpis = new_result.kpis
        new_mlps = new_result.mlps

        return_kpi = KPI(
            inference_time=old_kpis.inference_time+new_kpis.inference_time,
            inference_memory_usage=old_kpis.inference_memory_usage+new_kpis.inference_memory_usage,
            loss=old_kpis.loss + new_kpis.loss,
            mean_rewards=old_kpis.mean_rewards + new_kpis.mean_rewards,
            training_memory_usage=old_kpis.training_memory_usage+new_kpis.training_memory_usage,
            training_time=old_kpis.training_time+new_kpis.training_time,
        )

        return_result = ExperimentResult(kpis=return_kpi, mlps=new_mlps)
        return return_result

    def run_confidence(self, k=10, profile_once=True, merge=False, update=False, clear=True, random_seed=0):
        """
        Run the experiment with the given config k times.
        Updates the kpis and results_mlps attributes.
        """
        if clear: self.results.clear()
        for i in range(k):
            if random_seed!=0:
                self.run(update=update, seed=random_seed, merge=merge)
                random_seed = random_seed + 1
            else:
                self.run(update=update, seed=i, merge=merge)

        if (profile_once and
                not self.config.profile and                             # if profile than additional profiling is not needed
                not ('transformer' in self.config.strategy.value) and   # SetTransformer and SetTransformer crash profiler
                not self.device == torch.device("mps")):                # profiler only works on CUDA or CPU
            mem_usage = self.run_with_profiler()
            # print('Memory Usage', mem_usage)

            for res in self.results.values():
                res.kpis.training_memory_usage = mem_usage

    def _update_config_mlps(self, seed):
        updated_fields = {}

        for model_name, model in self.results[seed].mlps.items():
            if model_name == "policy_embedding_mlp":
                updated_fields["embedding_mlp_actor"] = model
            elif model_name == "policy_main_mlp":
                updated_fields["core_mlp_actor"] = model
            elif model_name == "critic_embedding_mlp":
                updated_fields["embedding_mlp_critic"] = model
            elif model_name == "critic_main_mlp":
                updated_fields["core_mlp_critic"] = model

        # Apply all updates at once
        new_config = replace(self.config, **updated_fields)
        self.config_dict[seed] = new_config

    def rollout(self):
        while True:
            self.experiment_mappo.rollout()

    def rollout_experiment_visual(self, changes_to_config: dict = None):
        """
        Rollout the experiment with the given changes to the config.
        changes_to_config: dict - dictionary of changes to the config. (e.g. {"n_agents": 5})
        """
        if changes_to_config is not None:
            param_keys = list(changes_to_config.keys())
            values = list(changes_to_config.values())
            config = replace(self.config.update(**dict(zip(param_keys, values))))
        else:
            config = self.config
        experiment = MultiAgentPPO(config, self.device)
        print(self.device)
        while True:
            experiment.rollout()

    def rollout_experiment_reward(self, changes_to_config: dict = None, k = 10):
        """
        Rollout the experiment with the given changes to the config.
        Return mean reward of rollout
        """
        if self.config_dict:
            list_rolls = []
            for key, value in self.config_dict.items():
                param_keys = list(changes_to_config.keys())
                values = list(changes_to_config.values())
                if changes_to_config is not None:
                    config = replace(self.config_dict[key])
                    config = config.update(**dict(zip(param_keys, values)))
                else:
                    config = self.config_dict[key]
                experiment = MultiAgentPPO(config, self.device, seed=key)
                list_rolls.append(experiment.test(n_rollouts=k))
            return list_rolls

        else:
            if changes_to_config is not None:
                param_keys = list(changes_to_config.keys())
                values = list(changes_to_config.values())
                config = replace(self.config.update(**dict(zip(param_keys, values))))
            else:
                config = self.config
            experiment = MultiAgentPPO(config, self.device)
            return experiment.test(n_rollouts=k)

class ExperimentSuite:
    def __init__(self, base_config: PPOConfig, param_grid: dict, name: str, device: torch.device = None):
        if name is None:
            raise ValueError("Name cannot be None.")
        self.base_config = base_config
        self.param_grid = param_grid
        self.name = name
        self.device = device
        self.experiments = self._generate_experiments()

    def _generate_experiments(self):
        param_keys = list(self.param_grid.keys())

        if len(param_keys) == 1:
            _param_values = self.param_grid[param_keys[0]]
            _param_keys = list(self.param_grid.keys())[0]
            return self.single_argument_grid(_param_keys, _param_values)

        param_values = itertools.product(*self.param_grid.values())

        try:
            return [
                Experiment(replace(self.base_config.update(**dict(zip(param_keys, values)))),
                          name=", ".join(f"{v.value}" for k, v in zip(param_keys, values)), device=self.device)
                for values in param_values
            ]
        except Exception:
            return [
                Experiment(replace(self.base_config.update(**dict(zip(param_keys, values)))),
                           name=", ".join(f"{k}={v}" for k, v in zip(param_keys, values)), device=self.device)
                for values in param_values
            ]

    def single_argument_grid(self, param_key, param_values):
        experiments = []
        for value in param_values:
            updated_config = self.base_config.update(**{param_key: value})
            name_str = getattr(value, "value", f"{param_key} = {value}")
            experiment = Experiment(
                replace(updated_config),
                name=str(name_str),
                device=self.device
            )
            experiments.append(experiment)
        return experiments

    def run_all(self):
        return {experiment.name: experiment.run() for experiment in self.experiments}

    def run_all_confidence(self, k=10, profile_once=True, update=False):
        """
        k : int, optional
            Number of times to repeat each experiment. Defaults to 10.
            Higher values improve statistical reliability.
        profile_once : bool, optional
            Whether to perform memory profiling once during the experiments. Defaults to True.
            If False, profiling is skipped entirely.
        update : bool, optional
            Whether to update the experimental configuration. Defaults to False.
            Must be True if the MLP (Multi-Layer Perceptron) is to be reused across experiments.
        """
        for experiment in self.experiments:
            experiment.run_confidence(k=k,  profile_once=profile_once, update=update)

    def rollout_all(self):
        for experiment in self.experiments:
            experiment.rollout()

    def rollout_all_get_rewards(self, changes, k=10):
        """
        Performs rollout experimentsOld for each change configuration across all experimentsOld.

        Parameters:
        - changes: List of dicts, where each dict represents a config change (e.g., {"n_agents": 5}).
        - k: Number of rollouts to average for each config change.

        Returns:
        - A pandas DataFrame with experiment names as rows and change descriptions as columns.
        """
        results = []

        for experiment in self.experiments:
            row = {"experiment": experiment.name if hasattr(experiment, 'name') else str(experiment)}

            rewards = experiment.rollout_experiment_reward(changes_to_config=changes, k=k)
            change_desc = ", ".join(f"{k}={v}" for k, v in changes.items())
            row[f"{change_desc}"] = rewards
            results.append(row)

        df = pd.DataFrame(results)
        return df

    def create_and_run_experiments_with_updated_config(self, changes_to_config: dict, create_new = True, k=10, seed = 0):
        """
        Note:
        This will not run the experiments again but create new experimentsOld with the updated config.
        Makes most sense to use this after original experiments have been run.

        changes_to_config: dict - dictionary of changes to the config. (e.g. {"n_agents": 5})
        """
        if create_new:
            added_experiments = []
            for experiment in self.experiments:
                config = replace(experiment.config.update(**changes_to_config))
                name = experiment.name + ", " + ", ".join(f"{k}={v}" for k, v in changes_to_config.items())
                new_experiment = Experiment(config, name, device=self.device)
                new_experiment.run()
                added_experiments.append(new_experiment)
            self.experiments += added_experiments
        else:
            for experiment in self.experiments:
                experiment.config.update(**changes_to_config)
                if experiment.config_dict:
                    for i in range(k):
                        seed_i = seed + i
                        if seed_i in experiment.config_dict: experiment.config_dict[seed_i].update(**changes_to_config)
                experiment.run_confidence(k=k, profile_once=False, merge=True, update=True, clear=False, random_seed=seed)

if __name__ == "__main__":
    base_config = PPOConfig(n_iters=50, num_epochs=1, decentralized_execution=True)
    param_grid = {
        "strategy": [
            EmbeddingStrategy.CONCAT,
            EmbeddingStrategy.MLP_LOCAL
        ],
    }

    suite = ExperimentSuite(base_config, param_grid, name="tryitout")
    results = suite.run_all_confidence(k=5)
    suite.plot_all_kpis_with_confidence()
    suite.create_table_with_confidence()
