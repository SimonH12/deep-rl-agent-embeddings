import os
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import re

all_kpi_names =['training_time',
                'training_memory_usage',
                'inference_time',
                'inference_memory_usage',
                'mean_rewards',
                'loss',
                'mmac_params',
                'flops',
                'parameters']

sns.set_theme(
    style="whitegrid",
    font_scale=1.5,
    palette="bright",
    rc={"lines.linewidth": 2}
)

class ExperimentUtils:
    def __init__(self, path=None, experiment_suite=None, all_kpis=all_kpi_names, map_strategy_names=False):
        self.path = path
        self.all_kpis = all_kpis
        if experiment_suite is None:
            self.df = self.load_df_from_file(mapping=map_strategy_names)
        else:
            self.df = ExperimentUtils.save_experiment_suite_to_df(experiment_suite)

    @staticmethod
    def save_experiment_to_df(experiment):
        """
        Converts an Experiment's results into a pandas DataFrame with columns:
        - 'name': name of the experiment
        - 'config': string representation of the config
        - 'kpi': dictionary of KPI metrics for each result
        """
        rows = []
        for seed in sorted(experiment.results.keys()):
            result = experiment.results[seed] # ensure ordering
            kpi_dict = {
                "inference_time": result.kpis.inference_time,
                "inference_memory_usage": result.kpis.inference_memory_usage,
                "loss": result.kpis.loss,
                "mean_rewards": result.kpis.mean_rewards,
                "training_memory_usage": result.kpis.training_memory_usage,
                "training_time": result.kpis.training_time,
                "flops": result.kpis.flops,
                "parameters": result.kpis.parameters,
            }
            rows.append({
                "name": experiment.name,
                "config": str(experiment.config),
                "kpi": kpi_dict
            })

        return pd.DataFrame(rows)

    @staticmethod
    def save_experiment_suite_to_df(experiment_suite):
        """
        Combines the DataFrames of all experimentsOld in the suite into one DataFrame.
        """
        all_dfs = []
        for experiment in experiment_suite.experiments:
            df = ExperimentUtils.save_experiment_to_df(experiment)
            all_dfs.append(df)

        return pd.concat(all_dfs, ignore_index=True)

    def mapping_strategy_names(self, df):
        mapping_dict = {
            "concat": "CONCAT",
            "mlp": "MLP",
            "mlp_local": "MLP_LOCAL",
            "mlp_global": "MLP_GLOBAL",
            "graph_gat": "GAT",
            "graph_gat_v2": "GATv2",
            "graph_sage": "GSAGE",
            "set_transformer_inv": "SET",
            "set_transformer_og": "SET_OG",
            "sab_transformer": "SAB",
            "isab_transformer": "ISAB",

        }
        df['name'] = df['name'].replace(mapping_dict)
        return df

    def load_df_from_file(self, mapping):
        if self.path is None or not os.path.exists(self.path):
            raise FileNotFoundError("The specified path is invalid or file does not exist.")
        df = pd.read_csv(self.path, converters={"kpi": eval, "config": str})
        if mapping: df = self.mapping_strategy_names(df)
        return df


    def save_df_to_file(self):
        if self.path is None:
            raise ValueError("Path not specified for saving DataFrame.")
        self.df.to_csv(self.path, index=False)

    def _is_kpi_list(self, kpi_name):
        try:
            # Helper method to check if the KPI is a list based on first occurrence
            first_kpi = self.df.iloc[0]["kpi"]
            return isinstance(first_kpi[kpi_name], list)
        except KeyError: return False

    @staticmethod
    def aggregate_kpi_across_runs(kpi_values, confidence=0.9):
        arr = np.array(kpi_values)
        mean = np.mean(arr)
        stderr = stats.sem(arr)
        margin = stderr * stats.t.ppf((1 + confidence) / 2., len(arr) - 1)
        return mean, margin

    def plot_experiment_suite_df(self, confidence=0.90, max_x_tick=None, xlabel=None,
                                 extract=True, y_limits=None, legend=None, cols=1, remove_legend=False,
                                 logscale=False, rotate=False, shorten=False, leg_location='best'):
        kpi_names = self.all_kpis
        kpi_names.remove("inference_memory_usage") if "inference_memory_usage" in kpi_names else None
        kpi_names.remove("mmac_params") if "mmac_params" in kpi_names else None

        for kpi_name in kpi_names:
            plt.figure(figsize=(10, 6))
            plt.grid(True)
            plt.xlabel("Iterations" if self._is_kpi_list(kpi_name) else "Experiment")
            if kpi_name == 'mean_rewards':
                plt.ylabel("Mean Reward") # the mean of all agents’ episode rewards across all runs -> the mean reward an agents gets in the env
            elif kpi_name == 'flops':
                    plt.ylabel("FLOPs (millions)")
            elif kpi_name == 'parameters':
                    plt.ylabel("Parameter (thousands)")
            elif kpi_name == 'training_time':
                    plt.ylabel("Training Time (s)")
            else:
                plt.ylabel(kpi_name.replace('_', ' ').capitalize())
            # plt.title(f"{kpi_name.replace('_', ' ').capitalize()} ({int(confidence * 100)}% CI)")

            for name in self.df["name"].unique():
                subset = self.df[self.df["name"] == name]["kpi"]
                try:
                    kpi_runs = [entry[kpi_name] for entry in subset]
                except KeyError:
                    print('Key not found in dataframe:')
                    print(kpi_name)
                    continue

                name = name.replace('MLP', 'DS')
                name = name.replace('transformer_global', 'Transformer')
                if shorten:
                    name = name.replace('GLOBAL', 'GL')
                    name = name.replace('LOCAL', 'LO')
                    name = name.replace('Transformer', 'Trans')

                if isinstance(kpi_runs[0], list):
                    # Per-epoch KPI
                    kpi_array = np.array(kpi_runs)
                    mean = np.mean(kpi_array, axis=0)
                    stderr = stats.sem(kpi_array, axis=0)
                    margin = stderr * stats.t.ppf((1 + confidence) / 2., len(kpi_array) - 1)

                    x = np.arange(len(mean))
                    plt.grid(True)

                    if legend is not None:
                        match = re.search(r'\d+', name).group()
                        label = legend + '=' + match
                        plt.plot(x, mean, label=label)
                    else:
                        plt.plot(x, mean, label=name)
                    plt.fill_between(x, mean - margin, mean + margin, alpha=0.3)
                    if max_x_tick is not None:
                        plt.xlim(0, max_x_tick)
                    else:
                        plt.xlim(0, len(mean) - 1)
                    plt.grid(True)

                    if kpi_name == "mean_rewards" and not y_limits:
                        plt.ylim(-20, 120)  # Consistent y-axis across plots
                    elif kpi_name == "mean_rewards" and y_limits:
                        plt.ylim(y_limits)

                else:
                    # Scalar KPI
                    mean, margin = self.aggregate_kpi_across_runs(kpi_runs, confidence)
                    match = re.search(r'\d+', name)
                    if match and extract: name = int(match.group())

                    # Special formatting for 'inference_time'
                    if kpi_name == 'inference_time':
                        mean *= 1000  # Convert to milliseconds
                        margin *= 1000
                        mean = mean
                        margin = margin
                        plt.ylabel("Inference Time (ms)")

                    sns.barplot(x=[name], y=[mean], yerr=[margin], capsize=0.2)
                    if xlabel is not None:
                        plt.xlabel(xlabel)
                    else:
                        plt.xlabel('Experiment')
                    if logscale:
                        plt.yscale('log')
                    # plt.xlabel(kpi_name)
                    if rotate:
                        plt.xticks(rotation=20)



            plt.legend(ncol=cols, loc=leg_location) if self._is_kpi_list(kpi_name) else None
            if remove_legend:
                legend = plt.gca().get_legend()
                if legend is not None:
                    legend.remove()

            plt.gca().set_axisbelow(True)
            plt.tight_layout()
            plt.show()


    @staticmethod
    def get_plt_params():
        return plt.rcParams

    @staticmethod
    def format_with_margin(mean, margin, threshold=0.01):
        if abs(mean) < threshold and abs(margin) < threshold:
            return f"{mean:.2e} ± {margin:.2e}"  # Both small
        elif abs(mean) < threshold and abs(margin) >= threshold:
            return f"{mean:.2e} ± {margin:.2f}"  # Mean small, margin large
        elif abs(mean) >= threshold and abs(margin) < threshold:
            return f"{mean:.2f} ± {margin:.2e}"  # Mean large, margin small
        else:
            return f"{mean:.2f} ± {margin:.2f}"  # Both large

    def create_table_with_confidence(self, confidence=0.90):
        """
        Create a table (Pandas DataFrame) showing the mean and confidence interval for each KPI,
        extracted from the internal experiment suite DataFrame.
        """
        kpi_names = self.all_kpis
        kpi_names.remove("inference_memory_usage") if "inference_memory_usage" in kpi_names else None
        kpi_names.remove("mmac_params") if "mmac_params" in kpi_names else None

        data = []

        for name in self.df["name"].unique():
            row = {"Experiment": name}
            subset = self.df[self.df["name"] == name]["kpi"]

            for kpi_name in kpi_names:
                try:
                    kpi_values = [entry[kpi_name] for entry in subset if entry[kpi_name] is not None]
                except KeyError:
                    print('Key not found in dataframe:')
                    print(kpi_name)
                    continue

                if not kpi_values:
                    row[kpi_name] = "N/A"
                    continue

                # Use last element of list if KPI is a list
                if isinstance(kpi_values[0], list):
                    kpi_values = [v[-1] if v else np.nan for v in kpi_values]

                try:
                    mean, margin = self.aggregate_kpi_across_runs(kpi_values, confidence=confidence)
                    row[kpi_name] = ExperimentUtils.format_with_margin(mean, margin)
                except Exception:
                    row[kpi_name] = "Error"

            data.append(row)

        df = pd.DataFrame(data)
        return df

    def create_table_verbose(self):
        kpi_names = self.all_kpis
        kpi_names.remove("inference_memory_usage") if "inference_memory_usage" in kpi_names else None
        kpi_names.remove("mmac_params") if "mmac_params" in kpi_names else None

        data = []

        for name in self.df["name"].unique():
            row = {"Experiment": name}
            subset = self.df[self.df["name"] == name]["kpi"]

            for kpi_name in kpi_names:
                try:
                    kpi_values = [entry[kpi_name] for entry in subset if entry[kpi_name] is not None]
                except KeyError:
                    print('Key not found in dataframe:')
                    print(kpi_name)
                    continue

                if not kpi_values:
                    row[kpi_name] = "N/A"
                    continue

                # Use last element of list if KPI is a list
                if isinstance(kpi_values[0], list):
                    kpi_values = [v[-1] if v else np.nan for v in kpi_values]
                    row[kpi_name] = kpi_values
                else:
                    row[kpi_name] = kpi_values

            data.append(row)


        df = pd.DataFrame(data)
        return df


