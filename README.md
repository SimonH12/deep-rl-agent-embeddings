# Exploring Deep Learning Methods for Embedding Agent States in Multi-Agent Reinforcement Learning

This repository contains the code and experiments conducted for the Master's thesis project: **"Exploring Deep Learning Methods for Embedding Agent States in Multi-Agent Reinforcement Learning (MARL)"**.  

**Department of Computer Science and Mathematics**  
**University of Applied Sciences Munich**

**Author:** Simon Hampp  

---
## Abstract
This thesis investigates permutation-invariant and agent-count invariant methods for centralized critics in multi-agent reinforcement learning (MARL). Conventional approaches typically concatenate all agent observations to form the critic’s input. A simple strategy, but neither permutation-invariant nor adaptable to variable agent numbers. As a result, such critics are unsuitable for dynamic environments with changing agent counts and do not exploit the potential sample efficiency gains of permutation-invariant critics.
To address these limitations, critic architectures were developed based on prior work in DeepSets, GNNs, and transformers. These methods were benchmarked and tuned across multiple environments and evaluated in terms of performance, scalability, and generalizability.
The results demonstrate that permutation-invariant critics can match, and in some cases exceed, the performance of the concatenation baseline, which remains a strong reference point. The scalability analysis shows that invariant critics handle increasing agent counts more efficiently in terms of parameters and FLOPs, with transformer-based methods scaling particularly well. Generalization experiments confirm that invariant critics can extend to unseen agent counts, provided training covers a sufficiently broad agent range. The developed methods can handle different agent counts without retraining the critic from scratch, whereas concatenation requires retraining for each configuration, which hinders its application for dynamic environments with varying agent counts during training. 
Sample efficiency gains were not apparent under the popular Centralized Training and Decentralized Execution framework, but could be observed using Centralized Training and Centralized Execution.
Overall, this thesis establishes that permutation- and agent-count invariant critics offer clear advantages for MARL in dynamic and large-scale environments. Among the tested methods, transformer-based architectures proved consistently robust and competitive, highlighting invariant critics as a promising foundation for scalable and generalizable MARL.

---
## Thesis PDF

The full Master's thesis can be downloaded or viewed here:  

[Download Thesis (PDF)](thesis/thesis.pdf)


---


## Code Reused
Parts of this repository build upon the following projects:

- [VectorizedMultiAgentSimulator (Prorok Lab)](https://github.com/proroklab/VectorizedMultiAgentSimulator)
- [PyTorch RL Multi-Agent PPO Tutorial](https://docs.pytorch.org/rl/stable/tutorials/multiagent_ppo.html)
- [Set Transformer by Juho Lee](https://github.com/juho-lee/set_transformer)

---

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/SimonH12/deep-rl-agent-embeddings.git
   cd deep-rl-agent-embeddings
   ```
2. Create conda environment from `.yml` file
    ```bash
    conda env create -f environment.yml
    conda activate ma_rl
    ```
Note: Python 3.9 is recommended for compatibility.

---

## Run Experiments
- To run the experiments, start the Jupyter notebooks located in the `/experiments_ma/` directory.
- Each subfolder (e.g., `1_concat_tune`, `2_mlp_approaches`, etc.) contains the relevant notebooks and configurations for a specific experimental method.
- The folder numbering corresponds to the respective subsections in the Master's thesis for easy cross-reference.
- To replicate the plots, run notebooks with _plot_ in their name, corresponding to the correct experiment
- Note: the naming in the thesis was changed from `MLP` to `DS` for clarity

### Run Example:
**Option A: Run the script directly**  
- You can run src/mappo.py directly through your IDE (e.g., PyCharm, VSCode) by opening the file and clicking “Run”.
- The script will also show a rollout of the policy

Alternatively:
1. Open a terminal in the project root directory.  
2. Run the following command in the root:  
```bash
python -m src.mappo
```


**Option B: Run an experiment**
- Run the Jupyter Notebook `src/experiments_ma/example.ipynb`
- All other experiments can also be tested in `src/experiments_ma`


### VS Code
When running Jupyter notebooks in VS Code, you may encounter import errors (e.g., modules in `src/` cannot be found).  
To fix this, make sure you have a `.vscode/settings.json` file in your project with the following content:

```json
{
    "python.envFile": "${workspaceFolder}/.env",
    "jupyter.notebookFileRoot": "${workspaceFolder}/src"
}
```

---

## Repository Structure
- **`experiments_ma/`**: Each folder is a numbered experiment corresponding to sections in the thesis.
- **`nn/`**: Contains all neural network code (MLP, GNN, Transformer, Set Transformer).
- **Core scripts**:
  - `mappo.py`: Implements the MAPPO training loop based on [PyTorch Tutorial](https://docs.pytorch.org/rl/stable/tutorials/multiagent_ppo.html)
  - `experiments.py`: Entrypoint to run experiments via script.
- **Custom scenarios** (`multi_give_way_com.py`, `navigation.py`) adapted [VMAS](https://github.com/proroklab/VectorizedMultiAgentSimulator)
 environments


```
├── experiments_ma                  # Experiments of MA
│   ├── 1_concat_tune
│   ├── 2_mlp_approaches
│   ├── 3_gnn_approaches
│   ├── 4_transformer_approaches
│   ├── 5_cross_method_comparison
│   ├── 6_scalability
│   └── 7_generalizability
├── nn                              # Neural Network Logic
│   ├── actor_critic
│   ├── feature_embeddings.py
│   ├── gnn
│   ├── mlp
│   ├── multiagent.py
│   ├── set_transformer
│   ├── transfer_trained_nn.py
│   └── transformer.py
├── experiments.py
├── extractor.py
├── mappo.py                        # MAPPO Logic
├── multi_give_way_com.py           # Custom Scenario
├── navigation.py                   # Custom Scenario
├── README.md
└── utils.py
```
---
# References
To cite this work, use:
```bibtex
@mastersthesis{hampp2025marl,
  author       = {Simon Hampp},
  title        = {Exploring Deep Learning Methods for Embedding Agent States in Multi-Agent Reinforcement Learning},
  school       = {University of Applied Sciences Munich, Department of Computer Science and Mathematics},
  year         = {2025},
  type         = {Master's Thesis},
  url          = {https://github.com/SimonH12/deep-rl-agent-embeddings} 
}
```
