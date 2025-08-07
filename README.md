# Intro
Official implementation of the GLOBECOM 2025 paper [Robust Bandwidth Estimation for Real-Time Communication with Offline Reinforcement Learning](https://arxiv.org/abs/2507.05785) (offline part)

## Quick start
- Download training data and configure the environment according to Schaferct's instructions: [README_Schaferct](https://github.com/jiu2021/RBWE_offline/blob/main/README_Schaferct.md)

- We mapped the action space, restructured the dataset, and provided data examples in dir: training_dataset_pickle

- Run the training script:
    ```bash 
    cd code
    python riql6_ensemble.py