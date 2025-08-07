# Intro
Official implementation of the GLOBECOM 2025 paper [Robust Bandwidth Estimation for Real-Time Communication with Offline Reinforcement Learning](https://arxiv.org/abs/2507.05785) (offline part)

## Quick start
- Download training data and configure the environment according to Schaferct's instructions: [README_Schaferct](https://github.com/jiu2021/RBWE_offline/blob/main/README_Schaferct.md)

- We mapped the action space, restructured the dataset, and provided data examples in dir: training_dataset_pickle

- Run the training script:
    ```bash 
    cd code
    python riql6_ensemble.py

## Offline eval
1. To run a small evaluation on a [small dataset](https://github.com/microsoft/RL4BandwidthEstimationChallenge/tree/main/data): (download the 24 sessions and modify their path first)
        
    ```bash
    cd code
    python detail_evaluate_on_24_sessions.py
    ```
        
2. To evaluate the metrics (mse, errorate) over all evaluation dataset:

    ```bash
    cd code
    python evaluate_all.py
    ```
        
The whole evaluate process takes more than 10 hours.

## Online eval
For online evaluation, we developed it based on the Pandia platform, and we also sourced the deployment code: [Pandia](https://github.com/jiu2021/Pandia)
1. Clone repo and install packets

2. Run the script
    ```bash
    python -m pandia.agent.env_emulator_offline