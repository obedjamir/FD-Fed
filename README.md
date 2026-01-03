# FD-Fed: Feature-Driven Layer Specialization for Label-Heterogeneous Federated Learning

This repository provides the official implementation of **FD-Fed**, a federated learning framework designed to address **label heterogeneity** across distributed clients through **feature-driven layer specialization**. The code supports experiments on both **medical** and **natural image datasets** and enables fair comparison with several personalized federated learning baselines.

---

## Dataset Preparation

The `dataset/` directory contains scripts to generate datasets for federated learning experiments. Each dataset has a dedicated script that prepares client-wise data splits suitable for label-heterogeneous settings.

### Available Dataset Scripts

- generate_CheXpert.py
- generate_MIMIC.py
- generate_NIHChestXray.py
- generate_cifar10.py
- generate_cifar100.py
- generate_cinic10.py

### Generating a Dataset

From the `dataset/` directory, run:

    python3 <SCRIPT_NAME>

Example:

    python3 generate_NIHChestXray.py

Within each dataset script, the following parameters can be modified:

- Number of clients
- Number of runs
- Dataset split configuration
- Output directory paths

All parameters are defined directly inside the corresponding script.

---

## Running Experiments

All federated learning experiments are executed from the `system/` directory.

### Main Components

- main.py: Core federated learning framework
- run.sh: Script to launch experiments sequentially

### Running Experiments

    cd system
    ./run.sh

Hyperparameters for each method and dataset can be adjusted directly inside `run.sh`.

---

## Example Command

Below is an example command for running a single experiment using **FD-Fed**. Similar commands are used for other methods and datasets.

    nohup python3 main.py \
      -algo OursGCAM \
      -lr 0.01 \
      -al 0.001 \
      -m effnet \
      -mn effnet \
      -lam 0.35 \
      -th 2 \
      -nc 3 \
      -data nihchestxray \
      -t 5 \
      -go experiment \
      -gpu 0 \
      > OursGCAM_nihchestxray.log 2>&1 &

---

## Supported Methods

- FD-Fed (proposed)
- FedBABU
- FedPer
- FedRep
- FedPav
- Local training

---

## Notes

- Experiments are executed asynchronously using `nohup`.
- Training logs are saved to `.log` files for later inspection.
- GPU selection is controlled via the `-gpu` argument.
