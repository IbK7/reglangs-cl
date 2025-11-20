# Exploring Curriculum Learning for Languages: Lessons from Regular Language Tasks

This repository contains the code the reproduce the results for the paper "Exploring Curriculum 
Learning for Languages: Lessons from Regular Language Tasks". This covers data generation for 
the different problems, hyperparameter search as described in the paper, and training scripts.

## Installation
To project was done with Python 3.12. Make sure to include the following packages in your conda 
environment:
- numpy
- pandas
- matplotlib
- scikit-learn
- seaborn
- pytorch
- tqdm
- click
- pygraphviz (for visualizing automata)
- networkx

## Before getting started
Go to `defaultvalues.py` and set the default values for your PROJECT_PATH (top level folder of the 
repository), DATASET_PATH (where the data will be stored), and RESULT_PATH (where your results 
will be saved). It makes sense to use COMPUTERNAME if working on different machines parallely. 

## Project Structure
- `hyperparameter_search/`: contains the actual hyperparameter search currently implemented for 
  RNNs and LSTMs as well as a script to aggregate the results (`eval_hyper_search.py`)
- `models/`: contains the model implementations for RNNs and LSTMs
- `problems/`: contains the class defining the Regular Language tasks as well as all 
  yaml files and helper scripts that specify the different problems.
- `scripts/`: contains scripts for data generation (`create_problems.py`), strategy generation 
  (`create_dfa_strategy.py`), experiment orchestration (`run_comparison.py`), and model 
  training (`train_models.py`) as well as configuration files for the training of each problem. 
  Also, contains global configuration files for each model type (see `models.yml`).

## Top-level Scripts
This project also contains some bash scripts that show how to run the different scripts defined 
above. They are written to be run from the top-level folder. Make sure to adapt the first few 
lines where you activate your conda environment and export PYTHONPATH and COMPUTERNAME (if needed).
- `create_problems.sh`: creates the datasets for all problems as described in the paper. Will be 
  saved under DATASET_PATH.
- `hyperparameter_search.sh`: runs the hyperparameter search for both architectures.
- `main.sh`: used to run experiments. Contains examples how to call the `run_comparison.py` 
  script with different configurations (that might override the default values in the config files).