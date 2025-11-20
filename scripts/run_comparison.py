import os
import sys
import copy
import yaml
import time
import click
import torch
import shutil
import itertools
import numpy as np
from tqdm import tqdm
from pathlib import Path
from joblib import Parallel, delayed

from train_models import train
from utils.dataset import count_non_padding_batch
from defaultvalues import DATASET_PATH, RESULT_PATH
from utils.util import set_random_seeds, get_unique_path
from problems.problem import load_single_problem, RecognitionDataset, load_compression_data, load_language


def problem_run(prob_id, hyperparameter_problem, config, model_config):
    prob, curr, excl, cwb, s, model_seed = hyperparameter_problem
    # distribute the problems over the available cuda devices
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if "device" in config and config["device"] == "cpu":
        device = torch.device("cpu")
    if device == torch.device("cuda"):
        num_cuda_devices = torch.cuda.device_count()
        if num_cuda_devices > 1:
            device_id = prob_id % num_cuda_devices
            device = torch.device(f"cuda:{device_id}")

    base_path = Path(config["base_path"])

    # Define the dataset
    train_ins, train_outs, language = load_single_problem(prob)
    unseen_ins, unseen_outs = load_single_problem(prob, unseen=True)
    train_compression, unseen_compression = load_compression_data(prob)
    if language is not None:
        alphabet_size = len(language.sigma)
    else:
        # get first column entries of train_ins
        first_column = train_ins[:, 0]
        unique_objects = torch.unique(first_column)
        alphabet_size = len(unique_objects)
    unseen_dataset = RecognitionDataset(unseen_ins, unseen_outs, count_non_padding_batch(unseen_ins), alphabet_size)

    # Define the parameters for the model and training
    params = config["params"]
    params['experiment_id'] = config.get('experiment_id', None)
    params["problem"] = prob
    params["exclusive"] = excl
    params["continue_with_best"] = cwb
    params["seed"] = s
    params["model_seed"] = model_seed

    model_params = model_config["model_params"]

    data = {'language': language,
            'unseen_data': unseen_dataset,
            'train_compression': train_compression,
            'unseen_compression': unseen_compression}

    for model in params["models"]:
        params["model"] = model
        # DEFAULT: load fixed data splits from disk
        data["train_data"] = train_ins, train_outs
        # get 1000 random indices for the validation set
        # set the seed to the model seed
        np.random.seed(model_seed)
        val_indices = np.random.choice(np.arange(len(train_ins)), 1000)
        val_ins, val_outs = train_ins[val_indices], train_outs[val_indices]
        data["val_data"] = val_ins, val_outs
        params["training_strategy"] = curr
        params["run_dir"] = f"{curr}_c{'T' if cwb else 'F'}_m{model_seed}_s{s}"

        experiment_path = Path(base_path) / config["experiment_id"] / "_".join(
            [params["model"], str(Path(prob).name)]) / params["run_dir"]

        print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}: Starting {params['run_dir']} with model {params['model']} for problem {prob}")
        # start time
        start_time = time.time()
        # this experiment was already run
        if experiment_path.exists() and (experiment_path / "results.json").exists():
            pass
        # this experiment was started, but never finished; start anew
        elif experiment_path.exists():
            shutil.rmtree(experiment_path)
            train(base_path, data, copy.deepcopy(params), copy.deepcopy(model_params), config, device=device)
        # train and evaluate the model
        else:
            train(base_path, data, copy.deepcopy(params), copy.deepcopy(model_params), config, device=device)

        # print the time taken to run the experiment
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}: {params['run_dir']} took "
              f"{round((time.time() - start_time) / 60, 2)} minutes")
    return experiment_path

def patch_config(config, strategies, num_buckets, num_updates, dataset, num_model_seeds, num_gpus, models,
                 max_input_length, metric, prefix):
    # Patch the config with the provided parameters
    if strategies:
        config["strategies"] = strategies.split(',')
    if num_buckets is not None:
        config["params"]["num_buckets"] = num_buckets
    if num_updates:
        config["params"]["num_updates"] = num_updates
    if dataset:
        config["dataset_folder"] = "_".join([dataset, config["dataset_folder"]])
    else:
        raise ValueError("Please specify the dataset")
    if num_model_seeds is not None:
        config["n_model_seeds"] = num_model_seeds
    if num_gpus is not None:
        config["parallel"] = True
        config["n_jobs"] = num_gpus
    if models:
        config['params']['models'] = models.split(',')
    if max_input_length is not None:
        config['params']['max_input_length'] = max_input_length
    if metric:
        config["metric"] = metric
    if prefix:
        config["experiment_id_prefix"] = prefix
    return config

def create_experiment_id(config):
    if config.get("experiment_id_prefix", None):
        exp_id = "_".join([config["experiment_id_prefix"],
                                            f"{config['params']['num_updates'] // 1000}k",
                                            config["metric"],
                                            str(config["params"]["num_buckets"]),
                                            config["dataset_folder"]])
    elif config["params"].get("max_input_length", None):
        exp_id = "_".join([f"{config['params']['max_input_length']}max",
                                            f"{config['params']['num_updates'] // 1000}k",
                                            config["metric"],
                                            str(config["params"]["num_buckets"]),
                                            str(config['params']["max_input_length"]),
                                            config["dataset_folder"]])
    else:
        exp_id = "_".join([config["metric"],
                           f"{config['params']['num_updates'] // 1000}k",
                           str(config["params"]["num_buckets"]),
                           config["dataset_folder"]])
    return exp_id

@click.command()
@click.option('--config_file', default="config_run_comparison_recognition_252.yml", type=str)
@click.option('--strategies', default=None, type=str, help='Comma-separated list of strategies to use, e.g., "RNN,LSTM"')
@click.option('--num_buckets', default=None, type=int)
@click.option('--num_updates', default=None, type=int)
@click.option('--dataset', default='uniform-length', type=str)
@click.option('--num_model_seeds', default=None, type=int)
@click.option('--num_gpus', default=None, type=int)
@click.option('--models', default=None, type=str, help='Comma-separated list of models to use, e.g., "RNN,LSTM"')
@click.option('--max_input_length', default=None, type=int, help='Maximum input length for generated samples')
@click.option('--metric', default=None, type=str)
@click.option('--prefix', default=None, type=str)
def main(config_file, strategies, num_buckets, num_updates, dataset, num_model_seeds, num_gpus, models, max_input_length, metric, prefix):
    # get absolute python path
    absolute_path = os.path.dirname(os.path.abspath(__file__))
    # add config_file to the path
    config_filename = os.path.basename(config_file)
    config_path = os.path.join(absolute_path, "configs", config_filename)
    model_config_path = os.path.join(absolute_path, "configs", "models.yml")
    print(f"Starting run_comparison.py for {config_filename}")

    # Open and read the config file
    with open(config_path, "r") as config_file_obj:
        try:
            config = yaml.load(config_file_obj, Loader=yaml.FullLoader)
        except:
            raise ValueError("Error reading the config file")

    with open(model_config_path, "r") as model_config_file_obj:
        try:
            model_config = yaml.load(model_config_file_obj, Loader=yaml.FullLoader)
        except:
            raise ValueError("Error reading the model config file")

    # Patch the config with the provided parameters
    config = patch_config(config, strategies, num_buckets, num_updates, dataset, num_model_seeds, num_gpus, models, max_input_length, metric, prefix)

    config["datasets_path"] = DATASET_PATH / config["dataset_folder"]
    config["base_path"] = RESULT_PATH

    # Define the problems (datasets)
    # get a list of all folders in the datasets directory as string
    problems = [str(x) for x in Path(config["datasets_path"]).iterdir() if x.is_dir()]
    problems = problems[0:config.get("num_problems", len(problems))]

    # generate a list of seeds from the seed
    set_random_seeds(config["seed"])
    seeds = np.random.choice(range(0, 100), size=config["n_data_seeds"], replace=False)
    seeds = seeds.tolist()

    # generate a list of seeds from the seed
    set_random_seeds(config["seed"])
    model_seeds = np.random.choice(range(0, 100), size=config["n_model_seeds"], replace=False)
    model_seeds = model_seeds.tolist()

    # create experiment id based on parameters
    config['experiment_id'] = create_experiment_id(config)
    comparison_over = list(itertools.product(problems,
                                             config['strategies'],
                                             config['params']['exclusive'],
                                             config['params']['continue_with_best'],
                                             seeds,
                                             model_seeds
                                             ))

    # create directory under Path(base_path) / config["experiment_id"] and save the config file
    parent_dir = Path(config["base_path"]) / config["experiment_id"]
    parent_dir.mkdir(parents=True, exist_ok=True)
    print(f"##########\n##########\nStarting experiments for parent directory: {parent_dir}\n##########\n##########")

    list_of_processed_problems = []
    if config["parallel"]:
        # parallelize the training over the problems using joblib,
        # joblib should wait for all processes to finish before starting the next iteration
        list_of_processed_problems = Parallel(n_jobs=config['n_jobs'])(
            delayed(problem_run)(p_id, p, config, model_config) for p_id, p in enumerate(tqdm(comparison_over)))
    else:
        for p_id, p in enumerate(tqdm(comparison_over)):
            result = problem_run(p_id, p, config, model_config)
            list_of_processed_problems.append(result)

    print(f"Time taken: {time.time() - start}")
    config["result_dirs"] = list_of_processed_problems

    config["params"].pop("train_data", None)
    config["params"].pop("val_data", None)
    config["params"].pop("unseen_data", None)
    config["params"].pop("test_data", None)
    config["params"].pop("train_compression", None)
    config["params"].pop("val_compression", None)
    config["params"].pop("test_compression", None)
    config["params"].pop("unseen_compression", None)
    config["params"].pop("language", None)

    # Save the config file to the new directory
    unique_config_path = get_unique_path(parent_dir, Path(config_filename))
    with open(unique_config_path, "w") as new_config_file_obj:
        yaml.dump(config, new_config_file_obj)


if __name__ == "__main__":
    # measure the time it takes to run the script
    start = time.time()
    main()
