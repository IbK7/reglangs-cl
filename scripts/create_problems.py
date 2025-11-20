import os
import sys
import yaml
import click
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from joblib import Parallel, delayed

from utils.util import TorchRandomSeed
from problems.problem import create_better_problems
from problems.regular import RandomAutomaton, FiniteAutomaton
from defaultvalues import DATASET_PATH, PROJECT_PATH, RESULT_PATH

implemented_problems = ["paritycheck", "evenpairs", "firsta", "lasta", "cyclenavigation", "recognition", "modulararithmetic"]


def create_problem(config, s):
    with TorchRandomSeed(s):
        np.random.seed(s)
        # dataset parameters
        train_batch_size = config["dataset"]["training_set_size"]
        unseen_batch_size = config["dataset"]["unseen_batch_size"]
        max_train_length = config["dataset"]["max_train_length"]
        max_length = config["dataset"]["max_length"]

        # save directory
        save_dir = Path(config["save_dir"])
        if config["problem"] == "paritycheck":
            automaton = FiniteAutomaton.from_yaml(config["problem_file"])
        elif config["problem"] == "evenpairs":
            automaton = FiniteAutomaton.from_yaml(config["problem_file"])
        elif config["problem"] == "firsta":
            automaton = FiniteAutomaton.from_yaml(config["problem_file"])
        elif config["problem"] == "lasta":
            automaton = FiniteAutomaton.from_yaml(config["problem_file"])
        elif config["problem"] == "cyclenavigation":
            automaton = FiniteAutomaton.from_yaml(config["problem_file"], zero_is_final_state=True)
        elif config["problem"] == "modulararithmetic":
            automaton = FiniteAutomaton.from_yaml(config["problem_file"], zero_is_final_state=True)
        elif config.get("problem", None) is None or config["problem"] == "recognition":
            # automaton parameters
            symbols = config["symbols"]
            num_states = config["num_states"]
            automaton = RandomAutomaton(symbols, num_states, config.get("ratio_max_path", 0.5),
                                        config.get("ratio_final_states", 0.5),
                                        config.get("ratio_max_connectivity", 1.0),
                                        final_states=config.get("final_states", None))
        else:
            raise ValueError(f"Problem is {config['problem']} should be in {implemented_problems}.")

        run_folder = (f"{len(automaton.sigma)}_{len(automaton.Q)}_{config['dataset']['max_length']}_"
                      f"{len(automaton.F)}_{s}")

        if Path(save_dir / run_folder).exists():
            pass
        else:
            create_better_problems(automaton,
                                   max_length,
                                   max_train_length,
                                   train_batch_size,
                                   unseen_batch_size,
                                   save_dir / run_folder,
                                   sampling=config.get("sampling", "uniform-length"),
                                   rng=s)


# use click to choose the config file
@click.command()
@click.option('--config_file', default="config_create_problems_cyclenavigation.yml", type=str)
def main(config_file):
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except NameError or ModuleNotFoundError:
        pass

    # load config file using yaml
    # get absolute python path
    absolute_path = os.path.dirname(os.path.abspath(__file__))
    with open(f"{absolute_path}/configs/{config_file}", "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    seed = config["dataset"]["seed"]
    config["save_dir"] = DATASET_PATH / config["save_folder"]
    number_of_problems = config["dataset"].get("num_problems", 1)

    # create number of problems random seeds
    np.random.seed(seed)
    seeds = np.random.choice(range(0, 500), size=number_of_problems, replace=False)
    seeds = seeds.tolist()

    if config["parallel"]:
        # use joblib to parallelize the creation of the problems
        Parallel(n_jobs=config['n_jobs'])(
            delayed(create_problem)(config, s) for s in tqdm(seeds))
    else:
        for s in tqdm(seeds):
            create_problem(config, s)


if __name__ == "__main__":
    main()
