# %%
import os
import sys
import click
import pickle
import warnings
from problems.problem import load_single_problem
from utils.dataset import get_state2index_dict
from defaultvalues import DATASET_PATH
"""
Script that will traverse the RegularLanguages dataset and create the dfa strategy for each automaton
if it doesn't exist yet. Based on the structure of the automaton and the following statistics:
* first bucket will contain always state 0
* First bucket should already contain states in such a way that they form a loop in the automaton
* Each following bucket should add the minimum number of states in such a way that no dead ends are created
"""

warnings.simplefilter(action='ignore', category=FutureWarning)

# %%
@click.command()
@click.option('--root_dir', type=str, default=DATASET_PATH)
def main(root_dir):
    # Traverse the root directory
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Check if "dfa_strategy.json" exists in subdirectories
        if dirpath != root_dir and ("dfa.json" not in filenames or "unseen_states2idx.pkl" not in filenames):
            train_ins, train_outs, automaton = load_single_problem(dirpath)
            del train_outs
            dfa_path = os.path.join(dirpath, 'dfa.json')
            if not os.path.exists(dfa_path):
                automaton.create_dfa_strategy(savepath=os.path.join(dirpath, 'dfa.json'))

            train_file_path = os.path.join(dirpath, 'train_states2idx.pkl')
            if not os.path.exists(train_file_path):
                train_states2idx = get_state2index_dict(train_ins, automaton)
                with open(train_file_path, 'wb') as f:
                    pickle.dump(train_states2idx, f, protocol=pickle.HIGHEST_PROTOCOL)
                print(f"Done creating train_states2idx")
                del train_ins
                del train_states2idx

            unseen_file_path = os.path.join(dirpath, 'unseen_states2idx.pkl')
            if not os.path.exists(unseen_file_path):
                unseen_ins, unseen_outs = load_single_problem(dirpath, unseen=True)
                unseen_states2idx = get_state2index_dict(unseen_ins, automaton)
                with open(unseen_file_path, 'wb') as f:
                    pickle.dump(unseen_states2idx, f, protocol=pickle.HIGHEST_PROTOCOL)
                del unseen_ins
                del unseen_states2idx

        elif dirpath != root_dir and "dfa.json" in filenames and "unseen_states2idx.pkl" in filenames:
            print(f"DFA strategy already exists for {dirpath}")


if __name__ == '__main__':
    main()