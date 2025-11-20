#!/bin/bash

# activate conda
eval "$(conda shell.bash hook)"
conda activate RuleGNN
export PYTHONPATH=/home/mlai21/share/code/chomsky_curriculum/
export PYTHONPATH=/home/florian/Documents/Code/chomsky_curriculum/
export COMPUTERNAME=florian

# run the python scripts
python tests/test_rnn.py --with_torch True --model lstm --threads 32
#
python tests/test_rnn.py --with_torch True --model rnn --threads 32
