#!/bin/bash

# activate conda
eval "$(conda shell.bash hook)"
conda activate ../.conda/envs/myenv
export PYTHONPATH=/home/iailab73/khanm2/reglangs-cl
export PYTHONPATH=/home/iailab73/khanm2/reglangs-cl
export COMPUTERNAME=iailab73

# run the python scripts
# python tests/test_rnn.py --with_torch True --model lstm --threads 32
# #
# python tests/test_rnn.py --with_torch True --model rnn --threads 32

python hyperparameter_search/hyper_search.py --with_torch True --model transformer --threads 32