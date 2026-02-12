#!/bin/bash

# activate conda
eval "$(conda shell.bash hook)"
conda activate ../.conda/envs/myenv
export PYTHONPATH=/home/iailab73/khanm2/reglangs-cl
export COMPUTERNAME=iailab73

# run the python scripts
# parity check
python scripts/create_problems.py --config_file config_create_problems_paritycheck.yml &
# even pairs
python scripts/create_problems.py --config_file config_create_problems_evenpairs.yml &
# first a
python scripts/create_problems.py --config_file config_create_problems_firsta.yml &
# last a
python scripts/create_problems.py --config_file config_create_problems_lasta.yml
# modular arithmetic
python scripts/create_problems.py --config_file config_create_problems_modarithmetic.yml &
# cycle navigation
python scripts/create_problems.py --config_file config_create_problems_cyclenavigation.yml &
# cycle navigation S
python scripts/create_problems.py --config_file config_create_problems_cyclenavigation_small.yml &
# cycle navigation L
python scripts/create_problems.py --config_file config_create_problems_cyclenavigation_large.yml
# cycle navigation One Direction
python scripts/create_problems.py --config_file config_create_problems_cyclenavigation_one_direction.yml &
# cycle navigation One Direction L
python scripts/create_problems.py --config_file config_create_problems_cyclenavigation_large_one_direction.yml &
# 252
python scripts/create_problems.py --config_file config_create_problems_recognition_252.yml &
# 333
python scripts/create_problems.py --config_file config_create_problems_recognition_333.yml
# 355
python scripts/create_problems.py --config_file config_create_problems_recognition_355.yml &
# reverse string
python scripts/create_problems_no_automata.py --config_file config_create_problems_reversestring.yml &
# bucket sort
python scripts/create_problems_no_automata.py --config_file config_create_problems_bucketsort.yml &

# create the dfa state set strategy for each problem
python scripts/create_dfa_strategy.py