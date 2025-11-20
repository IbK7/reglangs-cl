#!/bin/bash

# activate conda
eval "$(conda shell.bash hook)"
conda activate chomsky
export PYTHONPATH=/home/mlai21/share/code/chomsky_curriculum/
export COMPUTERNAME=mlai_florian

# cycle navigation
python scripts/run_comparison.py --config_file config_run_comparison_cyclenavigation.yml --models "RNN, LSTM" --strategies "anti,uniform,curr,single" --metric "states"

# modular arithmetic; note: if not specified, all models and strategies are used as described in the config file
python scripts/run_comparison.py --config_file config_run_comparison_modarithmetic.yml --num_buckets 20


# example how to run with different number of buckets
num_buckets=(2 5 10 15 20)

for bucket in "${num_buckets[@]}"; do
    # cycle navigation S
    python scripts/run_comparison.py \
    --config_file config_run_comparison_cyclenavigationS.yml \
    --strategies "curr" \
    --num_buckets $bucket \
    --metric "length"
done