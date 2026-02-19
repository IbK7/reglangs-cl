#!/bin/bash

# activate conda
eval "$(conda shell.bash hook)"
conda activate ../.conda/envs/myenv/
export PYTHONPATH=/home/iailab73/khanm2/reglangs-cl
export COMPUTERNAME=iailab73

# cycle navigation
# python scripts/run_comparison.py --config_file config_run_comparison_cyclenavigation.yml --models "RNN, LSTM" --strategies "anti,uniform,curr,single" --metric "states"

# modular arithmetic; note: if not specified, all models and strategies are used as described in the config file
# python scripts/run_comparison.py --config_file config_run_comparison_modarithmetic.yml --num_buckets 20

# example how to run with different number of buckets
# num_buckets=(2 5 10 15 20)
# num_buckets=(2 5)

# for bucket in "${num_buckets[@]}"; do
#     # cycle navigation S
#     python scripts/run_comparison.py \
#     --config_file config_run_comparison_paritycheck.yml \
#     --strategies "curr" \
#     --num_buckets $bucket \
#     --metric "length"
# done

# python scripts/run_comparison.py --config_file scripts/configs/config_run_comparison_paritycheck.yml --models "TransformerEncoder"

# configs=(
#   "scripts/configs/config_run_comparison_cyclenavigationS.yml"
#   "scripts/configs/config_run_comparison_paritycheck.yml"
# )

# for config in "${configs[@]}"; do
#   python scripts/run_comparison.py \
#     --config_file "$config" \
#     --models "TransformerEncoder" \
#     --strategies "curr" \
#     --num_buckets 20 \
#     --metric "length"
# done


python scripts/run_comparison.py --config_file scripts/configs/config_run_comparison_evenpairs.yml --models "TransformerRelative" --strategies "curr" --num_buckets 20 --metric "length"

# python scripts/run_comparison.py --config_file scripts/configs/config_run_comparison_modarithmetic.yml --models "TransformerEncoder" --strategies "curr,no-curr" --num_buckets 20 --metric "length"

# python scripts/run_comparison.py --config_file scripts/configs/config_create_problems_cyclenavigation.yml --models "TransformerEncoder" --strategies "curr,no-curr" --num_buckets 20 --metric "length"


# python scripts/run_comparison.py --config_file scripts/configs/config_run_comparison_paritycheck.yml --models "TransformerEncoder" --strategies "no-curr" --metric "states"
# python scripts/run_comparison.py --config_file config_run_comparison_paritycheck.yml --models "RNN. LSTM" --strategies "anti,uniform,curr,single" --metric "states"