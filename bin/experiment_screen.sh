#!/bin/bash

# Array of dataset names
datasets=("hppb" "hlm")
# Array of adaptation methods
methods=("KMM" "KLIEP" "TrAdaBoostR2")

# Loop through each dataset and adaptation method
for dataset in "${datasets[@]}"; do
    for method in "${methods[@]}"; do
        echo "=============================================="
        echo "Dataset: ${dataset} | Method: ${method}"
        echo "----------------------------------------------"

        # reset timer
        SECONDS=0

        # run the experiment, with special flag for solubility
        if [ "$dataset" == "solubility" ]; then
            python domain_adaptation.py \
                --dataset_name "${dataset}" \
                --adaptation_method "${method}" \
                --num_processes 4
        else
            python domain_adaptation.py \
                --dataset_name "${dataset}" \
                --adaptation_method "${method}"
        fi

        # capture and print elapsed time
        duration=$SECONDS
        echo ">> Time taken: ${duration} seconds"
        echo
    done
done