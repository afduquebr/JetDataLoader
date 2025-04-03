#!/bin/bash
#PBS -o ./logs/preprocessing.$PBS_JOBID.txt
#PBS -j oe

# Source Conda environment
source /atlas/tools/anaconda/anaconda3/etc/profile.d/conda.sh

# Define the name of the Conda environment
conda_env="/AtlasDisk/user/duquebran/conda/weaver"

# Activate the Conda environment
if ! conda activate $conda_env; then
    echo "Error: Failed to activate Conda environment."
    exit 1
fi

# Go to directory
cd /AtlasDisk/home2/duquebran/JetDataLoader/utils || exit

# Run Python scripts
if ! python skim.py; then
    echo "Error: Failed to run skimming."
    exit 1
fi

if ! python reweight.py; then
    echo "Error: Failed to run reweighting."
    exit 1
fi

if ! python balance.py; then
    echo "Error: Failed to run balancing."
    exit 1
fi

# Deactivate the Conda environment
if ! conda deactivate; then
    echo "Error: Failed to deactivate Conda environment."
    exit 1
fi

echo "Script executed successfully."
exit 0