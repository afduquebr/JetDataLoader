#!/bin/bash
#SBATCH -o ./logs/preprocess.%j.log    # Output file for both stdout and stderr
#SBATCH --mem=128G                     # Memory allocation
#SBATCH --time=96:00:00                # Walltime (96 hours)
#SBATCH --ntasks=2                     # Number of tasks
#SBATCH --cpus-per-task=2              # CPUs per task
#SBATCH --job-name=preprocess          # Job name
#SBATCH --partition=htc                # Partition name
#SBATCH --account=atlas                # Account name


# Load Conda environment
module add conda

# Define and activate Conda environment
conda_env="/sps/atlas/a/aduque/conda/weaver"
if ! conda activate "$conda_env"; then
    echo "Error: Failed to activate Conda environment."
    exit 1
fi

# Go to directory
cd /pbs/home/a/aduque/private/JetDataLoader/utils || exit

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