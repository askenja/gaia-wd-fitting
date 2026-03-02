#!/bin/bash
#SBATCH --job-name=wd_mcmc                       # Job name
#SBATCH --output=./output/mcmc_pop5/test_mcmc.out      # Standard output log
#SBATCH --error=./output/mcmc_pop5/test_mcmc.err       # Standard error log
#SBATCH --ntasks=1                                # Single process (for multiprocessing Pool)
#SBATCH --cpus-per-task=42                        # Number of CPU cores for Pool
#SBATCH --mem=16G                                 # Memory for your job
#SBATCH --partition=test24                        # Chosen partition; check timelimit, # test24

# Load path tp libffi
export LD_LIBRARY_PATH=$HOME/software/libffi/lib64:$LD_LIBRARY_PATH

# Activate your environment
source ~/venv/env02/bin/activate

# Move to the working directory
cd ~/work/model_tests/8_Test_WDs

# Run your Python script
python mcmc_run.py

# Copy scripts to output 
scp mcmc_run.py ./output/mcmc_pop5/run_script.py
scp test_mcmc.slurm ./output/mcmc_pop5/job_script.slurm
