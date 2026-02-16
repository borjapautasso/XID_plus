#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --partition=short
#SBATCH --output=SLURM_OUTPUTS/prior/xid-%A.out

python run_prior.py