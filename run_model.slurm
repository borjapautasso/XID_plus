#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
# SBATCH --array=0-3583
# SBATCH --partition=short,general
# SBATCH --time=2:00:00
#SBATCH --array=3072-3327
#SBATCH --partition=short,general
#SBATCH --time=2:00:00
# SBATCH --array=2304-3071
# SBATCH --partition=short
#SBATCH --output=SLURM_OUTPUTS/modelling/%A/xid-%A_%a.out

python run_model.py $SLURM_ARRAY_TASK_ID