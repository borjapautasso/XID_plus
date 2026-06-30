#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --partition=short
#SBATCH --exclude=artemis-general-[0-1]
#SBATCH --output=/mnt/pact/bp259/prima/xid_plus/SLURM_OUTPUTS/prior/xid-%A.out

# Print which node this is running on (no real point for prior tbh)
echo "Running on $SLURM_NODELIST"

if [ -n "$VIRTUAL_ENV" ]; then
    deactivate
fi

module purge
module load Python/3.12

cd ~/prima/xid_plus

. .xid/bin/activate

python run_prior.py