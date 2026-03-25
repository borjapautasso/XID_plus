#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH --gres=gpu:1
#SBATCH --array=0-255
#SBATCH --partition=gpu,long,general,short
#SBATCH --exclude=artemis-a40-[00-03]
#SBATCH --output=/mnt/pact/bp259/prima/xid_plus/SLURM_OUTPUTS/modelling/%A/xid-%A_%a.out



# region DEFAULTS (DO NOT TOUCH)
#######################

flux_stepwise=False
output_path=None
cirrus_structure_path=None
num_samples=500
num_warmup=500
num_chains=4
chain_method="vectorised"
output=True

#######################
# endregion



# region EDIT AS NEEDED
#######################

prior_name="euclid_wide_v2.2_order9_7"
output_name="euclid_wide_v2.2_order9_7_Stepwise"
order=9
order_large=7
id_large_tile=77828
flux_prior=1
flux_stepwise=True
# output_path=None
# cirrus_structure_path=None
# num_samples=500
# num_warmup=500
# num_chains=4
# chain_method="vectorised"
# output=True

#######################
# endregion


model_params=(
    "$prior_name"
    "$output_name"
    "$SLURM_ARRAY_TASK_ID"
    "$order"
    "$order_large"
    "$id_large_tile"
    "$flux_prior"
    "$flux_stepwise"
    "$output_path"
    "$cirrus_structure_path"
    "$num_samples"
    "$num_warmup"
    "$num_chains"
    "$chain_method"
    "$output"
    )

#
# Print which node this is running on
echo "Running on $SLURM_NODELIST"

module purge
module load Python/3.12

cd ~/prima/xid_plus

. .xid/bin/activate

# If requesting GPUs, check that requested GPUs match the output from nvidia-smi.
# Sometimes some GPUs quietly fail on artemis but are still used by the scheduler as if they work
if [[ -n "$CUDA_VISIBLE_DEVICES" ]]; then
    IFS=',' read -ra DEVICES <<< "$CUDA_VISIBLE_DEVICES"
    echo "GPUs assigned: ${#DEVICES[@]}"
    
    
    detected_gpus=$(nvidia-smi -L | grep -c "^GPU")
    
    if [[ $detected_gpus -ne ${#DEVICES[@]} ]]; then
        echo "Expected $expected_gpus GPUs, found $detected_gpus"
        echo "bad node/gpu?"
        nvidia-smi -L

        # scontrol update JobId=$SLURM_JOB_ID ExcNodeList=$(hostname)
        # scontrol requeue $SLURM_JOB_ID
        # exit 0

        exit 1
    fi
    
    echo "No GPU issues"
    
else
    echo "No GPUs assigned"

    if [ "$SLURM_CPUS_PER_TASK" -gt 4 ]; then
        NUM_DEVICES=4
    else
        NUM_DEVICES=$SLURM_CPUS_PER_TASK
    fi
    export XLA_FLAGS="--xla_force_host_platform_device_count=${NUM_DEVICES}"

fi

numactl --membind=1,2,5,6 python run_model.py "${model_params[@]}"

# python run_model.py "${model_params[@]}"
