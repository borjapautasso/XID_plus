#!/bin/bash
#SBATCH --ntasks=1
# SBATCH --cpus-per-task=1
# SBATCH --gres=gpu:1
# SBATCH --partition=gpu,long,verylong
# SBATCH --exclude=artemis-a40-[00-03]
# SBATCH --mem=16GB
# SBATCH --array=0-95
# SBATCH --array=0-11,16-27,32-43,48-59,64-75,80-91
#SBATCH --cpus-per-task=4
#SBATCH --partition=general,gpu
#SBATCH --array=704-767
#SBATCH --mem=16GB
#SBATCH --output=/mnt/pact/bp259/prima/xid_plus/SLURM_OUTPUTS/modelling/%A/xid-%A_%a.out


# EDIT AS NEEDED
################

prior_name="euclid_wide_v2.2_order10_7_fwhmexpansion"
output_name="euclid_wide_v2.2_order10_7_fwhmexpansion_default_CPU"
order=10
order_large=7
id_large_tile=77828
# flux_prior=None
# flux_stepwise=0
# output_path="None"
# cirrus_structure_path="inputs/fbm_maps/cirrus_2.3_v2.2.npy"
# num_samples=500
# num_warmup=500
# num_chains=4
# chain_method="vectorised"
# output=True
# expand_fwhm=1


model_params=(
    "${prior_name}"
    "${output_name}"
    "${SLURM_ARRAY_TASK_ID}"
    "${order}"
    "${order_large}"
    "${id_large_tile:=77828}"
    "${flux_prior:=None}"
    "${flux_stepwise:=0}"
    "${output_path:=None}"
    "${cirrus_structure_path:=None}"
    "${num_samples:=500}"
    "${num_warmup:=500}"
    "${num_chains:=4}"
    "${chain_method:=parallel}"
    "${output:=True}"
    "${expand_fwhm:=0}"
    )


#
# Print which node this is running on
echo "Running on $SLURM_NODELIST"

if [ -n "$VIRTUAL_ENV" ]; then
    deactivate
fi

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

python run_model.py "${model_params[@]}"
