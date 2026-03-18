#!/bin/bash
#SBATCH --nodes=2
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node 1
#SBATCH --job-name=mdm-prime-v2
#SBATCH -p p01

#SBATCH --output=/dev/null 
#SBATCH --error=slurm.err
#SBATCH --signal=B:USR1@120

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_GPUS_ON_NODE)) # Total GPUs (16)

echo "Master: $MASTER_ADDR | Port: $MASTER_PORT | World Size: $WORLD_SIZE"
srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 \
    --export=ALL \
    --container-image mdm_prime_v2_image.sqsh \
    --container-writable \
    --container-mounts mdm-prime-v2/lit_gpt:/workspace \
    bash /workspace/pretrain_mdm_prime_v2.sh