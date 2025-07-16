#!/bin/bash
#SBATCH --job-name=k2-sgl
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1

set -x

export MODEL_NAME=moonshotai/Kimi-K2-Instruct

export MASTER_IP=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

srun -l bash -c 'set -x; python -m sglang.launch_server \
    --host 0.0.0.0 \
    --model-path $MODEL_NAME \
    --tp 16 \
    --dist-init-addr $MASTER_IP:50000 \
    --nnodes $SLURM_JOB_NUM_NODES \
    --node-rank $SLURM_NODEID \
    --trust-remote-code \
    --tool-call-parser kimi_k2'
