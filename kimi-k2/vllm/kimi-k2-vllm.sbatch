#!/bin/bash
#SBATCH --job-name=k2-vllm
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1

set -x

export MODEL_NAME=moonshotai/Kimi-K2-Instruct

export MASTER_IP=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export PORT=13579

srun -l bash -c 'set -x; 
    [ "$SLURM_PROCID" -eq 0 ] && EXTRA_FLAGS="" || EXTRA_FLAGS="--headless --data-parallel-start-rank 8";
vllm serve $MODEL_NAME \
    $EXTRA_FLAGS \
    --port 8000 \
    --served-model-name kimi-k2 \
    --trust-remote-code \
    --data-parallel-size 16 \
    --data-parallel-size-local 8 \
    --data-parallel-address $MASTER_IP \
    --data-parallel-rpc-port $PORT \
    --enable-expert-parallel \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 256 \
    --gpu-memory-utilization 0.85 \
    --enable-auto-tool-choice \
    --tool-call-parser kimi_k2'
