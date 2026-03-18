#!/bin/bash

lightning run model \
    --node-rank=$SLURM_NODEID  \
    --main-address=$MASTER_ADDR \
    --accelerator=cuda \
    --devices=8 \
    --num-nodes=2 \
    /workspace/pretrain/train_prime_rl.py \
    --nodes_num 2 \
    --gpu_num 8 \
    --eval_freq 5000 \
    --model 1028 \
    --flops 3300. \
    --ssl_ratio 0.01 \
    --result_path /workspace/workdir \
    --wandb_project mdm_prime_v2_1028M \
    --data_path /workspace/download/slim_star_combined