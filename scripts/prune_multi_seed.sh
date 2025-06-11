#!/bin/bash

# Repeatedly prune and evaluate using ten different random seeds.
# Usage: bash scripts/prune_multi_seed.sh [calib_dataset]
# Default calibration dataset is 'bookcorpus'.

calib_dataset=${1:-bookcorpus}
base_model=baffo32/decapoda-research-llama-7B-hf

for seed in {1..10}; do
    run_name="${calib_dataset}_seed${seed}"
    echo "[RUN] pruning with seed ${seed} using ${calib_dataset}"

    python moreauprune.py \
        --pruning_ratio 0.25 \
        --device cpu \
        --eval_device cuda \
        --block_wise \
        --block_mlp_layer_start 4 \
        --block_mlp_layer_end 30 \
        --block_attention_layer_start 4 \
        --block_attention_layer_end 30 \
        --num_examples 10 \
        --save_ckpt_log_name ${run_name} \
        --pruner_type taylor \
        --taylor param_first \
        --save_model \
        --base_model ${base_model} \
        --test_after_train \
        --iterative_steps 10 \
        --std 0.05 \
        --lamb 0.2 \
        --lr 0.0002 \
        --soft 0.000005 \
        --seed ${seed} \
        --calib_dataset ${calib_dataset}

    CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_pretrain.sh ${base_model} prune_log/${run_name}
done
