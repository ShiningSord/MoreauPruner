#!/bin/bash

#SBATCH --job-name=wikipedia_us
#SBATCH --mail-user=zixiaowang97@qq.com
#SBATCH --output=logs/wikipedia_us.log
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --constraint=3090
#SBATCH --exclude=proj[77,192,194,203,199,198,202,197]

# base_model=baffo32/decapoda-research-llama-7B-hf
base_model=apple/DCLM-7B
# calib_dataset=bookcorpus
# calib_dataset=c4
calib_dataset=wikipedia
# calib_dataset=slimpajama
# calib_dataset=dclm

for seed in {1..1}; do
    run_name="${calib_dataset}_us_seed${seed}"
    echo "[RUN] unstructured pruning with seed ${seed} using ${calib_dataset}"

    python moreaupruner_us.py \
        --pruning_ratio 0.2 \
        --device cpu \
        --eval_device cuda \
        --block_wise \
        --block_mlp_layer_start 0 \
        --block_mlp_layer_end 33 \
        --block_attention_layer_start 0 \
        --block_attention_layer_end 33 \
        --num_examples 4 \
        --save_ckpt_log_name ${run_name} \
        --moredata \
        --max_seq_len 1024 \
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

    # CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_pretrain.sh ${base_model} prune_log/${run_name}
done
