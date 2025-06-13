#!/bin/bash

#SBATCH --job-name=slimpajama
#SBATCH --mail-user=zixiaowang97@qq.com
#SBATCH --output=logs/slimpajama.log
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --constraint=3090
#SBATCH --exclude=proj[77,192,194,203,199]

base_model=baffo32/decapoda-research-llama-7B-hf
# calib_dataset=bookcorpus
# calib_dataset=c4
# calib_dataset=wikipedia
calib_dataset=slimpajama
# calib_dataset=dclm


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
        --num_examples 4 \
        --save_ckpt_log_name ${run_name} \
        --moredata \
        --max_seq_len 128 \
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

# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1

# function rand(){
#     min=$1
#     max=$(($2 - $min + 1))
#     num=$(($RANDOM+1000000000)) 
#     echo $(($num%$max + $min))
# }

# port=$(rand 4000 6000)


# echo "[START] - Start Tuning"
# deepspeed --include=localhost:0 --master_port=$port  post_training.py --prune_model prune_log/$prune_ckpt_path/pytorch_model.bin --data_path yahma/alpaca-cleaned  --output_dir tune_log/$tune_ckpt_path --wandb_project llama_tune_ --lora_r 8 --num_epochs 2 --learning_rate 1e-4  --cache_dataset --batch_size 64 --micro_batch_size 4 
# echo "[FINISH] - Finish Prune and Post-Training."
# echo "[INFO] - The pruned model is at {prune_log/$prune_ckpt_path/pytorch_model.bin}, and the recovery weight is at {tune_log/$prune_ckpt_path}/"


# python eval.py --base_mode ${base_model}  --model_type tune_prune_LLM --ckpt prune_log/$prune_ckpt_path/pytorch_model.bin --lora_ckpt tune_log/$tune_ckpt_path

# CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate.sh ${base_model} tune_log/${tune_ckpt_path} prune_log/${prune_ckpt_path} 1400