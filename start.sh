#!/bin/bash


base_model=baffo32/decapoda-research-llama-7B-hf
# base_model=lmsys/vicuna-7b-v1.1
# base_model=jeffwan/llama-13b-hf
prune_ckpt_path=llama_7b_0.2_element_moreau_std005_lamb02_lr00002_groupsoft0000005_10iter
tune_ckpt_path=llama_7b_0.2_element_moreau_std005_lamb02_lr00002_groupsoft0000005_10iter_alpaca


echo "[START] - Start Pruning Model"
python moreauprune.py --pruning_ratio 0.25 --device cpu  --eval_device cuda --block_wise --block_mlp_layer_start 4 --num_examples 10   --block_mlp_layer_end 30 --block_attention_layer_start 4 --block_attention_layer_end 30 --save_ckpt_log_name $prune_ckpt_path --pruner_type taylor --taylor param_first --save_model  --base_model ${base_model} --test_after_train  --iterative_steps 10 --std 0.05 --lamb 0.2 --lr 0.0001 --soft 0.000005
echo "[FINISH] - Finish Pruning Model"

CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_pretrain.sh ${base_model} prune_log/${prune_ckpt_path} 

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

function rand(){
    min=$1
    max=$(($2 - $min + 1))
    num=$(($RANDOM+1000000000)) 
    echo $(($num%$max + $min))
}

port=$(rand 4000 6000)


echo "[START] - Start Tuning"
deepspeed --include=localhost:0,1 --master_port=$port  post_training.py --prune_model prune_log/$prune_ckpt_path/pytorch_model.bin --data_path yahma/alpaca-cleaned  --output_dir tune_log/$tune_ckpt_path --wandb_project llama_tune_ --lora_r 8 --num_epochs 2 --learning_rate 1e-4  --cache_dataset
echo "[FINISH] - Finish Prune and Post-Training."
echo "[INFO] - The pruned model is at {prune_log/$prune_ckpt_path/pytorch_model.bin}, and the recovery weight is at {tune_log/$prune_ckpt_path}/"


python eval.py --base_mode ${base_model}  --model_type tune_prune_LLM --ckpt prune_log/$prune_ckpt_path/pytorch_model.bin --lora_ckpt tune_log/$tune_ckpt_path

CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate.sh ${base_model} tune_log/${tune_ckpt_path} prune_log/${prune_ckpt_path} 1400