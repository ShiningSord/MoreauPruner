#!/bin/bash
export PYTHONPATH='.'

base_model=$1 # e.g., decapoda-research/llama-7b-hf
prune_ckpt=$2



python lm-evaluation-harness/main.py --model hf-causal-experimental --model_args checkpoint=$prune_ckpt/pytorch_model.bin,config_pretrained=$base_model --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq --device cuda:0 --output_path results/${prune_ckpt}.json --no_cache 

