### This project is forked from [LLM-Pruner](https://github.com/horseee/LLM-Pruner). We sincerely appreciate their incredible work and contribution to the community.

## Quick Start

### Create an environment for MoreauPruner
```
conda create --name moreau python=3.8
conda activate moreau
```

### Install requirements
```
pip install -r requirement.txt
```

### Multi-GPU acceleration
```
pip install deepspeed
```

### Prune-Finetune-Evaluate in one line!
```
bash start.sh > logs/llama_7b_0.2_element_moreau_std005_lamb02_lr00002_groupsoft0000005_10iter_alpaca.log
```
The script requires at least 1 RTX3090 GPU and about 200G memory (on CPU). But multi-gpu acceleration for finetuning is also support.

The script support the pruning of LLaMA-7B, LLaMA-13B and Vicuna-7B now. We will support more models (e.g., LLaMA3-8B) in future work.

Note: due to hardware limitation, we never try models larger than 13B before. But it can be easily extend to larger model. We are welcome the results on larger models from you.

In this script, we finish several things together
- prune LLaMA-7B by 20%
- evaluate the pruned model
- finetune the model on alpaca
- evaluate the finetuned model

A detailed explanation of the parameters can be found in LLM-Pruner's readme (below). And it takes about 8 hours in our device with two 3090s.

Once all the programs are finished, you can process the results by
```
python dataprocess.py logs/llama_7b_0.2_element_moreau_std005_lamb02_lr00002_groupsoft0000005_10iter_alpaca.log
```
And it will return the PPL on WikiText2 and PTB and the 0-shot accuracy on 7 datasets in order of ["boolq", "piqa", "hellaswag", "winogrande", "arc_easy", "arc_challenge", "openbookqa"].

We also attached a [log](logs/llama_7b_0.2_element_moreau_std005_lamb02_lr00002_groupsoft0000005_10iter_alpaca.log) for your reference.

