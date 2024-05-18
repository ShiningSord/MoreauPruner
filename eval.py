
import argparse

import torch

from transformers import  AutoModelForCausalLM, AutoTokenizer
from LLMPruner.evaluator.ppl import PPLMetric
from LLMPruner.peft import PeftModel

#from utils.callbacks import Iteratorize, Stream
#from utils.prompter import Prompter

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
torch_version = int(torch.__version__.split('.')[1])

def main(args):
    if args.model_type == 'pretrain':
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            low_cpu_mem_usage=True if torch_version >=9 else False
        )
        description = "Model Type: {}\n Base Model: {}".format(args.model_type, args.base_model)
    elif args.model_type == 'pruneLLM':
        pruned_dict = torch.load(args.ckpt, map_location='cpu')
        tokenizer, model = pruned_dict['tokenizer'], pruned_dict['model']
        description = "Model Type: {}\n Pruned Model: {}".format(args.model_type, args.ckpt)
    elif args.model_type == 'tune_prune_LLM':
        pruned_dict = torch.load(args.ckpt, map_location='cpu')
        tokenizer, model = pruned_dict['tokenizer'], pruned_dict['model']

   
        model = PeftModel.from_pretrained(
            model,
            args.lora_ckpt,
            torch_dtype=torch.float16,
        )
        description = "Model Type: {}\n Pruned Model: {}\n LORA ckpt: {}".format(args.model_type, args.ckpt, args.lora_ckpt)
    else:
        raise NotImplementedError

    if device == "cuda":
        model.half()
        model = model.cuda()
    
    # unwind broken decapoda-research config
    # import pdb; pdb.set_trace()
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
    tokenizer.bos_token_id = 0
    tokenizer.eos_token_id = 0
    tokenizer.unk_token_id = 0
    model.eval()

    ppl = PPLMetric(model, tokenizer, ['wikitext2', 'ptb'], 128, device="cuda")
    print("PPL after pruning: {}".format(ppl))
    print("Memory Requirement: {} MiB\n".format(torch.cuda.memory_allocated()/1024/1024))
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tuning Pruned LLaMA (huggingface version)')

    parser.add_argument('--base_model', type=str, default="decapoda-research/llama-7b-hf", help='base model name')
    parser.add_argument('--model_type', type=str, required=True, help = 'choose from ')
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--lora_ckpt', type=str, default=None)
    parser.add_argument('--share_gradio', action='store_true')

    args = parser.parse_args()
    main(args)


