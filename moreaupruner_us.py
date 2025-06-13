
import gc
import random
import argparse

import torch
import numpy as np
from transformers import LlamaTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
from LLMPruner.models.hf_llama.modeling_llama import LlamaForCausalLM

from LLMPruner.utils.logger import LoggerWithDepth
from LLMPruner.evaluator.ppl import PPLMetric
from LLMPruner.datasets.example_samples import get_examples
from LLMPruner.templates.prompts import prompts

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def soft_threshold(approxi, threshold):
    larger = approxi > threshold
    smaller = approxi < -1 * threshold
    mask = torch.logical_or(larger, smaller)
    approxi = approxi * mask
    subtracted = larger * -1 * threshold
    added = smaller * threshold
    approxi = approxi + subtracted + added

    return approxi

def soft_element_threshold(approxi, ref, threshold):
    threshold = torch.abs(ref) * threshold
    larger = approxi > threshold
    smaller = approxi < -1 * threshold
    mask = torch.logical_or(larger, smaller)
    approxi = approxi * mask
    subtracted = larger * -1 * threshold
    added = smaller * threshold
    approxi = approxi + subtracted + added

    return approxi
    
def main(args):
    set_random_seed(args.seed)

    logger = LoggerWithDepth(
        env_name="{}".format(args.save_ckpt_log_name), 
        config=args.__dict__,
        root_dir='prune_log',
        setup_sublogger=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model,trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True,
    )
    if args.device != "cpu":
        model.half()
    model.to(args.device)

    if args.test_before_train:
        logger.log("\n==================Generation Results before Pruning================\n")
        model.eval()
        with torch.no_grad():
            for prompt in prompts:
                input_ids = tokenizer(prompt, return_tensors="pt")['input_ids'].to(args.device)

                generation_output = model.generate(
                    input_ids=input_ids,
                    do_sample=True,
                    top_k=50,
                    max_length=args.max_seq_len,
                    top_p=args.top_p,
                    temperature=args.temperature,
                )
                
                result = tokenizer.decode(generation_output[0])
                logger.log(result)
    
        ppl = PPLMetric(model, tokenizer, ['wikitext2', 'ptb', 'alpaca'], args.max_seq_len, device=args.device)
        logger.log("PPL before pruning: {}".format(ppl))

    pruner_type = args.pruner_type.lower()
    assert pruner_type in ['random', 'l2', 'l1', 'taylor']

    for param in model.parameters():
        param.requires_grad_(True)
    before_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    forward_prompts = torch.tensor([
        [    1,   306,  4658,   278,  6593,   310,  2834,   338],
        [    1,  3439, 17632,  1925, 29892,   278,  6368,   310],
    ]).to(args.device) # Only for building the dependency graph. Any input will be fine since the computation result are not taken into consideration.

    logger.log("Use {} pruner...".format(pruner_type))

    cnt = 0
    if not args.moredata:
        example_prompts = get_examples(args.calib_dataset, tokenizer, args.num_examples, seq_len = 2048).to(args.device)
    for i in range(args.iterative_steps):
        gc.collect()
        if args.moredata:
            example_prompts = get_examples(args.calib_dataset, tokenizer, args.num_examples, seq_len = 2048).to(args.device)
        if pruner_type in ['taylor']:
            logger.log("Start Backwarding in iterative steps = {}...".format(i))
            model.zero_grad()

            with torch.no_grad():
                for p in model.parameters():
                    if i == 0:
                        p.delta = torch.zeros_like(p)
                    sigma = torch.abs(args.std * p.data)
                    p.noise = torch.randn(p.shape).to(args.device) * sigma
                    del sigma
                    p.data = p.data + p.noise + p.delta

            loss = model(example_prompts, labels=example_prompts).loss

            l2norm = 0

            for p in model.parameters():
                l2norm = l2norm + 0.5 * (torch.norm(p.delta, p=2) ** 2) / args.lamb
            logger.log("Loss = {}, l2-norm = {}".format(loss, l2norm))
            loss = loss + l2norm
            loss.backward()
            with torch.no_grad():
                for pm in model.parameters():
                    pm.data = pm.data - pm.delta - pm.noise
                    del pm.noise
                    pm.delta.data = (pm.delta- args.lr * pm.grad).data

                    tmp = pm.delta ** 2
                    tmp = tmp if len(tmp.shape) == 1 else tmp.mean(dim=tuple(list(range(len(tmp.shape)))[1:]))
                    grouped_map = tmp  ** .5
                    map_above_threshold = grouped_map > (args.soft * args.lr)
                    grouped_map *= map_above_threshold
                    is_zero = grouped_map == 0
                    grouped_map += is_zero
                    coeff = (1 - args.soft / grouped_map)
                    coeff *= map_above_threshold
                    if len(pm.delta.shape) == 1:
                        pm.delta = pm.delta * coeff.reshape([-1])
                    else:
                        pm.delta = pm.delta * coeff.reshape([-1]+[1]*(len(pm.delta.shape)-1))

        with torch.no_grad():
            if args.iterative_steps - i <= 5:
                cnt += 1
                for pm in model.parameters():
                    if cnt != 1:
                        pm.acc_grad = pm.acc_grad + pm.delta
                    else:
                        pm.acc_grad = pm.delta.clone().detach()
    model.zero_grad()
    for p in model.parameters():
        p.grad = p.acc_grad.clone().detach() / cnt
        del p.delta

    # Unstructured pruning
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and isinstance(module.weight, torch.Tensor) and module.weight.grad is not None:
            if module.weight.dim() <= 1:
                continue
            W_metric = torch.abs(module.weight.grad * module.weight.data)
            W_mask = torch.zeros_like(W_metric, dtype=torch.bool)
            if args.prune_n != 0:
                for ii in range(W_metric.shape[1]):
                    if ii % args.prune_m == 0:
                        tmp = W_metric[:, ii:(ii+args.prune_m)].float()
                        idx = torch.topk(tmp, args.prune_n, dim=1, largest=False)[1]
                        W_mask.scatter_(1, ii + idx, True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)
                k = int(W_metric.shape[1]*args.pruning_ratio)
                if k > 0:
                    indices = sort_res[1][:, :k]
                    W_mask.scatter_(1, indices, True)

            if args.keep_low_ratio > 0 and W_mask.sum() > 0:
                flat = W_mask.view(-1)
                prune_idx = torch.nonzero(flat, as_tuple=False).squeeze(1)
                num_keep = int(len(prune_idx)*args.keep_low_ratio)
                if num_keep > 0:
                    keep_idx = prune_idx[torch.randperm(len(prune_idx))[:num_keep]]
                    flat[keep_idx] = False
                W_mask = flat.view_as(W_mask)

            module.weight.data[W_mask] = 0

    after_pruning_parameters = sum((p.data!=0).sum().item() for p in model.parameters() if p.requires_grad)

    logger.log("After unstructured pruning, #nonzero parameters: {}".format(after_pruning_parameters))

    # Clean the gradient in the model
    model.zero_grad()
    for name, module in model.named_parameters():
        if 'weight' in name:
            module.grad = None

    after_pruning_parameters = after_pruning_parameters

    logger.log("#Param before: {}, #Param after: {}, Ratio = {:.4f}%".format(before_pruning_parameters, after_pruning_parameters,  100.0*after_pruning_parameters/before_pruning_parameters))

    gc.collect()
    torch.cuda.empty_cache()

    if args.save_model:
        model.half()
        torch.save({
            'model': model, 
            'tokenizer': tokenizer,
        }, logger.best_checkpoint_path)
    
    if args.eval_device != "cpu":
        model.half()
    model.to(args.eval_device)

    model.config.pad_token_id = tokenizer.pad_token_id = 0 
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if args.test_after_train:
        logger.log("\n==================Generation Results After Pruning================\n")
        
        model.eval()
        with torch.no_grad():
            for prompt in prompts:
                input_ids = tokenizer(prompt, return_tensors="pt")['input_ids'].to(args.eval_device)

                generation_output = model.generate(
                    input_ids=input_ids,
                    do_sample=True,
                    top_k=50,
                    max_length=args.max_seq_len,
                    top_p=args.top_p,
                    temperature=args.temperature,
                )
                
                result = tokenizer.decode(generation_output[0])
                logger.log(result)
        
        logger.log("\n==================Finish================\n")
    
    ppl = PPLMetric(model, tokenizer, ['wikitext2', 'ptb', 'alpaca'], args.max_seq_len, device=args.eval_device)
    logger.log("PPL after pruning: {}".format(ppl))
    logger.log("Memory Requirement: {} MiB\n".format(torch.cuda.memory_allocated()/1024/1024))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pruning LLaMA (huggingface version)')

    # argument for parsing
    parser.add_argument('--base_model', type=str, default="decapoda-research/llama-7b-hf", help='base model name')
    parser.add_argument('--save_ckpt_log_name', type=str, default="llama_prune", help='the path for save the checkpoint and the log. The final path would be log/{your_name_here}_{pruner_type}_{pruning_ratio}')
    parser.add_argument('--pruning_ratio', type=float, default=0.5, help='ratio of weights to prune per layer')
    parser.add_argument('--pruner_type', type=str, default='l2', help='pruner type')
    parser.add_argument('--prune_n', type=int, default=0, help='n of n:m semi-structured pruning; 0 for unstructured')
    parser.add_argument('--prune_m', type=int, default=0, help='m of n:m semi-structured pruning')
    parser.add_argument('--keep_low_ratio', type=float, default=0.01, help='ratio of least important weights to keep')

    # argument for generation
    parser.add_argument('--temperature', type=float, default=1.0, help='temperature')
    parser.add_argument('--top_p', type=float, default=0.95, help='top p')
    parser.add_argument('--max_seq_len', type=int, default=1024, help='max sequence length')

    # argument for layer-wise pruning/column-wise pruning
    parser.add_argument('--channel_wise', action='store_true', help='channel wise')
    parser.add_argument('--block_wise', action='store_true', help='block wise')
    parser.add_argument('--layer_wise', action='store_true', help='layer wise')
    parser.add_argument('--layer', type=int, default=12, help='remain the previous n layers')

    parser.add_argument('--block_attention_layer_start', type=int, help='start layer of block attention layers', default=3)
    parser.add_argument('--block_attention_layer_end', type=int, help='end layer of block attention layers', default=31)
    parser.add_argument('--block_mlp_layer_start', type=int, help='start layer of block mlp layers', default=3)
    parser.add_argument('--block_mlp_layer_end', type=int, help='end layer of block mlp layers', default=31)

    parser.add_argument('--iterative_steps', type=int, default=1, help="Iteration step for pruning. Default=1")
    parser.add_argument('--grouping_strategy', type=str, default='sum', help='Reduce method for grouping')
    parser.add_argument('--global_pruning', action='store_true', help='whether global pruning')
    parser.add_argument('--taylor', type=str, default='param_first', help='choose from [vectorize, param_second, param_first, param_mix]')
    parser.add_argument('--num_examples', type=int, default=10)

    # general argument
    parser.add_argument('--device', type=str, default="cuda", help='device')
    parser.add_argument('--test_before_train', action='store_true', help='whether test before train')
    parser.add_argument('--eval_device', type=str, default="cuda", help='eval device')
    parser.add_argument('--test_after_train', action='store_true', help='whether test after train')

    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--save_model', action='store_true', help='if save model')

    # smooth argument
    parser.add_argument('--std', type=float, default=0.05, help='std of noise')
    parser.add_argument('--lamb', type=float, default=1.00, help='lambda of moreau envelope')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--soft', type=float, default=5e-3, help='soft threshold')
    parser.add_argument('--moredata', action='store_true', help='if usemoredata')
    parser.add_argument('--calib_dataset', type=str, default='bookcorpus',
                        choices=['bookcorpus', 'c4', 'wikipedia', 'slimpajama', 'dclm'],
                        help='dataset used for calibration')

    args = parser.parse_args()

    torch_version = float('.'.join(torch.__version__.split('.')[:2]))
    args.torch_version = torch_version
    main(args)
