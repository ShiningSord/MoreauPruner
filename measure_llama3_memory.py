import argparse
import json
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def instrument_model(model, records):
    for layer_idx, layer in enumerate(model.model.layers):
        for block_name in ["input_layernorm", "self_attn", "post_attention_layernorm", "mlp"]:
            block = getattr(layer, block_name)
            original_forward = block.forward

            def make_wrapped_forward(forward_fn, idx, name):
                def wrapped_forward(*args, **kwargs):
                    torch.cuda.reset_peak_memory_stats()
                    start_time = time.time()
                    output = forward_fn(*args, **kwargs)
                    torch.cuda.synchronize()
                    end_time = time.time()
                    peak_mem = torch.cuda.max_memory_allocated()
                    records.append(
                        {
                            "layer": idx,
                            "block": name,
                            "peak_memory_mb": peak_mem / 1024 / 1024,
                            "latency_s": end_time - start_time,
                        }
                    )
                    return output

                return wrapped_forward

            block.forward = make_wrapped_forward(original_forward, layer_idx, block_name)

        # instrument q, k, v projections in attention
        for proj_name in ["q_proj", "k_proj", "v_proj"]:
            proj = getattr(layer.self_attn, proj_name)
            original_forward = proj.forward

            def make_wrapped_proj(forward_fn, idx, name):
                def wrapped_forward(*args, **kwargs):
                    torch.cuda.reset_peak_memory_stats()
                    start_time = time.time()
                    out = forward_fn(*args, **kwargs)
                    torch.cuda.synchronize()
                    end_time = time.time()
                    peak_mem = torch.cuda.max_memory_allocated()
                    records.append(
                        {
                            "layer": idx,
                            "block": name,
                            "peak_memory_mb": peak_mem / 1024 / 1024,
                            "latency_s": end_time - start_time,
                        }
                    )
                    return out

                return wrapped_forward

            proj.forward = make_wrapped_proj(original_forward, layer_idx, f"self_attn.{proj_name}")

        # instrument up and down projections in MLP
        for proj_name in ["up_proj", "down_proj"]:
            proj = getattr(layer.mlp, proj_name)
            original_forward = proj.forward

            def make_wrapped_mlp_proj(forward_fn, idx, name):
                def wrapped_forward(*args, **kwargs):
                    torch.cuda.reset_peak_memory_stats()
                    start_time = time.time()
                    out = forward_fn(*args, **kwargs)
                    torch.cuda.synchronize()
                    end_time = time.time()
                    peak_mem = torch.cuda.max_memory_allocated()
                    records.append(
                        {
                            "layer": idx,
                            "block": name,
                            "peak_memory_mb": peak_mem / 1024 / 1024,
                            "latency_s": end_time - start_time,
                        }
                    )
                    return out

                return wrapped_forward

            proj.forward = make_wrapped_mlp_proj(original_forward, layer_idx, f"mlp.{proj_name}")


def main():
    parser = argparse.ArgumentParser(description="Measure Llama3 inference usage")
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Meta-Llama-3-8B",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Hello, how are you?",
        help="Prompt text",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=16,
        help="Maximum generation tokens",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="layer_usage.json",
        help="Path to save the measurement results",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, low_cpu_mem_usage=True)
    model.to(device)
    model.eval()

    records = []
    instrument_model(model, records)

    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=args.max_tokens)

    with open(args.output, "w") as f:
        json.dump(records, f, indent=2)

    for r in records:
        print(
            f"Layer {r['layer']:02d} | {r['block']:24s} | "
            f"Peak Memory: {r['peak_memory_mb']:.2f} MB | Latency: {r['latency_s']:.4f}s"
        )


if __name__ == "__main__":
    main()
