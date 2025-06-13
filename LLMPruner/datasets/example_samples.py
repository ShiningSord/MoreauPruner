import random
import torch

from datasets import load_dataset



def _concat_all_text(dataset, field_name):
    """Join every entry under ``field_name`` into a single large string."""

    return " ".join(dataset[field_name])


def _sample_raw_patches(text, n_samples, seq_len):
    """Pick ``n_samples`` random substrings from ``text`` before tokenization."""

    patch_len = seq_len * 8
    if patch_len >= len(text):
        patch_len = len(text) - 1
    max_start = max(0, len(text) - patch_len - 1)
    patches = []
    for _ in range(n_samples):
        start = random.randint(0, max_start)
        patches.append(text[start : start + patch_len])
    return patches


def _tokenize_patches(patches, tokenizer, seq_len):
    tokenized_samples = []
    for patch in patches:
        enc = tokenizer(
            patch,
            max_length=seq_len,
            truncation=True,
            return_tensors="pt",
        )

        tokenized_samples.append(enc.input_ids)
 
    return torch.cat(tokenized_samples, dim=0)
    

def get_c4(tokenizer, n_samples, seq_len):
    dataset = load_dataset("c4", "en", split="train", streaming=True,trust_remote_code=True)

    # Take just the first 10000 samples
    small_subset = {}
    small_subset['text'] = []
    for i, sample in enumerate(dataset):
        small_subset['text'].append(sample['text'])
        if i >= 10000:
            break
 

    text = _concat_all_text(small_subset, "text")
    patches = _sample_raw_patches(text, n_samples, seq_len)
    return _tokenize_patches(patches, tokenizer, seq_len)

def get_bookcorpus(tokenizer, n_samples, seq_len):
    traindata = load_dataset("bookcorpus", split="train", streaming=True,trust_remote_code=True)

    # Take just the first 10000 samples
    small_subset = {}
    small_subset['text'] = []
    for i, sample in enumerate(traindata):
        small_subset['text'].append(sample['text'])
        if i >= 10000:
            break
 

    text = _concat_all_text(small_subset, "text")
    
    
    patches = _sample_raw_patches(text, n_samples, seq_len)
    return _tokenize_patches(patches, tokenizer, seq_len)


def get_wikipedia(tokenizer, n_samples, seq_len):
    """Load the first shard of the cleaned English Wikipedia from 2023-11-01."""
    traindata = load_dataset(
        'wikimedia/wikipedia',
        data_files={'train': '20231101.en/train-00000-of-00041.parquet'},
        split='train',
        streaming=True,
        trust_remote_code=True
    )


    # Take just the first 10000 samples
    small_subset = {}
    small_subset['text'] = []
    for i, sample in enumerate(traindata):
        small_subset['text'].append(sample['text'])
        if i >= 10000:
            break
 

    text = _concat_all_text(small_subset, "text")
    patches = _sample_raw_patches(text, n_samples, seq_len)
    return _tokenize_patches(patches, tokenizer, seq_len)


def get_slimpajama(tokenizer, n_samples, seq_len):
    """Load a shard from the SlimPajama dataset."""
    traindata = load_dataset(
        'DKYoon/SlimPajama-6B',
        data_files={'train': 'data/train-00000-of-00048-ab2b35705f029d94.parquet'},
        split='train',
        streaming=True,
        trust_remote_code=True
    )


    # Take just the first 10000 samples
    small_subset = {}
    small_subset['text'] = []
    for i, sample in enumerate(traindata):
        small_subset['text'].append(sample['text'])
        if i >= 10000:
            break
 

    text = _concat_all_text(small_subset, "text")
    patches = _sample_raw_patches(text, n_samples, seq_len)
    return _tokenize_patches(patches, tokenizer, seq_len)


def get_dclm(tokenizer, n_samples, seq_len):
    """Load a subset of the DCLM dataset used for DCLM-7B pre-training."""
    traindata = load_dataset(
        'coai/dclm-baseline-subset_100k',
        data_files={'train': 'data/train-00000-of-00002.parquet'},
        split='train',
        streaming=True,
        trust_remote_code=True
    )


    # Take just the first 10000 samples
    small_subset = {}
    small_subset['text'] = []
    for i, sample in enumerate(traindata):
        small_subset['text'].append(sample['text'])
        if i >= 10000:
            break
 

    text = _concat_all_text(small_subset, "text")
    patches = _sample_raw_patches(text, n_samples, seq_len)
    return _tokenize_patches(patches, tokenizer, seq_len)


def get_examples(dataset, tokenizer, n_samples, seq_len = 128):
    if dataset == 'c4':
        return get_c4(tokenizer, n_samples, seq_len)
    elif dataset == 'bookcorpus':
        return get_bookcorpus(tokenizer, n_samples, seq_len)
    elif dataset == 'wikipedia':
        return get_wikipedia(tokenizer, n_samples, seq_len)
    elif dataset == 'slimpajama':
        return get_slimpajama(tokenizer, n_samples, seq_len)
    elif dataset == 'dclm':
        return get_dclm(tokenizer, n_samples, seq_len)
    else:
        raise NotImplementedError
