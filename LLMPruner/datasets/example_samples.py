import random
import torch

from datasets import load_dataset


def _collect_random_text(dataset, field_name, n_samples, seq_len, char_multiplier=50):
    """Return a single long string sampled from ``dataset``.

    The function picks random documents from ``dataset`` until the combined
    character length exceeds ``n_samples * seq_len * char_multiplier``.  This
    long text is then tokenized once to avoid repeatedly tokenizing many short
    documents.
    """

    target_chars = n_samples * seq_len * char_multiplier
    selected = []
    char_count = 0
    while char_count < target_chars:
        idx = random.randint(0, len(dataset) - 1)
        text = dataset[idx][field_name]
        selected.append(text)
        char_count += len(text)
    return " ".join(selected)


def _sample_sequences(text, tokenizer, n_samples, seq_len):
    """Tokenize ``text`` once and draw ``n_samples`` random subsequences."""

    token_ids = tokenizer(text, return_tensors="pt").input_ids[0]
    if token_ids.shape[0] <= seq_len:
        raise ValueError("Sampled text is shorter than the requested sequence length")

    samples = []
    for _ in range(n_samples):
        start = random.randint(0, token_ids.shape[0] - seq_len - 1)
        samples.append(token_ids[start : start + seq_len])
    return torch.stack(samples, dim=0)

def get_c4(tokenizer, n_samples, seq_len):
    traindata = load_dataset(
        "allenai/c4",
        "allenai--c4",
        data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
        split="train",
    )

    long_text = _collect_random_text(traindata, "text", n_samples, seq_len)
    return _sample_sequences(long_text, tokenizer, n_samples, seq_len)

def get_bookcorpus(tokenizer, n_samples, seq_len):
    traindata = load_dataset("bookcorpus", split="train")

    long_text = _collect_random_text(traindata, "text", n_samples, seq_len)
    return _sample_sequences(long_text, tokenizer, n_samples, seq_len)

def get_wikipedia(tokenizer, n_samples, seq_len):
    """Load the first shard of the cleaned English Wikipedia from 2023-11-01."""
    traindata = load_dataset(
        'wikimedia/wikipedia',
        data_files={'train': '20231101.en/train-00000-of-00041.parquet'},
        split='train'
    )

    long_text = _collect_random_text(traindata, "text", n_samples, seq_len)
    return _sample_sequences(long_text, tokenizer, n_samples, seq_len)

def get_slimpajama(tokenizer, n_samples, seq_len):
    """Load a shard from the SlimPajama dataset."""
    traindata = load_dataset(
        'DKYoon/SlimPajama-6B',
        data_files={'train': 'data/train-00000-of-00052.parquet'},
        split='train'
    )

    long_text = _collect_random_text(traindata, "text", n_samples, seq_len)
    return _sample_sequences(long_text, tokenizer, n_samples, seq_len)

def get_dclm(tokenizer, n_samples, seq_len):
    """Load a subset of the DCLM dataset used for DCLM-7B pre-training."""
    traindata = load_dataset(
        'coai/dclm-baseline-subset_100k',
        data_files={'train': 'data/train-00000-of-00002.parquet'},
        split='train'
    )

    long_text = _collect_random_text(traindata, "text", n_samples, seq_len)
    return _sample_sequences(long_text, tokenizer, n_samples, seq_len)

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
