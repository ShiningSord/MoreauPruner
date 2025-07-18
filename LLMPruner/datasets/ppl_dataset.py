'''
Some of the code refer to
https://github.com/IST-DASLab/gptq/blob/main/datautils.py
'''

import random
import numpy as np
import torch

from datasets import load_dataset
from torch.utils.data.dataset import Dataset

def get_wikitext2(seq_len, tokenizer):
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    return traindata, testdata

def get_ptb(seq_len, tokenizer):
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    valdata = load_dataset('ptb_text_only', 'penn_treebank', split='validation')
    return traindata, valdata

def get_alpaca(seq_len, tokenizer):
    """Load the Alpaca dataset used for fine-tuning."""
    traindata = load_dataset('yahma/alpaca-cleaned', split='train').select(range(10000))

    # The official dataset has no predefined validation split.  For perplexity
    # evaluation we simply reuse the training data.
    return traindata, traindata

class IndexDataset(Dataset):
    def __init__(self, tensors):
        self.tensors = tensors

    def __getitem__(self, index):
        return self.tensors[index]

    def __len__(self):
        return len(self.tensors)

def process_data(samples, tokenizer, seq_len, field_name):
    test_ids = tokenizer("\n\n".join(samples[field_name]), return_tensors='pt').input_ids[0]
    test_ids_batch = []
    nsamples = test_ids.numel() // seq_len

    for i in range(nsamples):
        batch = test_ids[(i * seq_len):((i + 1) * seq_len)]
        test_ids_batch.append(batch)
    test_ids_batch = torch.stack(test_ids_batch)
    return IndexDataset(tensors=test_ids_batch)
       

def get_loaders(name, tokenizer, seq_len=2048, batch_size = 8):
    if 'wikitext2' in name:
        train_data, test_data = get_wikitext2(seq_len, tokenizer)
        test_dataset = process_data(test_data, tokenizer, seq_len, 'text')
    if 'ptb' in name:
        train_data, test_data = get_ptb(seq_len, tokenizer)
        test_dataset = process_data(test_data, tokenizer, seq_len, 'sentence')
    if 'alpaca' in name:
        train_data, test_data = get_alpaca(seq_len, tokenizer)
        test_dataset = process_data(test_data, tokenizer, seq_len, 'output')

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_data, test_loader