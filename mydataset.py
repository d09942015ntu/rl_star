import json

import numpy as np
import pandas as pd
from transformers import GPT2Tokenizer
from torch.utils.data import Dataset
import torch

MAX_LENGTH = 30


def make_input_data(input_text, target_text, tokenizer, max_length=None):
    s1 = input_text + '[=]'
    s2 = target_text + tokenizer.eos_token

    input_text_full = s1 + s2
    if max_length is None:
        max_length = len(tokenizer.encode(input_text_full)) + 1
    # print(f"input_text_full:{input_text_full}")
    # Encode sentences with the format [s1, s2]
    encoding = tokenizer(input_text_full, truncation=True, max_length=max_length, padding='max_length')
    # input_encoding = self.tokenizer.encode(input_text_full)
    s1_encoding = tokenizer.encode(s1)
    s2_encoding = tokenizer.encode(s2)

    # target_index_low = 0
    target_index = len(s2_encoding)

    # Create labels with the target sequence (s2) masked for autoregressive training
    labels = encoding["input_ids"].copy()
    labels[:] = [-100] * len(labels)  # Mask `s1` tokens
    labels[len(s1_encoding): len(s1_encoding) + target_index] = encoding["input_ids"][
                                                                len(s1_encoding): len(s1_encoding) + target_index]

    attention_mask = encoding['attention_mask'].copy()
    attention_mask[len(s1_encoding):] = [0] * len(attention_mask[len(s1_encoding):])

    position_ids = list(range(len(encoding['input_ids'])))

    output_item = {
        'input_ids': torch.tensor(encoding['input_ids']),
        'attention_mask': torch.tensor(attention_mask),
        'position_ids': torch.tensor(position_ids),
        'labels': torch.tensor(labels)
    }
    return output_item


def make_input_data_generate(input_text, tokenizer):
    s1 = input_text + '[=]'

    input_text_full = s1  # ???
    # Encode sentences with the format [s1, s2]
    s1_encoding = tokenizer.encode(s1)
    encoding = tokenizer(input_text_full, truncation=True, max_length=len(s1_encoding), padding='max_length')
    # input_encoding = self.tokenizer.encode(input_text_full)

    attention_mask = encoding['attention_mask'].copy()
    attention_mask[len(s1_encoding):] = [0] * len(attention_mask[len(s1_encoding):])

    position_ids = list(range(len(encoding['input_ids'])))

    output_item = {
        'input_ids': torch.tensor(encoding['input_ids']),
        'attention_mask': torch.tensor(attention_mask),
        'position_ids': torch.tensor(position_ids),
    }
    return output_item


class TrainDataset(Dataset):
    def __init__(self, file_path, tokenizer, n=None, start_step=0):
        self.data = pd.read_csv(file_path)
        self.tokenizer = tokenizer
        if n is not None:
            self.n = n
        else:
            self.n = len(self.data.keys()) - 1
        self.max_length = MAX_LENGTH
        self.start_step = start_step
        self.rng = np.random.RandomState(0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        # Randomly choose a starting point vghbv
        start_index = self.rng.randint(self.start_step, self.n)
        # start_index = 0
        # Match s_i as input and s_{i+1} as the target
        output_item = make_input_data(row[f's{start_index}'], row[f's{start_index + 1}'], self.tokenizer,
                                      self.max_length)

        return output_item


class EvalDataset(TrainDataset):
    def __init__(self, file_path, tokenizer, n=None, start_step=0):
        super().__init__(file_path, tokenizer, n, start_step)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        output = dict([(f's{i}', row[f's{i}']) for i in range(self.n + 1)])
        return output


if __name__ == '__main__':
    model_name = 'gpt2'
    dataset_name = 'data/zip2_3_gt.csv'
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({'pad_token': '[PAD]', 'eos_token': '[EOS]'})

    tokens = json.load(open(dataset_name.replace(".csv", "_token.json"), "r"))
    special_tokens_dict = {'additional_special_tokens': tokens}
    tokenizer.add_special_tokens(special_tokens_dict)

    dataset = TrainDataset(dataset_name, tokenizer, start_step=0)
    for item in dataset:
        for key, value in item.items():
            if key == 'input_ids':  # or key == 'labels':
                raw = tokenizer.decode(value)
            else:
                raw = "0"
            print(f"{key}: {raw} : {value}")  # Adjust processing logic as necessary
