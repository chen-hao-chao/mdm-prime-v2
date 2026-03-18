import torch

from datasets import load_dataset
from torch.utils.data import Dataset
from tqdm import tqdm


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def preprocess_reverse_train(tokenizer):
    data = load_dataset('json', data_files='data/reverse_experiments/june_version_7921032488/all_prompts_train.jsonl')

    train_dataset = []
    for i in tqdm(range(len(data['train']))):
        input = data['train'][i]['prompt'] + data['train'][i]['completion']
        input_ids = tokenizer(input, return_tensors="pt")['input_ids'][0]
        input_ids = torch.cat((input_ids, torch.tensor([tokenizer.eos_token_id])), dim=-1)

        length = input_ids.shape[-1]
        padding_length = 2048 - length
        padding = torch.full((padding_length,), tokenizer.eos_token_id, dtype=input_ids.dtype) # padding with |EOS|
        padded_data = torch.cat((input_ids, padding), dim=-1)

        train_dataset.append(dict(data=padded_data, length=length))
    train_dataset = CustomDataset(train_dataset)
    return train_dataset
