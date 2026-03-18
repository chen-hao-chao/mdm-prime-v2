import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def preprocess_gsm8k(tokenizer, max_length=2048):
    train_dataset = []

    data = []
    file_path = 'data/gsm8k/train.txt'
    with open(file_path, 'r') as f:
        for line in f:
            data.append(line)

    for i in range(len(data)):
        d = data[i]

        if len(d.split('||')) != 2:
            continue
        if len(d.split('||')[1].split('####')) != 2:
            continue

        question, thought, answer = d.split('||')[0], d.split('||')[1].split('####')[0], d.split('####')[1]
        question = 'Question: ' + question
        thought = 'Answer: ' + thought
        answer = '####' + answer

        question = tokenizer(question, return_tensors="pt")['input_ids'][0]
        thought = tokenizer(thought, return_tensors="pt")['input_ids'][0]
        answer = tokenizer(answer, return_tensors="pt")['input_ids'][0]
        answer = torch.cat((answer, torch.tensor([tokenizer.eos_token_id])), dim=-1)

        length1 = question.shape[-1] + thought.shape[-1]
        length2 = length1 + answer.shape[-1]
        if length2 > max_length:
            # exclude prompts that are too long
            continue

        padding_length = 2048 - length1
        padding = torch.full((padding_length,), tokenizer.eos_token_id, dtype=question.dtype)
        padded_data = torch.cat((question, thought, padding), dim=-1)
        train_dataset.append(dict(data=padded_data, input_length=torch.tensor(question.shape[-1]),
                                  length=torch.tensor(length1)))


        padding_length = 2048 - (question.shape[-1] + thought.shape[-1] + answer.shape[-1])
        padding = torch.full((padding_length,), tokenizer.eos_token_id, dtype=question.dtype)
        padded_data = torch.cat((question, thought, answer, padding), dim=-1)
        train_dataset.append(dict(data=padded_data, input_length=torch.tensor(length1),
                                  length=torch.tensor(length2)))


    train_dataset = CustomDataset(train_dataset)
    return train_dataset