import torch

from tqdm import tqdm
from fastchat.model.model_adapter import get_conversation_template
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def preprocess_sharegpt(data, tokenizer):
    train_dataset = []
    for i in tqdm(range(len(data))):
        d = data[i]
        if len(d["conversations"]) < 2:
            continue

        from0, prompt = d["conversations"][0]["from"], d["conversations"][0]["value"]
        from1, answer = d["conversations"][1]["from"], d["conversations"][1]["value"]

        if from0 != 'human' or from1 != 'gpt':
            continue

        conv = get_conversation_template("models/vicuna-7b-v1.5")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], "")
        prompt_with_template = conv.get_prompt()

        input = tokenizer(prompt_with_template, return_tensors="pt")['input_ids'][0]
        output = tokenizer(answer, return_tensors="pt")['input_ids'][0]
        output = torch.cat((output, torch.tensor([tokenizer.eos_token_id])), dim=-1)

        length = input.shape[-1] + output.shape[-1]
        if length > 2048:
            # exclude prompts that are too long
            continue

        padding_length = 2048 - length
        padding = torch.full((padding_length,), output[-1], dtype=output.dtype) # padding with |EOS|
        padded_data = torch.cat((input, output, padding), dim=-1)

        train_dataset.append(
            dict(data=padded_data, input_length=torch.tensor(input.shape[-1]), length=torch.tensor(length)))
    train_dataset = CustomDataset(train_dataset)
    return train_dataset
