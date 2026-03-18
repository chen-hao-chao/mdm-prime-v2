import torch
import math
import argparse
import pickle
from pathlib import Path
import re

from lit_gpt.model import GPT, Config
from lit_gpt.diffmodel import TransEncoder
from transformers import AutoTokenizer

import torch.nn.functional as F
from tqdm import tqdm
from evaluate_diff import set_seed
from safetensors.torch import load_file


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--type",
        type=str,
        required=True,
        help="arm or mdm"
    )
    parser.add_argument(
        "--model",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        required=True
    )
    parser.add_argument(
        "--fineweb",
        type=str,
        required=True
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128
    )
    parser.add_argument(
        "--mc-samples",
        type=int,
        default=1024
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234
    )
    args = parser.parse_args()
    return args


def forward_process(batch, total_dim=32000, eps=1e-3):
    b, l = batch.shape
    t = torch.rand((b,), device=batch.device)

    p_mask = (1 - eps) * t + eps
    p_mask = p_mask[:, None].repeat(1, l)

    mask_indices = torch.rand((b, l), device=batch.device) < p_mask
    noisy_batch = torch.where(mask_indices, total_dim, batch)
    return noisy_batch, mask_indices, p_mask


@torch.no_grad()
def get_loss_diff(model, input_ids):
    noisy_input, mask_indices, p_mask = forward_process(input_ids)
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        logits = model(noisy_input)
    loss = F.cross_entropy(logits[mask_indices], input_ids[mask_indices], reduction='none') / p_mask[mask_indices]
    loss = loss.sum() / (input_ids.shape[0] * input_ids.shape[1])
    return loss


@torch.no_grad()
def get_loss_ar(model, input_ids):
    target = input_ids[:, 1:].contiguous()
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        logits = model(input_ids[:, :-1])
    loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), target.view(-1))
    return loss


if __name__ == "__main__":
    args = get_args()
    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_name = f"Diff_LLaMA_{args.model}M"
    config = Config.from_name(model_name)
    tokenizer = AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T',
                                              padding_side="right", use_fast=True)

    if args.type == 'arm':
        model = GPT(config).to(device)
    elif args.type == 'mdm':
        model = TransEncoder(config).to(device)
    else:
        raise NotImplementedError(args.type)


    model.load_state_dict(load_file(args.ckpt_path))

    pickle_filename = f'data/fineweb/{args.fineweb}.pkl'
    print(f'load from: {pickle_filename}')
    with open(pickle_filename, 'rb') as f:
        loaded_tokens = pickle.load(f)

    assert len(loaded_tokens) == 100 * 1024 * 2048
    assert len(loaded_tokens) % (args.batch_size * 2048) == 0
    num_iterations = len(loaded_tokens) // (args.batch_size * 2048)

    losses = []
    for index in tqdm(range(num_iterations)):
        data_list = loaded_tokens[index * (args.batch_size * 2048): (index + 1) * (args.batch_size * 2048)]
        data = torch.tensor(data_list).to(device)
        data = data.view(args.batch_size, 2048)

        if args.type == 'arm':
            loss = get_loss_ar(model, data)
            losses.append(loss.item())
        elif args.type == 'mdm':
            for _ in range(args.mc_samples): # mc number
                loss = get_loss_diff(model, data)
                losses.append(loss.item())
        else:
            raise NotImplementedError(args.type)
    ppl = math.exp(sum(losses) / len(losses))

    message = f'{args.ckpt_path}, {args.fineweb}, ppl={ppl}'
    print(message)





