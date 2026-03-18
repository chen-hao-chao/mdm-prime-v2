import torch
import argparse
import re

from lit_gpt.model_cache import GPTCache, Config
from lit_gpt.diffmodel import TransEncoder
from transformers import AutoTokenizer
from tqdm import tqdm
from datasets import load_dataset

from eval.gen_model_answer import  ar_sample_kvcache, diff_sample
from evaluate_diff import set_seed
from nltk.translate.bleu_score import sentence_bleu
from pathlib import Path
from safetensors.torch import load_file


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--qs_type",
        type=str,
        required=True,
        help="ntd or dtn"
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
        "--steps",
        default=32,
        type=int,
    )
    parser.add_argument(
        "--length",
        type=int,
        default=52
    )
    parser.add_argument(
        "--cfg",
        default=0.8,
        type=float,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234
    )
    args = parser.parse_args()
    return args


def get_diff_sample(args, data, model, tokenizer):
    results = []
    for i in tqdm(range(len(data['train']))):
        input_ids = tokenizer(data['train'][i]['prompt'], return_tensors="pt")['input_ids'].to('cuda')
        output_ids = diff_sample(model,
                                 tokenizer,
                                 input_ids,
                                 alg='greddy',
                                 steps=args.steps,
                                 temperature=0.,
                                 cfg_scale=args.cfg,
                                 context_length=args.length,
                                 device='cuda')
        output = tokenizer.decode(output_ids[0, input_ids.shape[-1]:], skip_special_tokens=True)
        results.append(dict(generation=output, reference=data['train'][i]['completion']))
    return results


if __name__ == "__main__":
    args = get_args()
    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_name = f"Diff_LLaMA_{args.model}M"
    config = Config.from_name(model_name)
    tokenizer = AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T',
                                              padding_side="right", use_fast=True)


    model = TransEncoder(config).to(device)
    model.load_state_dict(load_file(args.ckpt_path))

    if args.qs_type == 'dtn':
        data = load_dataset('json', data_files='data/reverse_experiments/june_version_7921032488/d2p_prompts_test.jsonl')
        data_reverse = load_dataset('json', data_files='data/reverse_experiments/june_version_7921032488/p2d_reverse_prompts_test.jsonl')
    elif args.qs_type == 'ntd':
        data = load_dataset('json', data_files='data/reverse_experiments/june_version_7921032488/p2d_prompts_test.jsonl')
        data_reverse = load_dataset('json', data_files='data/reverse_experiments/june_version_7921032488/d2p_reverse_prompts_test.jsonl')
    else:
        raise NotImplementedError(args.qs_type)

    result = get_diff_sample(args, data, model, tokenizer)
    result_reverse = get_diff_sample(args, data_reverse, model, tokenizer)
    assert len(result) == len(result_reverse)

    accs, accs_reverse = 0, 0
    blues, blues_reverse = 0, 0
    for i in range(len(result)):
        generation, reference = result[i]['generation'].strip().lower(), result[i]['reference'].strip().lower()
        generation_reverse, reference_reverse = result_reverse[i]['generation'].strip().lower(), result_reverse[i]['reference'].strip().lower()
        accs = accs + 1 if reference in generation else accs
        accs_reverse = accs_reverse + 1 if reference_reverse in generation_reverse else accs_reverse
        if args.qs_type == 'ntd':
            blues += sentence_bleu([reference], generation)
            blues_reverse += sentence_bleu([reference_reverse], generation_reverse)
    accs, accs_reverse = accs / len(result), accs_reverse / len(result_reverse)
    blues, blues_reverse = blues / len(result), blues_reverse / len(result_reverse)

    if args.qs_type == 'ntd':
        message = f'qs_type: {args.qs_type}, accs: {accs}, accs_reverse: {accs_reverse}, blue: {blues}, blues_reverse: {blues_reverse}'
    else:
        message = f'qs_type: {args.qs_type}, accs: {accs}, accs_reverse: {accs_reverse}'
    print(message)








