import torch
import argparse
import re

from lit_gpt.model_cache import Config
from lit_gpt.diffmodel import TransEncoder
from transformers import AutoTokenizer
from datasets import load_dataset
from safetensors.torch import load_file

from eval.gen_model_answer import diff_sample
from evaluate_diff import set_seed
from eval.math_normalization import normalize_final_answer, check_sympy_equivalence


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--steps",
        default=256,
        type=int,
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
    )
    parser.add_argument(
        "--length",
        type=int,
        default=256
    )
    parser.add_argument(
        "--cfg1",
        default=0.1,
        type=float,
    )
    parser.add_argument(
        "--cfg2",
        default=0.1,
        type=float,
    )
    parser.add_argument(
        "--temperature",
        default=0.1,
        type=float,
    )
    args = parser.parse_args()
    return args


def get_diff_sample(args, question, model, tokenizer):
    question_ids = tokenizer(question, padding="longest", truncation=True, return_tensors="pt")['input_ids'].to('cuda')
    prefix_ids = diff_sample(model,
                             tokenizer,
                             question_ids,
                             alg='greddy',
                             steps=args.steps,
                             temperature=args.temperature,
                             cfg_scale=args.cfg1,
                             context_length=args.length,
                             device='cuda')
    prefix = tokenizer.batch_decode(prefix_ids, skip_special_tokens=True)

    prefix_ids = tokenizer(prefix, padding="longest", truncation=True, return_tensors="pt")['input_ids'].to('cuda')
    answer_ids = diff_sample(model,
                             tokenizer,
                             prefix_ids,
                             alg='greddy',
                             steps=args.steps,
                             temperature=args.temperature,
                             cfg_scale=args.cfg2,
                             context_length=args.length,
                             device='cuda')
    answer = tokenizer.batch_decode(answer_ids, skip_special_tokens=True)

    return answer


def get_acc(pred, right_answer):
    pattern = "#### (.*)$"

    preds = re.findall(pattern, pred)
    pred = preds[-1] if len(preds) >= 1 else ""

    pred = normalize_final_answer(pred)
    right_answer = normalize_final_answer(right_answer)
    return check_sympy_equivalence(pred, right_answer)


if __name__ == "__main__":
    args = get_args()
    set_seed(1234)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_name = f"Diff_LLaMA_1028M"
    config = Config.from_name(model_name)
    tokenizer = AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T',
                                              padding_side="right", use_fast=True)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token_id = 32000

    model = TransEncoder(config).to(device)
    model.load_state_dict(load_file(args.ckpt_path))

    acc = 0
    num = 0
    batch_size = 64

    dataset = load_dataset('json', data_files='data/gsm8k/test.jsonl')
    length = len(dataset['train'])
    iter = length // batch_size if length % batch_size == 0 else length // batch_size + 1

    for i in range(iter):
        end_index = (i + 1) * batch_size if (i + 1) * batch_size < length else length
        data = dataset['train'][i * batch_size: end_index]
        questions = ['Question: ' + q for q in data["question"]]
        right_answers = data["target"]

        preds = get_diff_sample(args, questions, model, tokenizer)

        for index in range(len(questions)):
            print(preds[index])
            print(f'Ground truth answers:\n', f'{right_answers[index]}\n')
            print(f'***************************************************')

        for pred, right_answer in zip(preds, right_answers):
            if get_acc(pred, right_answer):
                acc += 1
        num += len(questions)

    print(f'Acc: {acc/num}')

