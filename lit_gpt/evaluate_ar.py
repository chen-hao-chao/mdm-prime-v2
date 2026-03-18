'''
This file is inspired by the code provided by the author of https://arxiv.org/abs/2406.11473
'''
import torch
import re
from pathlib import Path
import torch.nn.functional as F
from datasets import Dataset
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm

from transformers import AutoTokenizer
from lit_gpt.model import GPT, Config
from safetensors.torch import load_file

@register_model("ar")
class ArEvalHarness(LM):
    def __init__(
            self,
            batch_size,
            model_name="tiny",
            ckpt_path=None,
            device="cuda",
    ):
        super().__init__()

        model_name = f'Diff_LLaMA_{model_name}M'
        config = Config.from_name(model_name)
        self.model = GPT(config).to(device)

        self.model.load_state_dict(load_file(ckpt_path))
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T')  # TODO: bos in data?
        self.device = torch.device(device)

    @torch.no_grad()
    def _eval_target_nll(self, prefix, target):
        seq = torch.concatenate([prefix, target])[None, :]
        seq, target = seq.to(self.device), target.to(self.device)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            logits = self.model(seq)

        logits = logits[:, len(prefix) - 1: -1, :].view(-1, logits.shape[-1])
        loss = F.cross_entropy(logits, target, reduction='sum')
        loss = loss.cpu().float()

        return loss

    @torch.no_grad()
    def suffix_greedy_prediction(self, prefix, target):
        prefix, target = prefix.to(self.device), target.to(self.device)
        s = torch.cat([prefix, target]).unsqueeze(0)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            logits = self.model(s)
        logits = logits[:, len(prefix) - 1: -1, :]
        assert logits.shape[0] == 1
        logits = torch.squeeze(logits, dim=0)
        target_preds = torch.argmax(logits, dim=-1)
        correct = target == target_preds
        correct = torch.all(correct)
        return correct

    def _encode_pair(self, context, continuation):
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole_enc = self.tokenizer(context + continuation)["input_ids"]
        context_enc = self.tokenizer(context)["input_ids"]

        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]

        return context_enc, continuation_enc

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        def _tokenize(e):
            prefix, target = self._encode_pair(e["prefix"], e["target"])
            return {
                "prefix_text": e["prefix"],
                "target_text": e["target"],
                "prefix": prefix,
                "target": target,
            }

        ds = []
        ds = [{"prefix": req.args[0], "target": req.args[1]} for req in requests]
        ds = Dataset.from_list(ds)
        ds = ds.map(_tokenize)
        ds = ds.with_format("torch")
        prompt_len = [len(x["prefix"]) + len(x["target"]) for x in ds]

        assert max(prompt_len) <= 2048

        out = []
        with torch.no_grad():
            for elem in tqdm(ds, desc="Computing likelihood..."):
                prefix = elem["prefix"]
                target = elem["target"]

                ll = -self._eval_target_nll(prefix, target)
                is_target_greedy_dec = self.suffix_greedy_prediction(prefix, target)

                out.append((ll, 1.0 if is_target_greedy_dec else 0.0))
        return out

    def loglikelihood_rolling(self, requests: list[Instance]):
        raise NotImplementedError

    def generate_until(self, context, max_length, stop, **generation_kwargs):
        raise NotImplementedError


if __name__ == "__main__":
    cli_evaluate()
