'''
This file is inspired by the code provided by the author of https://arxiv.org/abs/2406.11473
'''
import torch
import re
from pathlib import Path
import random
import numpy as np
import torch.nn.functional as F
from datasets import Dataset
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm

from transformers import AutoTokenizer
from lit_gpt.diffmodel import TransEncoder, Config
from safetensors.torch import load_file


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@register_model("mdlm")
class MDLMEvalHarness(LM):
    def __init__(
            self,
            model_name="tiny",
            ckpt_path=None,
            mask_id=32000,
            max_length=2048,
            batch_size=32,
            mc_num=1024,
            padding=False,
            nll_type='mc',
            greddy=False,
            cfg=0.,
            device="cuda",
    ):
        super().__init__()
        assert nll_type in ['mc', 'chain_rule']

        model_name = f'Diff_LLaMA_{model_name}M'
        config = Config.from_name(model_name)
        self.model = TransEncoder(config).to(device)

        self.model.load_state_dict(load_file(ckpt_path))
        self.model.eval()

        self.mask_id = mask_id
        self.tokenizer = AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T')

        self.mc_num = mc_num
        self.batch_size = int(batch_size)
        assert mc_num % self.batch_size == 0
        self.sampling_eps = 0.
        self.max_length = max_length
        self.padding = padding
        self.nll_type = nll_type
        self.greddy = greddy

        self.cfg = cfg
        self.device = torch.device(device)

    def _forward_process(self, batch):
        b, l = batch.shape
        # sample from U[0, 1] following https://arxiv.org/pdf/2107.00630 I.1
        u0 = torch.rand(1, device=batch.device, dtype=torch.float32)
        indices = torch.arange(b, device=batch.device).float()
        t = (u0 + indices / b) % 1

        p_mask = (1 - self.sampling_eps) * t + self.sampling_eps

        p_mask = p_mask[:, None].repeat(1, l)

        mask_indices = torch.rand((b, l), device=batch.device) < p_mask
        noisy_batch = torch.where(mask_indices, self.mask_id, batch)

        return noisy_batch, p_mask

    @torch.no_grad()
    def get_logits(self, batch, prompt_index):
        '''
        prompt_index : 1D bool tensor, length=batch.shape[1]
        '''
        if self.cfg > 0.:
            assert len(prompt_index) == batch.shape[1]
            prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
            un_batch = batch.clone()
            un_batch[prompt_index] = self.mask_id
            batch = torch.cat([batch, un_batch])

        if self.padding:
            input = torch.full((batch.size(0), 2048), self.mask_id, device=self.device)
            input[:, :batch.shape[1]] = batch
        else:
            input = batch

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            logits = self.model(input)

        if self.cfg > 0.:
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (self.cfg + 1) * (logits - un_logits)
        return logits[:, :batch.shape[1]]

    @torch.no_grad()
    def _eval_target_nll_ar(self, prefix, target):
        '''
        Utilize the chain rule to compute the likelihood
        We need to perform len(target) forward passes in parallel
        '''
        prefix, target = prefix.unsqueeze(0), target.unsqueeze(0) # 1*l1, 1*l2

        prompt_index = torch.arange(prefix.shape[1] + target.shape[1], device=self.device) < prefix.shape[1]
        perturbed_ = target.repeat(target.shape[1], 1).clone().contiguous() # l2*l2

        mask_index = torch.ones((perturbed_.shape[1], perturbed_.shape[1]), dtype=torch.bool)
        mask_index = torch.triu(mask_index)

        perturbed_[mask_index] = self.mask_id
        perturbed_seq = torch.cat([prefix.repeat(perturbed_.shape[0], 1), perturbed_], dim=-1)

        logits_ = []
        num = len(perturbed_seq) // self.batch_size if len(perturbed_seq) % self.batch_size == 0 else len(perturbed_seq) // self.batch_size + 1
        for i in range(num):
            end = (i + 1) * self.batch_size if (i + 1) * self.batch_size < len(perturbed_seq) else len(perturbed_seq)
            perturbed_seq_ = perturbed_seq[i * self.batch_size: end]
            perturbed_seq_ = perturbed_seq_.to(self.device)
            if len(perturbed_seq_.shape) == 1:
                perturbed_seq_ = perturbed_seq_.unsqueeze(0)
            logits = self.get_logits(perturbed_seq_, prompt_index)
            logits_.append(logits.cpu())
        logits = torch.cat(logits_, dim=0)

        temp_index = torch.ones((perturbed_.shape[1], perturbed_.shape[1]), dtype=torch.bool)
        temp_index = torch.triu(temp_index, diagonal=1)
        mask_index[temp_index] = False
        logits_index = torch.cat([torch.zeros((perturbed_.shape[1], prefix.shape[1]), dtype=torch.bool), mask_index], dim=-1)
        loss = F.cross_entropy(logits[logits_index], target[0], reduction='sum').cpu().float()
        return loss


    @torch.no_grad()
    def _eval_target_nll_mc(self, prefix, target):
        '''
        Employ Monte Carlo estimation to establish a lower bound of the log-likelihood
        '''
        seq = torch.concatenate([prefix, target])[None, :]
        seq = seq.repeat((self.batch_size, 1)).to(self.device)

        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)

        loss_acc = []
        for _ in range(self.mc_num // self.batch_size):
            perturbed_seq = seq.clone()
            perturbed_seq_, p_mask = self._forward_process(seq)
            perturbed_seq[:, -len(target):] = perturbed_seq_[:, -len(target):]

            mask_indices = perturbed_seq == self.mask_id

            logits = self.get_logits(perturbed_seq, prompt_index)

            loss = F.cross_entropy(logits[mask_indices], seq[mask_indices], reduction='none') / p_mask[mask_indices]
            loss = loss.sum() / self.batch_size
            loss_acc.append(loss.cpu())

        return sum(loss_acc) / len(loss_acc)

    @torch.no_grad()
    def suffix_greedy_prediction(self, prefix, target):
        if not self.greddy:
            return False

        seq = torch.full((1, len(prefix) + len(target)), self.mask_id, device=self.device)
        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)
        prefix, target = prefix.to(self.device), target.to(self.device)
        seq[0, :len(prefix)] = prefix

        for i in range(len(target)):
            mask_index = (seq == self.mask_id)
            logits = self.get_logits(seq, prompt_index)[mask_index]
            x0 = torch.argmax(logits, dim=-1)

            p = torch.softmax(logits.to(torch.float32), dim=-1)
            confidence = torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)).squeeze(dim=-1)
            _, index = torch.sort(confidence, descending=True)
            x0[index[1:]] = self.mask_id
            seq[mask_index] = x0.clone()
        correct = target == seq[0, len(prefix):]
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

                if self.nll_type == 'mc':
                    ll = -self._eval_target_nll_mc(prefix, target)
                elif self.nll_type == 'chain_rule':
                    ll = -self._eval_target_nll_ar(prefix, target)
                else:
                    raise NotImplementedError(self.nll_type)

                is_target_greedy_dec = self.suffix_greedy_prediction(prefix, target)

                out.append((ll, 1.0 if is_target_greedy_dec else 0.0))
        return out

    def loglikelihood_rolling(self, requests: list[Instance]):
        raise NotImplementedError

    def generate_until(self, context, max_length, stop, **generation_kwargs):
        raise NotImplementedError


if __name__ == "__main__":
    set_seed(1234)
    cli_evaluate()