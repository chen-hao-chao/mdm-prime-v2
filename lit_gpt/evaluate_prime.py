import torch
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

import os
from subtokenizer.layers import BasebShufflingLayer
from huggingface_hub import hf_hub_download

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@register_model("prime")
class MDLMEvalHarness(LM):
    def __init__(
            self,
            model_name="1028",
            ckpt_path=None,
            vocab_size=32000,
            base=2,
            target_length=15,
            max_length=2048,
            batch_size=32,
            mc_num=1024,
            cfg=0.,
            chunk="0.0/1",
            device="cuda",
            temp=1.0,
            fname="subtokenizer/perm/perm_32768.pt",
            use_hf=False,
            cache_dir_hf=None
    ):

        super().__init__()

        model_name = f'Diff_LLaMA_{model_name}M'
        config = Config.from_name(model_name)
        self.model = TransEncoder(config, target_length=target_length, base=base, sum_emb=True).to(device)
        if use_hf:
            ckpt_path = hf_hub_download(repo_id="chen-hao-chao/mdm-prime-v2-slimpajama", 
                                        filename="mdm-prime-v2-3300flops-weight-only.pth",
                                        cache_dir=cache_dir_hf)
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()

        self.mask_id = base
        self.tokenizer = AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T')

        self.base = base
        self.target_length = target_length
        self.vocab_size = vocab_size

        # Handle the case where 'chunk' comes in as a string from CLI
        if isinstance(chunk, str):
            try:
                # specific format: "0.0_0.5/1_15"
                p_part, s_part = chunk.split('/')
                points = [float(x) for x in p_part.split('_')]
                sizes = [int(x) for x in s_part.split('_')]
                chunk = (points, sizes)
            except ValueError:
                print(f"Error parsing chunk string: {chunk}")
                raise

        points, sizes = chunk
        self.chunk_boundaries = torch.tensor(points[1:], dtype=torch.float, device=device)
        self.chunk_sizes = torch.tensor(sizes, dtype=torch.long, device=device)
        self.temp = temp
        self.temp_scheduler = lambda t: (1-self.temp) * t + self.temp
        if os.path.exists(fname):
            perm = torch.load(fname, map_location="cpu")
            self.subtokenizer = BasebShufflingLayer(base=self.base, target_length=self.target_length, perm=perm, vocab_size=vocab_size)
        else:
            raise ValueError(f"Cannot find the permutation file.")
        self.mc_num = mc_num
        self.batch_size = int(batch_size)
        assert mc_num % self.batch_size == 0
        self.sampling_eps = 0.
        self.max_length = max_length
        self.cfg = cfg
        self.device = torch.device(device)
        self.max_cont_len = 0
        self.max_len = 0

    def _forward_process(self, batch):
        B, L_l = batch.shape
        L = L_l // self.target_length

        # sample from U[0, 1] following https://arxiv.org/pdf/2107.00630 I.1
        u0 = torch.rand(1, device=batch.device, dtype=torch.float32)
        indices = torch.arange(B, device=batch.device).float()
        t = (u0 + indices / B) % 1

        alpha_t = 1 - t
        negative_weights = 1 / (t+1e-6)
        temp = self.temp_scheduler(t)
        p_mask = (1 - self.sampling_eps) * (1-alpha_t) + self.sampling_eps
        chunk_ratio_scheduler = lambda t: self.chunk_sizes[torch.bucketize(t, self.chunk_boundaries.to(t.device))]
        batch_ratios = torch.tensor([chunk_ratio_scheduler(val) for val in t], device=batch.device).view(B, 1)
        base_indices = torch.arange(L_l, device=batch.device).unsqueeze(0)
        gather_indices = base_indices.div(batch_ratios, rounding_mode='floor')
        raw_noise = torch.rand((B, L_l), device=batch.device)
        correlated_noise = torch.gather(raw_noise, dim=1, index=gather_indices)
        p_mask_expand = p_mask.view(B, 1)
        mask_indices = correlated_noise < p_mask_expand

        noisy_batch = torch.where(mask_indices, self.mask_id, batch)

        return noisy_batch, mask_indices, negative_weights[:, None].repeat(1, L), temp[:, None, None].repeat(1, L, self.vocab_size)

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

        input = batch

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            logits = self.model(input)

        if self.cfg > 0.:
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (self.cfg + 1) * (logits - un_logits)
        return logits[:, :batch.shape[1]]

    @torch.no_grad()
    def _eval_target_nll_mc(self, prefix, target):
        '''
        Employ Monte Carlo estimation to establish a lower bound of the log-likelihood
        '''
        seq = torch.concatenate([prefix, target])[None, :]
        seq = seq.repeat((self.batch_size, 1)).to(self.device)
        B, L = seq.shape
        target_L = len(target)
        if target_L > self.max_cont_len:
            self.max_cont_len = target_L
        if L > self.max_len:
            self.max_len = L
        loss_acc = []
        for _ in range(self.mc_num // self.batch_size):
            perturbed_seq = seq.clone()
            perturbed_seq_sub = self.subtokenizer(perturbed_seq)
            perturbed_seq_, mask_indices, negative_weights, temperature = self._forward_process(perturbed_seq_sub)
            perturbed_seq_sub[:, -(target_L*self.target_length):] = perturbed_seq_[:, -(target_L*self.target_length):]
            prompt_index = torch.arange(perturbed_seq_sub.shape[1], device=self.device) < len(prefix)*self.target_length
            logits = self.get_logits(perturbed_seq_sub, prompt_index) / temperature
            loss_idx = torch.ones_like(seq).bool()
            loss_idx[:, :-target_L] = False
            loss = F.cross_entropy(logits[loss_idx], seq[loss_idx], reduction='none') * negative_weights[loss_idx]
            loss = loss.sum() / self.batch_size
            loss_acc.append(loss.cpu())

        return sum(loss_acc) / len(loss_acc)

    def _encode_pair(self, context, continuation):
        # More standard approach
        context_enc = self.tokenizer.encode(context, add_special_tokens=False)
        continuation_enc = self.tokenizer.encode(continuation, add_special_tokens=False)
        
        # Check if tokenizer adds space when concatenating
        test_enc = self.tokenizer.encode(context + continuation, add_special_tokens=False)
        if test_enc != context_enc + continuation_enc:
            # Tokenizer merges tokens across boundary
            # Use the concatenated version and find the split point
            for i in range(len(context_enc), 0, -1):
                if test_enc[:i] == context_enc[:i]:
                    context_enc = test_enc[:i]
                    continuation_enc = test_enc[i:]
                    break
        
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
                ll = -self._eval_target_nll_mc(prefix, target)
                out.append((ll, 0.0))
        return out

    def loglikelihood_rolling(self, requests: list[Instance]):
        raise NotImplementedError

    def generate_until(self, requests: list[Instance]) -> list[str]:
        raise NotImplementedError

def parse_value(value_str):
    """Helper to convert string args to int/float if possible"""
    try:
        return int(value_str)
    except ValueError:
        try:
            return float(value_str)
        except ValueError:
            return value_str # Keep as string (e.g. 'cosine')

if __name__ == "__main__":
    set_seed(1234)
    cli_evaluate()