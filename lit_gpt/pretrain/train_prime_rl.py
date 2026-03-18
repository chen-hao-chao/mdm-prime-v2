import glob
import math
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, Union
import math
import lightning as L
import torch
from lightning.fabric.strategies import FSDPStrategy, XLAStrategy
from torch.utils.data import DataLoader
from functools import partial
# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))
# from apex.optimizers import FusedAdam #torch optimizer has a cuda backend, which is faster actually
from lit_gpt.diffmodel import TransEncoder, Block, Config
from lit_gpt.packed_dataset import CombinedDataset, PackedDataset
from lit_gpt.speed_monitor import SpeedMonitorFabric as Monitor
from lit_gpt.speed_monitor import estimate_flops, measure_flops
from lit_gpt.utils import chunked_cross_entropy, get_default_supported_precision, num_parameters, step_csv_logger, lazy_load
from pytorch_lightning.loggers import WandbLogger
from flash_attn.losses.cross_entropy import CrossEntropyLoss
import random
import argparse

# ($)
import os
from subtokenizer.layers import BasebShufflingLayer

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--model', type=int, default=1028, help='model parameters')
    parse.add_argument('--nodes_num', type=int, default=1, help='number of nodes')
    parse.add_argument('--gpu_num', type=int, default=8, help='number of gpus')
    parse.add_argument('--flops', type=float, help='FLOPs, *e18')
    parse.add_argument('--ssl_ratio', type=float, help='stochastic sequence length ratio')
    parse.add_argument('--eval_freq', type=int, default=1000, help='eval freq')
    parse.add_argument('--batch_size', type=int, default=256, help='global_batch_size')
    parse.add_argument('--eval_batch_size', type=int, default=64, help='eval bs')
    parse.add_argument('--result_path', type=str, default='workdir', help='path to the results')
    parse.add_argument('--resume_path', type=str, default='', help='path to the checkpoint')
    parse.add_argument('--wandb_id', type=str, default='', help='wandb project ID')
    parse.add_argument('--run_name', type=str, default=None, help='run name')
    parse.add_argument('--chunk_point', type=float, default=0.5, help='time chunk point for different masking size')
    parse.add_argument('--wandb_project', type=str, default='mdm_prime_v2_1028M', help='wandb project name')
    parse.add_argument('--data_path', type=str, default='/download/slim_star_combined', help='data path')
    args = parse.parse_args()
    return args

args = parse_args()
args.base = 2
args.target_length = 15 # token granularity
chunk_size = args.target_length
model_name = f'Diff_LLaMA_{args.model}M'  # config
run_name = f'prime-{args.model}M-{args.flops}-ssl-{args.ssl_ratio}-l15' if args.run_name is None else args.run_name
result_dir = Path(args.result_path) / run_name

model_para_config = {
    '6': 6.294784, '19': 18.880896, '34': 33.563136, '48': 47.786688, '66': 65.54944,
    '85': 85.21408, '75': 75.38752, '113': 113.265408, '142': 141.581568, '170': 169.897728,
    '180': 179.856768, '206': 205.550464, '231': 231.24416, '268': 268.469248, '302': 302.027776,
    '336': 335.586304, '472': 471.90656, '551': 550.55744, '571': 571.001728, '629': 629.20832,
    '666': 666.168448, '717': 717.285888, '761': 761.335168, '831': 830.541312, '944': 943.796736,
    '1028': 1027.677952, '1233': 1233.213184, '1476': 1476.487168, '1678': 1677.826048, '2121': 2121.39328
}

# Hyperparameters
num_of_devices = args.gpu_num
global_batch_size = args.batch_size
learning_rate = 2e-4
if args.model <= 80:
    micro_batch_size = 32
elif args.model <= 200:
    micro_batch_size = 16
elif args.model <= 1500:
    micro_batch_size = 16
else:
    micro_batch_size = 4
average_length = 2048 * (1 - args.ssl_ratio) + (1 + 2048) * 0.5 * args.ssl_ratio
max_step = int(args.flops * 1e12 / (6 * model_para_config[f'{args.model}'] * global_batch_size * average_length))
warmup_steps = int(max_step / 100) if int(max_step / 100) > 100 else 100
log_step_interval = 10
eval_iters = int(100 * 1024 / global_batch_size)
save_step_interval = 5000
eval_step_interval = args.eval_freq


weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
min_lr = 2e-5

batch_size = global_batch_size // num_of_devices
gradient_accumulation_steps = batch_size // micro_batch_size
assert gradient_accumulation_steps > 0
warmup_iters = warmup_steps * gradient_accumulation_steps

max_iters = max_step * gradient_accumulation_steps
lr_decay_iters = max_iters
log_iter_interval = log_step_interval * gradient_accumulation_steps


# Treat all dataset equally by their size. If you want to use a different weight for a dataset, add it to the list with the weight.
train_data_config = [
    ("train_slimpajama", 1.0),
]

val_data_config = [
    ("validation", 1.0),
]

hparams = {k: v for k, v in locals().items() if isinstance(v, (int, float, str)) and not k.startswith("_")}
logger = step_csv_logger("out", model_name, flush_logs_every_n_steps=log_iter_interval)

# ($) --------------
from lm_eval import simple_evaluate
from lm_eval.api.model import LM
from transformers import AutoTokenizer
from lm_eval.api.instance import Instance
from datasets import Dataset
from tqdm import tqdm
# Ensure the imports from your evaluate_diff.py are present if you copy the class
import torch.nn.functional as F
class TrainingMDLMEvalHarness(LM):
    def __init__(
            self,
            model,
            tokenizer,
            subtokenizer,
            batch_size=args.eval_batch_size,
            base=2,
            target_length=15,
            device="cuda",
            **kwargs # Catch-all for other args
    ):
        super().__init__()
        self.model = model
        self.model.eval() # Ensure eval mode
        self.tokenizer = tokenizer
        self.subtokenizer = subtokenizer
        
        # Configuration from your original harness
        self.mask_id = base
        self.target_length = target_length
        self.base = base
        self.batch_size = int(batch_size)
        self.max_length = 2048 
        self.device = device
        
        # Hardcoded defaults from your evaluate_diff.py for consistency
        self.mc_num = 128  # reduced from 1024 for speed during training
        self.sampling_eps = 0.
        self.padding = False
        self.nll_type = 'mc' # or 'chain_rule'
        self.greddy = False
        self.cfg = 0.
        self.device = torch.device(device)

    def _forward_process(self, batch):
        B, L_l = batch.shape
        L = L_l // self.target_length
        # sample from U[0, 1] following https://arxiv.org/pdf/2107.00630 I.1
        u0 = torch.rand(1, device=batch.device, dtype=torch.float32)
        indices = torch.arange(B, device=batch.device).float()
        t = (u0 + indices / B) % 1

        p_mask = (1 - self.sampling_eps) * t + self.sampling_eps
        chunk_ratio_scheduler = lambda t: torch.where(t > args.chunk_point, chunk_size, 1)
        batch_ratios = torch.tensor([chunk_ratio_scheduler(val) for val in t], device=batch.device).view(B, 1)
        base_indices = torch.arange(L_l, device=batch.device).unsqueeze(0)
        gather_indices = base_indices.div(batch_ratios, rounding_mode='floor')
        raw_noise = torch.rand((B, L_l), device=batch.device)
        correlated_noise = torch.gather(raw_noise, dim=1, index=gather_indices)
        p_mask_expand = p_mask.view(B, 1)
        mask_indices = correlated_noise < p_mask_expand

        noisy_batch = torch.where(mask_indices, self.mask_id, batch)

        return noisy_batch, mask_indices, p_mask[:, None].repeat(1, L)

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
        B, L = seq.shape
        target_L = len(target)
        
        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)

        loss_acc = []
        for _ in range(self.mc_num // self.batch_size):
            perturbed_seq = seq.clone()
            perturbed_seq_sub = self.subtokenizer(perturbed_seq)
            perturbed_seq_, mask_indices, p_mask = self._forward_process(perturbed_seq_sub)
            perturbed_seq_sub[:, -(target_L*self.target_length):] = perturbed_seq_[:, -(target_L*self.target_length):]

            logits = self.get_logits(perturbed_seq_sub, prompt_index)
            loss_idx = torch.ones_like(seq).bool()
            loss_idx[:, :-target_L] = False
            loss = F.cross_entropy(logits[loss_idx], seq[loss_idx], reduction='none') / p_mask[loss_idx]
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

    def loglikelihood_rolling(self, requests): pass
    def generate_until(self, requests): pass

def run_zeroshot_eval(fabric, model, tokenizer, subtokenizer, step_count):
    fabric.print(f"Running Zero-Shot Evaluation at step {step_count}...")
    
    # 1. Wrap the current model in the Harness
    # Ensure model is unwrapped from Fabric if necessary, though usually Fabric handles it.
    # If using FSDP, we need to be careful. The simplest way is to pass the fabric model directly
    # and ensure the Harness uses fabric.print or standard torch ops.
    
    harness = TrainingMDLMEvalHarness(
        model=model,
        tokenizer=tokenizer,
        subtokenizer=subtokenizer,
        batch_size=args.eval_batch_size, # Adjust based on VRAM
        base=args.base,
        target_length=args.target_length,
        device=fabric.device
    )

    # 2. Run Evaluation
    results = simple_evaluate(
        model=harness,
        tasks=["arc_easy"], # Add your desired tasks here
        batch_size=args.eval_batch_size,
        device=str(fabric.device),
        limit=100
    )

    # 3. Log Results
    if fabric.global_rank == 0:
        # Extract scores
        for task, metrics in results['results'].items():
            acc = metrics.get('acc,none', metrics.get('acc', 0.0))
            acc_norm = metrics.get('acc_norm,none', metrics.get('acc_norm', 0.0))
            
            fabric.print(f"Task: {task} | Acc: {acc:.4f} | Acc Norm: {acc_norm:.4f}")
            
            # Log to WandB (using your existing logger setup)
            fabric.log_dict({
                f"eval/{task}_acc": acc,
                f"eval/{task}_acc_norm": acc_norm
            }, step=step_count)
            
    fabric.barrier()
# ------------------

def forward_process(batch, mask_id=2, eps=1e-3, subtokenizer=None):
    B, L = batch.shape
    y_0 = subtokenizer(batch)
    _, L_l = y_0.shape
    l = L_l // L

    t = torch.rand((B,), device=batch.device)

    p_mask = (1 - eps) * t + eps
    chunk_ratio_scheduler = lambda t: torch.where(t > args.chunk_point, chunk_size, 1)
    batch_ratios = torch.tensor([chunk_ratio_scheduler(val) for val in t], device=batch.device).view(B, 1)
    base_indices = torch.arange(L_l, device=batch.device).unsqueeze(0)
    gather_indices = base_indices.div(batch_ratios, rounding_mode='floor')
    raw_noise = torch.rand((B, L_l), device=batch.device)
    correlated_noise = torch.gather(raw_noise, dim=1, index=gather_indices)
    p_mask_expand = p_mask.view(B, 1)
    mask_indices = correlated_noise < p_mask_expand
    
    noisy_batch = torch.where(mask_indices, mask_id, y_0)
    return noisy_batch, mask_indices, p_mask[:, None].repeat(1, L)


def setup(
    devices: int = num_of_devices,
    train_data_dir: Path = Path(args.data_path),
    val_data_dir: Path = Path(args.data_path),
    precision: Optional[str] = None,
    tpu: bool = False,
    resume: Union[bool, Path] = True,
) -> None:
    wandb_save_dir = result_dir / 'wandb' 

    if args.wandb_id != '':
        wandb_logger = WandbLogger(name=f'{run_name}', save_dir=wandb_save_dir, project=args.wandb_project, resume="must", id=args.wandb_id)
    else:
        wandb_logger = WandbLogger(name=f'{run_name}', save_dir=wandb_save_dir, project=args.wandb_project)

    precision = precision or get_default_supported_precision(training=True, tpu=tpu)

    if devices > 1:
        if tpu:
            # For multi-host TPU training, the device count for Fabric is limited to the count on a single host.
            devices = "auto"
            strategy = XLAStrategy(sync_module_states=False)
        else:
            strategy = FSDPStrategy(
                auto_wrap_policy={Block},
                activation_checkpointing_policy=None,
                state_dict_type="full",
                limit_all_gathers=True,
                cpu_offload=False,
            )
    else:
        strategy = "auto"

    fabric = L.Fabric(devices=devices, strategy=strategy, precision=precision, loggers=[logger, wandb_logger])
    fabric.print(hparams)
    # fabric.launch(main, train_data_dir, val_data_dir, resume)
    main(fabric, train_data_dir, val_data_dir, resume)


def main(fabric, train_data_dir, val_data_dir, resume):
    monitor = Monitor(fabric, window_size=2, time_unit="seconds", log_iter_interval=log_iter_interval)

    if fabric.global_rank == 0:
        result_dir.mkdir(parents=True, exist_ok=True)

    config = Config.from_name(model_name)

    train_dataloader, val_dataloader = create_dataloaders(
        batch_size=micro_batch_size,
        block_size=config.block_size,
        fabric=fabric,
        train_data_dir=train_data_dir,
        val_data_dir=val_data_dir,
        seed=3407,
    )
    if val_dataloader is None:
        train_dataloader = fabric.setup_dataloaders(train_dataloader)
    else:
        train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)

    fabric.seed_everything(3407)  # same seed for every process to init model (FSDP)

    fabric.print(f"Loading model with {config.__dict__}")
    t0 = time.perf_counter()
    with fabric.init_module(empty_init=False):
        model = TransEncoder(config, target_length=args.target_length, base=args.base)
        model.apply(partial(model._init_weights ,n_layer=config.n_layer))
 

    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.")
    fabric.print(f"Total parameters {num_parameters(model):,}")

    model = fabric.setup(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2), foreach=False
    )
    # optimizer = FusedAdam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2),adam_w_mode=True)
    optimizer = fabric.setup_optimizers(optimizer)

    state = {"model": model, "optimizer": optimizer, "hparams": hparams, "iter_num": 0, "step_count": 0}

    if resume is True:
        import re
        def extract_number(filename):
            match = re.search(r'iter-(\d+)-ckpt\.pth', str(filename))
            return int(match.group(1)) if match else 0
        
        try:
            if args.resume_path != '':
                resume = args.resume_path
            else:
                resume = sorted(result_dir.glob("*.pth"), key=extract_number)[-1]
        except:
            resume = False
        
    if resume:
        fabric.print(f"Resuming training from {resume}")
        fabric.load(resume, state)

    # ($) subtokenizer
    fname = f"subtokenizer/perm/perm_32768.pt"
    if os.path.exists(fname):
        perm = torch.load(fname, map_location="cpu")
        subtokenizer = BasebShufflingLayer(base=args.base, target_length=args.target_length, perm=perm, vocab_size=32000)
    else:
        raise ValueError(f"Cannot find the permutation file.")
    
    # ($) Use the same tokenizer name as in evaluate_diff.py
    tokenizer = AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T')

    train_time = time.perf_counter()
    train(fabric, state, train_dataloader, val_dataloader, monitor, resume, subtokenizer=subtokenizer, tokenizer=tokenizer)
    fabric.print(f"Training time: {(time.perf_counter()-train_time):.2f}s")
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")


def train(fabric, state, train_dataloader, val_dataloader, monitor, resume, subtokenizer, tokenizer):
    model = state["model"]
    optimizer = state["optimizer"]

    # if val_dataloader is not None:
    #     validate(fabric, model, val_dataloader)  # sanity check

    with torch.device("meta"):
        meta_model = TransEncoder(model.config)
        # "estimated" is not as precise as "measured". Estimated is optimistic but widely used in the wild.
        # When comparing MFU or FLOP numbers with other projects that use estimated FLOPs,
        # consider passing `SpeedMonitor(flops_per_batch=estimated_flops)` instead
        estimated_flops = estimate_flops(meta_model) * micro_batch_size
        fabric.print(f"Estimated TFLOPs: {estimated_flops * fabric.world_size / 1e12:.2f}")
        x = torch.randint(0, 1, (micro_batch_size, model.config.block_size))
        # measured_flos run in meta. Will trigger fusedRMSNorm error
        #measured_flops = measure_flops(meta_model, x)
        #fabric.print(f"Measured TFLOPs: {measured_flops * fabric.world_size / 1e12:.2f}")
        del meta_model, x

    total_lengths = 0
    total_t0 = time.perf_counter()

    if fabric.device.type == "xla":
        import torch_xla.core.xla_model as xm

        xm.mark_step()
    
    
    initial_iter = state["iter_num"]
            
    loss_func = CrossEntropyLoss(reduction='none')
    for batch_idx, train_data in enumerate(train_dataloader):
        if state["iter_num"] < initial_iter:
            if state["iter_num"] % 100000 == 0:
                fabric.print(f'iter_num={state["iter_num"]}')
            continue 
        if state["iter_num"] >= max_iters:
            break
        
        # determine and set the learning rate for this iteration
        lr = get_lr(state["iter_num"]) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        iter_t0 = time.perf_counter()

        input_ids = train_data[:, 0 : model.config.block_size].contiguous()
        if torch.rand(1) < args.ssl_ratio:
            # This approach is not very elegant and involves some data waste.
            # However, since the actual data used for training is much smaller than the size of the dataset,
            # this method is still reasonable.
            length = torch.randint(1, model.config.block_size + 1, (1,))
            input_ids = input_ids[:, :length]
        noisy_input, mask_indices, p_mask = forward_process(input_ids, mask_id=args.base, subtokenizer=subtokenizer)
        is_accumulating = (state["iter_num"] + 1) % gradient_accumulation_steps != 0
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            logits = model(noisy_input)
            B, L = input_ids.shape
            loss = loss_func(logits.reshape(B*L, logits.size(-1)), input_ids.reshape(B*L)) / p_mask.reshape(B*L)
            loss = loss.sum() / (B*L)
            fabric.backward(loss / gradient_accumulation_steps)

        if not is_accumulating:
            fabric.clip_gradients(model, optimizer, max_norm=grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            state["step_count"] += 1
        elif fabric.device.type == "xla":
            xm.mark_step()
        state["iter_num"] += 1
        # input_id: B L 
        total_lengths += input_ids.size(1)
        t1 = time.perf_counter()
        fabric.print(
                f"iter {state['iter_num']} step {state['step_count']}: loss {loss.item():.4f}, iter time:"
                f" {(t1 - iter_t0) * 1000:.2f}ms{' (optimizer.step)' if not is_accumulating else ''}"
                f" remaining time: {(t1 - total_t0) / (state['iter_num'] - initial_iter) * (max_iters - state['iter_num']) / 3600:.2f} hours. " 
                # print days as well
                f" or {(t1 - total_t0) / (state['iter_num'] - initial_iter) * (max_iters - state['iter_num']) / 3600 / 24:.2f} days. "
            )
 
        monitor.on_train_batch_end(
            state["iter_num"] * micro_batch_size,
            t1 - total_t0,
            # this assumes that device FLOPs are the same and that all devices have the same batch size
            fabric.world_size,
            state["step_count"],
            flops_per_batch=estimated_flops,
            lengths=total_lengths,
            train_loss = loss.item()
        )

        if not is_accumulating and (state["step_count"] % eval_step_interval == 0 or state["step_count"] == max_step): 
            t0 = time.perf_counter()
            # ($)
            run_zeroshot_eval(fabric, model, tokenizer, subtokenizer, state["step_count"])
            model.train()
            t1 = time.perf_counter() - t0
            monitor.eval_end(t1)
            fabric.print(f"step {state['iter_num']}: val time: {t1 * 1000:.2f}ms")
            fabric.barrier()

        if not is_accumulating and (state["step_count"] % save_step_interval == 0 or state["step_count"] == max_step):
            checkpoint_path = result_dir / f"iter-{state['iter_num']:06d}-ckpt.pth"
            fabric.print(f"Saving checkpoint to {str(checkpoint_path)!r}")
            fabric.save(checkpoint_path, state)

        
@torch.no_grad()
def validate(fabric: L.Fabric, model: torch.nn.Module, val_dataloader: DataLoader, subtokenizer) -> torch.Tensor:
    fabric.print("Validating ...")
    model.eval()

    losses = torch.zeros(eval_iters, device=fabric.device)
    for k, val_data in enumerate(val_dataloader):
        if k >= eval_iters:
            break

        mc_loss = torch.zeros(128, device=fabric.device)  # mc_num=128
        for i in range(128):
            input_ids = val_data[:, 0 : model.config.block_size].contiguous()
            noisy_input, mask_indices, p_mask = forward_process(input_ids, subtokenizer=subtokenizer)
            logits = model(noisy_input)
            B, L = input_ids.shape
            loss =  torch.nn.functional.cross_entropy(logits.reshape(B*L, logits.size(-1)), input_ids.reshape(B*L), reduction='none') / p_mask.reshape(B*L)
            loss = loss.sum() / (B*L)
            mc_loss[i] = loss

        # loss_func = FusedCrossEntropyLoss()
        # loss = loss_func(logits, targets)
        losses[k] = mc_loss.mean().item()

    losses = fabric.all_reduce(losses, reduce_op="mean")
    out = losses.mean()

    model.train()
    return out


def create_dataloader(
    batch_size: int, block_size: int, data_dir: Path, fabric, shuffle: bool = True, seed: int = 12345, split="train"
) -> DataLoader:
    datasets = []
    data_config = train_data_config if split == "train" else val_data_config
    for prefix, _ in data_config:
        filenames = sorted(glob.glob(str(data_dir / f"{prefix}*")))
        random.seed(seed)
        random.shuffle(filenames)

        dataset = PackedDataset(
            filenames,
            # n_chunks control the buffer size. 
            # Note that the buffer size also impacts the random shuffle
            # (PackedDataset is an IterableDataset. So the shuffle is done by prefetch a buffer and shuffle the buffer)
            n_chunks=8 if split == "train" else 1,
            block_size=block_size,
            shuffle=shuffle,
            seed=seed+fabric.global_rank,
            num_processes=fabric.world_size,
            process_rank=fabric.global_rank,
        )
        datasets.append(dataset)

    if not datasets:
        raise RuntimeError(
            f"No data found at {data_dir}. Make sure you ran prepare_redpajama.py to create the dataset."
        )

    weights = [weight for _, weight in data_config]
    sum_weights = sum(weights)
    weights = [el / sum_weights for el in weights]

    combined_dataset = CombinedDataset(datasets=datasets, seed=seed, weights=weights)

    return DataLoader(combined_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)


def create_dataloaders(
    batch_size: int,
    block_size: int,
    fabric,
    train_data_dir: Path = Path(args.data_path),
    val_data_dir: Optional[Path] = None,
    seed: int = 12345,
) -> Tuple[DataLoader, DataLoader]:
    # Increase by one because we need the next word as well
    effective_block_size = block_size + 1
    train_dataloader = create_dataloader(
        batch_size=batch_size,
        block_size=effective_block_size,
        fabric=fabric,
        data_dir=train_data_dir,
        shuffle=True,
        seed=seed,
        split="train"
    )
    val_dataloader = (
        create_dataloader(
            batch_size=batch_size,
            block_size=effective_block_size,
            fabric=fabric,
            data_dir=val_data_dir,
            shuffle=False,
            seed=seed,
            split="validation"
        )
        if val_data_dir
        else None
    )
    return train_dataloader, val_dataloader


# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")
    setup()
