import math
import re
import torch
import torch.nn.functional as F
import numpy as np
import random
import contextlib
import argparse
from pathlib import Path
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from lit_gpt.diffmodel import TransEncoder, Config
from transformers import AutoTokenizer, AutoModelForCausalLM
from subtokenizer.layers import BasebShufflingLayer

def sampling_mask_diff(model, subtokenizer, seq_length, device, target_length=15, temperature=1.0,
                       mask_token_id=2, step_size=0.005, chunk_ratio_scheduler=None, seed=None,
                       use_corrector_steps=False):
    """
    Implements discrete diffusion via partial masking sampling loop (unconditional).
    Generates a fully masked sequence of `seq_length` tokens and denoises it.
    """
    # --- Set random seeds for reproducibility ---
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    # --- Configuration ---
    eps = 1e-12

    # --- Initialization ---
    # Create a fully masked sequence
    L_l = seq_length * target_length
    y_t = torch.full((1, L_l), mask_token_id, dtype=torch.long, device=device)
    B = 1
    L = seq_length

    # --- Discretize timesteps ---
    t_init = 1.0
    t_final = 1e-3
    n_steps = math.ceil((t_init - t_final) / step_size)

    # Create schedule
    t_discretization = torch.tensor( [t_init - step_size * i for i in range(n_steps)] + [t_final], device=device)
    steps_counter = 0
    ctx = tqdm(total=(t_init - t_final), desc=f"NFE: {steps_counter}")

    with ctx:
        for i in range(n_steps):
            t = t_discretization[i]
            s = t - step_size
            alpha_t = 1 - t
            alpha_s = 1 - s
            chunk_ratio = int(chunk_ratio_scheduler(t))

            # Assuming the wrapped model handles the arguments:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits = model(y_t)

            dist = torch.distributions.Categorical(logits=(logits / temperature))
            x_0 = dist.sample()
            y_0 = subtokenizer(x_0)

            # --- Update Step (Partial Masking) ---
            if i == n_steps - 1:
                is_mask = (y_t == mask_token_id)
                y_t[is_mask] = y_0[is_mask]
            else:
                is_mask = (y_t == mask_token_id)
                p_unmask = torch.full((B, L_l // chunk_ratio, 1), (alpha_s - alpha_t) / (1 - alpha_t + eps), device=y_t.device, dtype=torch.float32)
                unmask_indices = torch.rand(size=(B, L_l // chunk_ratio, 1), device=device) < p_unmask
                unmask_indices = unmask_indices.expand(-1, -1, chunk_ratio).reshape(B, L_l)
                flip_to_y0 = unmask_indices & is_mask

                y_t[flip_to_y0] = y_0[flip_to_y0]
            
            # --- Corrector Step ---
            # For t \in [0.5, 0.75], re-mask the lowest-confidence 10% of unmasked subtokens
            # (chunk_ratio=1), then predict again with the model.
            if use_corrector_steps and (t > 0.5 and t < 0.75):
                is_unmasked = (y_t != mask_token_id)
                num_unmasked = is_unmasked.sum(dim=-1)  # (B,)
                k = max(1, int(num_unmasked[0].item() * 0.1))

                # Per-token confidence from predictor logits, expanded to subtoken level
                token_probs = F.softmax(logits.float(), dim=-1)  # (B, L, vocab_size)
                token_conf = token_probs.gather(-1, x_0.unsqueeze(-1)).squeeze(-1)  # (B, L)
                subtoken_conf = token_conf.unsqueeze(-1).expand(-1, -1, target_length).reshape(B, L_l)

                # Exclude already-masked positions from selection
                subtoken_conf = torch.where(is_unmasked, subtoken_conf,
                                            torch.full_like(subtoken_conf, float('inf')))
                _, lowest_indices = subtoken_conf.topk(k, dim=-1, largest=False)
                # Randomly select 50% of the lowest-confidence candidates to re-mask
                rand_select = torch.rand(lowest_indices.shape, device=device) < 0.5
                selected_indices = lowest_indices[rand_select].unsqueeze(0)
                remask_mask = torch.zeros_like(y_t, dtype=torch.bool)
                if selected_indices.numel() > 0:
                    remask_mask.scatter_(1, selected_indices, True)
                y_t[remask_mask] = mask_token_id

                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    logits_corr = model(y_t)

                dist_corr = torch.distributions.Categorical(logits=(logits_corr / temperature))
                x_0_corr = dist_corr.sample()
                y_0_corr = subtokenizer(x_0_corr)

                # Update all lowest_indices positions with corrector predictions
                y_t.scatter_(1, lowest_indices, torch.gather(y_0_corr, 1, lowest_indices))

                steps_counter += 1

            # --- Logging ---
            steps_counter += 1
            ctx.n = (1 - t).item()
            ctx.refresh()
            ctx.set_description(f"NFE: {steps_counter}")
    
    return subtokenizer.inverse(y_t)

# 1. CONFIGURATION & ARGUMENT PARSING
parser = argparse.ArgumentParser(description="Run unconditional sampling (no prefix/suffix).")
parser.add_argument("--model_name", type=str, default=None, help="Name of the model (e.g., 'chen-hao-chao/mdm-prime-v2-slimpajama')")
parser.add_argument("--ckpt_path", type=str, default=None, help="Path to the model checkpoint")
parser.add_argument("--checkpoint_dir", type=str, default=None, help="Path to checkpoint folder containing iter-*-ckpt.pth files")
parser.add_argument("--model_size", type=int, default=1028, help="Model size in M (e.g. 170, 1028)")
parser.add_argument("--num_samples", type=int, default=5, help="Number of generated samples.")
parser.add_argument("--seq_length", type=int, default=2048, help="Number of tokens to generate per sample.")
parser.add_argument("--nfe", type=int, default=200, help="Number of function evaluations.")
parser.add_argument("--cache_dir_hf", type=str, default=None, help="Directory for HuggingFace cache")
parser.add_argument("--seed", type=int, nargs='+', default=[42], help="Random seed(s) for reproducibility. Provide one seed per sample, or a single seed to use as base.")
parser.add_argument("--eval_model", type=str, default='TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T',
                    help="Evaluation model for computing perplexity (e.g., 'TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T')")
parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature (default: 1.0)")

args = parser.parse_args()

# Assign arguments to variables
model_name = args.model_name
ckpt_path = args.ckpt_path
cache_dir_hf = args.cache_dir_hf
num_samples = args.num_samples
seq_length = args.seq_length
nfe = args.nfe
device = "cuda" if torch.cuda.is_available() else "cpu"

# Handle seed(s) - can be a single seed or one per sample
seeds = args.seed
if len(seeds) == 1:
    # Single seed provided - use it as base and add sample index
    seeds = [seeds[0] + i for i in range(num_samples)]
    print(f"Using base seed {args.seed[0]}, generating seeds: {seeds}")
else:
    # Multiple seeds provided - use them directly
    if len(seeds) < num_samples:
        raise ValueError(f"Not enough seeds provided: got {len(seeds)}, need {num_samples}")
    seeds = seeds[:num_samples]
    print(f"Using provided seeds: {seeds}")

# Set all random seeds for reproducibility (using first seed for initialization)
torch.manual_seed(seeds[0])
np.random.seed(seeds[0])
random.seed(seeds[0])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seeds[0])

checkpoint_dir = args.checkpoint_dir
eval_model_name = args.eval_model

print(f"Loading evaluation model: {eval_model_name}")
eval_model = AutoModelForCausalLM.from_pretrained(
    eval_model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    cache_dir=cache_dir_hf,
)
eval_model.eval()

# SMDM Model Configuration
if model_name == "nieshen/SMDM":
    print("Loading SMDM model with target_length=1...")
    tokenizer = AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T', cache_dir=cache_dir_hf)
    target_length = 1
    base = 32000
    vocab_size = 32000
    use_corrector_steps = False
    chunk_ratio_scheduler = lambda t: target_length
    nfe_ = nfe
    temperature = args.temperature
    # For target_length=1, we use perm=None (creates random permutation)
    # since perm_2.pt may not exist
    perm = None
    subtokenizer = BasebShufflingLayer(base=base, target_length=target_length, perm=perm, vocab_size=vocab_size)
    model_config_name = f'Diff_LLaMA_{args.model_size}M'
    config = Config.from_name(model_config_name)
    model = TransEncoder(config, target_length=target_length, base=base, sum_emb=True).to(device)

    # Download the safetensors file from HuggingFace
    print("Downloading SMDM checkpoint from HuggingFace...")
    ckpt_path = hf_hub_download(
        repo_id="nieshen/SMDM",
        filename="mdm_safetensors/mdm-1028M-3300e18-rsl-0.01-bs-1024.safetensors",
        cache_dir=cache_dir_hf
    )
    print(f"Checkpoint downloaded to: {ckpt_path}")

    # Load safetensors file
    state_dict = load_file(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()

# MDM-Prime-v2 Model Configuration
elif model_name == "chen-hao-chao/mdm-prime-v2-slimpajama" or ckpt_path is not None or checkpoint_dir is not None:
    tokenizer = AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T', cache_dir=cache_dir_hf)
    target_length = 15
    base = 2
    vocab_size = 32000
    use_corrector_steps = True
    temperature = args.temperature
    chunk_ratio_scheduler = lambda t: target_length
    nfe_ = nfe * 3 / 4 # since we are going to use corrector steps for 25% of the specified steps
    perm = torch.load("subtokenizer/perm/perm_32768.pt", map_location="cpu")
    subtokenizer = BasebShufflingLayer(base=base, target_length=target_length, perm=perm, vocab_size=vocab_size)
    model_config_name = f'Diff_LLaMA_{args.model_size}M'
    config = Config.from_name(model_config_name)
    model = TransEncoder(config, target_length=target_length, base=base, sum_emb=True).to(device)
    if checkpoint_dir is not None:
        # Load the latest checkpoint from the directory
        def extract_number(filename):
            match = re.search(r'iter-(\d+)-ckpt\.pth', str(filename))
            return int(match.group(1)) if match else 0

        checkpoint_dir_path = Path(checkpoint_dir)
        ckpt_files = sorted(checkpoint_dir_path.glob("*.pth"), key=extract_number)
        if not ckpt_files:
            raise RuntimeError(f"No .pth checkpoint files found in {checkpoint_dir}")
        ckpt_path = ckpt_files[-1]
        print(f"Loading checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location='cpu')
    elif model_name is not None:
        ckpt_path = hf_hub_download(repo_id="chen-hao-chao/mdm-prime-v2-slimpajama",
                                    filename="mdm-prime-v2-3300flops-weight-only.pth",
                                    cache_dir=cache_dir_hf)
        checkpoint = torch.load(ckpt_path, map_location='cpu')
    else:
        checkpoint = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(checkpoint["model"])
    model.eval()
else:
    raise NotImplementedError(f"Model '{model_name}' is not supported. Use 'nieshen/SMDM' or 'chen-hao-chao/mdm-prime-v2-slimpajama', or provide --ckpt_path or --checkpoint_dir.") 

# 2. UNCONDITIONAL GENERATION
predictions = []
print(f"Generating {num_samples} unconditional samples (seq_length={seq_length})...")

for sample_idx in range(num_samples):
    current_seed = seeds[sample_idx]
    print(f"\n--- Sample {sample_idx + 1}/{num_samples} (seed={current_seed}) ---")
    with torch.no_grad():
        outputs = sampling_mask_diff(model, subtokenizer, seq_length, device,
                                        step_size=1/nfe_, seed=current_seed,
                                        target_length=target_length, mask_token_id=base,
                                        chunk_ratio_scheduler=chunk_ratio_scheduler,
                                        use_corrector_steps=use_corrector_steps,
                                        temperature=temperature)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    predictions.append(generated_text.strip())
    print(f"GENERATED: {generated_text.strip()}")
    print("=" * 50)

# Save results to folder
if model_name == "nieshen/SMDM":
    model_short_name = "smdm"
elif model_name == "chen-hao-chao/mdm-prime-v2-slimpajama" or ckpt_path is not None or checkpoint_dir is not None:
    model_short_name = "prime"

# Create output folder inside sampling_results
output_folder = Path("sampling_results") / f"results_{model_short_name}_nfe={nfe}"
output_folder.mkdir(parents=True, exist_ok=True)

# Save samples to file
samples_filename = Path(output_folder) / "samples.txt"
with open(samples_filename, 'w') as f:
    for idx, text in enumerate(predictions):
        f.write(f"Sample {idx + 1}:\n{text}\n\n")
print(f"\nSamples saved to: {samples_filename}")

# 3. EVALUATE ENTROPY & GENERATIVE PERPLEXITY
print(f"\n{'='*50}")
print(f"Evaluating entropy and generative perplexity using {eval_model_name}...")
print(f"{'='*50}")

per_sample_entropy = []
per_sample_ppl = []

for i, gen_text in enumerate(tqdm(predictions, desc="Computing metrics")):
    inputs = tokenizer(gen_text, return_tensors="pt", truncation=True, max_length=2048).to(device)
    input_ids = inputs["input_ids"]

    if input_ids.shape[1] < 2:
        continue

    with torch.no_grad():
        outputs = eval_model(**inputs)
        logits = outputs.logits  # (1, seq_len, vocab_size)

    # Shift for next-token prediction: logits[:-1] predict tokens[1:]
    shift_logits = logits[:, :-1, :].float()
    shift_labels = input_ids[:, 1:]

    # --- Entropy: average token-level entropy of the predicted distribution ---
    probs = F.softmax(shift_logits, dim=-1)
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_entropy = -(probs * log_probs).sum(dim=-1)  # (1, seq_len-1)
    avg_entropy = token_entropy.mean().item()
    per_sample_entropy.append(avg_entropy)

    # --- Generative perplexity: exp(avg NLL under TinyLlama) ---
    nll = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        reduction='mean'
    ).item()
    per_sample_ppl.append(math.exp(nll))

mean_entropy = np.mean(per_sample_entropy)
mean_ppl = np.mean(per_sample_ppl)

print(f"\n{'='*50}")
print(f"Results over {len(per_sample_ppl)} generated samples:")
print(f"  Mean Entropy:                {mean_entropy:.4f}")
print(f"  Mean Generative Perplexity:  {mean_ppl:.4f}")
print(f"{'='*50}")

# Save metrics to separate file
metrics_filename = Path(output_folder) / "metrics.txt"
with open(metrics_filename, 'w') as f:
    f.write("=" * 50 + "\n")
    f.write(f"Results over {len(per_sample_ppl)} generated samples:\n")
    f.write(f"  Mean Entropy:                {mean_entropy:.4f}\n")
    f.write(f"  Mean Generative Perplexity:  {mean_ppl:.4f}\n")
    f.write("=" * 50 + "\n\n")
print(f"\nMetrics saved to: {metrics_filename}")
