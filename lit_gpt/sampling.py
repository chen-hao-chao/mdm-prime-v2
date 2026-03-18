import math
import torch
import random
import contextlib
import argparse
from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from lit_gpt.diffmodel import TransEncoder, Config
from transformers import AutoTokenizer, AutoModelForCausalLM
from subtokenizer.layers import BasebShufflingLayer

def sampling_mask_diff(model, tokenizer, subtokenizer, prefix, suffix, 
                       ref_token_count, device, target_length=15, mask_token_id=2, 
                       step_size = 0.005, chunk_pt=0.0, seed=None):
    """
    Implements discrete diffusion via partial masking sampling loop.
    """    
    # --- Configuration ---
    temperature = 1.0
    eps = 1e-12
    
    # --- Initialization ---
    # Create a fully masked batch
    # Shape: [Batch Size, Seq Len]
    prefix_enc = subtokenizer(prefix)
    suffix_enc = subtokenizer(suffix)
    mask_tokens = torch.full((1, ref_token_count*target_length,), mask_token_id, dtype=prefix.dtype, device=device)
    y_t = torch.cat([prefix_enc, mask_tokens, suffix_enc], dim=1)
    B, L_l = y_t.shape
    L = L_l // target_length

    # --- Discretize timesteps ---
    t_init = 1.0
    t_final = 1e-3
    n_steps = math.ceil((t_init - t_final) / step_size)
    chunk_ratio_scheduler = lambda t: 15 if t > chunk_pt else 1
    
    # Create schedule
    t_discretization = torch.tensor( [t_init - step_size * i for i in range(n_steps)] + [t_final], device=device)
    steps_counter = 0
    ctx = tqdm(total=(t_init - t_final), desc=f"NFE: {steps_counter}")

    rng_context = torch.random.fork_rng(devices=[device]) if seed is not None else contextlib.nullcontext()

    with rng_context:
        if seed is not None:
            torch.manual_seed(seed)

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

            # --- Logging ---
            steps_counter += 1
            ctx.n = (1 - t).item()
            ctx.refresh()
            ctx.set_description(f"NFE: {steps_counter}")

    return subtokenizer.inverse(y_t)

# 1. CONFIGURATION & ARGUMENT PARSING
parser = argparse.ArgumentParser(description="Run sampling with configurable parameters.")
parser.add_argument("--model_name", type=str, default=None, help="Name of the model (e.g., 'chen-hao-chao/mdm-prime-v2-slimpajama')")
parser.add_argument("--ckpt_path", type=str, default=None, help="Path to the model checkpoint")
parser.add_argument("--num_samples", type=str, default=5, help="Number of generated samples.")
parser.add_argument("--nfe", type=int, default=200, help="Number of function evaluations.")
parser.add_argument("--cache_dir_hf", type=str, default=None, help="Directory for HuggingFace cache")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

args = parser.parse_args()

# Assign arguments to variables
model_name = args.model_name
ckpt_path = args.ckpt_path
cache_dir_hf = args.cache_dir_hf
seed = args.seed
num_samples = str(args.num_samples)
nfe = args.nfe
device = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(seed)

if model_name == "chen-hao-chao/mdm-prime-v2-slimpajama" or ckpt_path is not None:
    tokenizer = AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T')
    target_length = 15
    base = 2
    vocab_size = 32000
    perm = torch.load("subtokenizer/perm/perm_32768.pt", map_location="cpu")
    subtokenizer = BasebShufflingLayer(base=base, target_length=target_length, perm=perm, vocab_size=vocab_size)
    config = Config.from_name('Diff_LLaMA_1028M')
    model = TransEncoder(config, target_length=target_length, base=base, sum_emb=True).to(device)
    if model_name is not None:
        ckpt_path = hf_hub_download(repo_id="chen-hao-chao/mdm-prime-v2-slimpajama", 
                                    filename="mdm-prime-v2-3300flops-weight-only.pth",
                                    cache_dir=cache_dir_hf)
        checkpoint = torch.load(ckpt_path, map_location='cpu')
    else:
        checkpoint = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(checkpoint["model"])
    model.eval()
else:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 2. DATA LOADING
print("Loading QMSum (Cleaned Mirror)...")
dataset = load_dataset("pszemraj/qmsum-cleaned", split=f"test[:{num_samples}]")

def parse_meeting_text(text):
    lines = text.split('\n')
    turns = []
    for line in lines:
        parts = line.split(':', 1)
        if len(parts) == 2:
            speaker = parts[0].strip()
            content = parts[1].strip()
        else:
            speaker = "Unknown"
            content = line.strip()

        if speaker == "Unknown" and "?" in content: continue
        if len(content) < 3: continue

        turns.append({"speaker": speaker, "text": content})
    return turns

def prepare_qmsum_input_randomized(row):
    turns = parse_meeting_text(row['input'])
    if len(turns) < 10: return None
    
    # FIND ALL VALID CANDIDATES 
    candidates = []
    for i in range(5, len(turns) - 5):
        word_count = len(turns[i]['text'].split())
        if word_count > 40: # Long turn constraint
            candidates.append(i)
            
    if not candidates: return None
    
    # PICK ONE RANDOMLY
    target_idx = random.choice(candidates)
    
    # BUILD PREFIX
    prefix_buffer = []
    current_tokens = 0
    for i in range(target_idx - 1, -1, -1):
        turn_text = f"{turns[i]['speaker']}: {turns[i]['text']}\n"
        turn_len = len(turn_text) // 4
        if current_tokens + turn_len > 200: 
            break
        prefix_buffer.insert(0, turn_text)
        current_tokens += turn_len
        
    prefix_str = "".join(prefix_buffer)
    prefix_prompt = prefix_str + f"{turns[target_idx]['speaker']}:"
    
    # BUILD SUFFIX
    suffix_buffer = []
    suffix_tokens = 0
    for i in range(target_idx + 1, len(turns)):
        turn_text = f"{turns[i]['speaker']}: {turns[i]['text']}\n"
        turn_len = len(turn_text) // 4
        if suffix_tokens + turn_len > 200: 
            break
        suffix_buffer.append(turn_text)
        suffix_tokens += turn_len
        
    suffix_str = "".join(suffix_buffer)

    return {
        "input_text": prefix_prompt,
        "reference": turns[target_idx]['text'],
        "suffix": suffix_str,
        "speaker": turns[target_idx]['speaker']
    }

predictions = []
references = []
valid_count = 0
seen_references = set()
print(f"Running QMSum Baseline (True Unique Samples)...")

for row in tqdm(dataset):    
    data = prepare_qmsum_input_randomized(row)
    if not data: continue
    ref_signature = data['reference']
    if ref_signature in seen_references:
        continue 
    seen_references.add(ref_signature)
    valid_count += 1
    
    # --- DYNAMIC LENGTH CALCULATION ---
    ref_tokens = tokenizer(data['reference'], add_special_tokens=False)['input_ids']
    ref_token_count = len(ref_tokens)
    prefix = tokenizer(data['input_text'], return_tensors="pt", truncation=True, max_length=2048).to(device)
    suffix = tokenizer(data['suffix'], return_tensors="pt", truncation=True, max_length=2048).to(device)

    if model_name == "chen-hao-chao/mdm-prime-v2-slimpajama":
        with torch.no_grad():
            outputs = sampling_mask_diff(model, tokenizer, subtokenizer, prefix['input_ids'], suffix['input_ids'], ref_token_count, device, step_size=1/nfe, seed=seed)
    else:
        outputs = model.generate(**prefix, max_new_tokens=ref_token_count, pad_token_id=tokenizer.eos_token_id, do_sample=True, eos_token_id=tokenizer.eos_token_id)

    input_len = prefix.input_ids.shape[1]
    generated_text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)    
    first_turn_only = generated_text.strip()
    predictions.append(first_turn_only)
    references.append(data['reference'])

    tqdm.write(f"\n=== Unique Sample {valid_count} ===")
    tqdm.write(f"PREFIX: {data['input_text']}\n\n")
    tqdm.write(f"SUFFIX: {data['suffix']}\n\n")
    tqdm.write(f"GENERATED: {first_turn_only}\n\n")
    tqdm.write(f"REFERENCE: {data['reference']}")
    tqdm.write("==============================\n")
