"""
Gradio chatbot demo for MDM-Prime-v2.

Usage (inside container):
    python /workspace/gradio_demo.py
    python /workspace/gradio_demo.py --test   # UI-only, no model loaded
"""

import sys
import os
import subprocess
import math
import time
import contextlib
import threading

# ---------------------------------------------------------------------------
# Auto-install gradio if not available.
# ---------------------------------------------------------------------------
try:
    import gradio as gr
except ImportError:
    print("Gradio not found — installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gradio"])
    import gradio as gr

# ---------------------------------------------------------------------------
# Test mode: skip all heavy imports, use dummy responses.
# ---------------------------------------------------------------------------
TEST_MODE = "--test" in sys.argv

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_LIT_GPT_DIR = os.path.join(_SCRIPT_DIR, "lit_gpt")
if _LIT_GPT_DIR not in sys.path:
    sys.path.insert(0, _LIT_GPT_DIR)

# set_deterministic
def set_deterministic(seed, torch):
    # Pytorch
    if seed:
        torch.manual_seed(seed)


# ============================= sampling logic ==============================

def sampling_mask_diff(
    torch, model, tokenizer, subtokenizer, prefix, suffix,
    ref_token_count, device, target_length=15, mask_token_id=2,
    step_size=0.005, chunk_pt=0.0, seed=None, progress=None, corrector_steps=5
):
    """Discrete-diffusion partial-masking sampling loop."""
    temperature = 0.5 if ref_token_count < 5 else 1.0
    eps = 1e-12

    prefix_enc = subtokenizer(prefix)
    suffix_enc = subtokenizer(suffix)
    mask_tokens = torch.full(
        (1, ref_token_count * target_length),
        mask_token_id, dtype=prefix.dtype, device=device,
    )
    y_t = torch.cat([prefix_enc, mask_tokens, suffix_enc], dim=1)
    B, L_l = y_t.shape

    t_init = 1.0
    t_final = 1e-3
    n_steps = math.ceil((t_init - t_final) / step_size)
    chunk_ratio_scheduler = lambda t: 15 if t > chunk_pt else 1

    t_discretization = torch.tensor(
        [t_init - step_size * i for i in range(n_steps)] + [t_final],
        device=device,
    )

    cuda_devices = [0] if device == "cuda" else []
    rng_context = (
        torch.random.fork_rng(devices=cuda_devices)
        if seed is not None
        else contextlib.nullcontext()
    )
    with rng_context:
        if seed is not None:
            torch.manual_seed(seed)

    for i in range(n_steps):
        t = t_discretization[i]
        s = t - step_size
        alpha_t = 1 - t
        alpha_s = 1 - s
        chunk_ratio = int(chunk_ratio_scheduler(t))

        autocast_ctx = (
            torch.cuda.amp.autocast(dtype=torch.bfloat16)
            if device == "cuda" else contextlib.nullcontext()
        )
        with autocast_ctx:
            logits = model(y_t)

        dist = torch.distributions.Categorical(logits=(logits / temperature))
        x_0 = dist.sample()
        y_0 = subtokenizer(x_0)

        if i == n_steps - 1:
            is_mask = y_t == mask_token_id
            y_t[is_mask] = y_0[is_mask]
        else:
            is_mask = y_t == mask_token_id
            p_unmask = torch.full(
                (B, L_l // chunk_ratio, 1),
                (alpha_s - alpha_t) / (1 - alpha_t + eps),
                device=y_t.device, dtype=torch.float32,
            )
            unmask_indices = (
                torch.rand(size=(B, L_l // chunk_ratio, 1), device=device) < p_unmask
            )
            unmask_indices = unmask_indices.expand(-1, -1, chunk_ratio).reshape(B, L_l)
            flip_to_y0 = unmask_indices & is_mask
            y_t[flip_to_y0] = y_0[flip_to_y0]

        
        if progress is not None:
            progress((i + 1) / n_steps, desc="Diffusion sampling...")
    
    for i in range(corrector_steps):
        p_unmask = torch.full((B, L_l, 1), 0.1, device=y_t.device, dtype=torch.float32,)
        unmask_indices = (torch.rand(size=(B, L_l, 1), device=device) < p_unmask)
        unmask_indices = unmask_indices.expand(-1, -1, 1).reshape(B, L_l)
        y_t[unmask_indices] = mask_token_id
        autocast_ctx = (
            torch.cuda.amp.autocast(dtype=torch.bfloat16)
            if device == "cuda" else contextlib.nullcontext()
        )
        with autocast_ctx:
            logits = model(y_t)
        dist = torch.distributions.Categorical(logits=(logits / temperature))
        x_0 = dist.sample()
        y_0 = subtokenizer(x_0)
        y_t[unmask_indices] = y_0[unmask_indices]
        if progress is not None:
            progress((i + 1) / corrector_steps, desc="Corrector step...")

    return subtokenizer.inverse(y_t)


# ============================== model loading ==============================

_model_cache = {"result": None, "loading": False, "error": None}


def load_model(ckpt_path=None, model_name=None):
    """Load tokenizer, subtokenizer, and diffusion model."""
    import torch
    from transformers import AutoTokenizer
    from huggingface_hub import hf_hub_download
    from lit_gpt.diffmodel import TransEncoder, Config
    from subtokenizer.layers import BasebShufflingLayer

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    target_length = 15
    base = 2
    vocab_size = 32000

    perm_path = os.path.join(_LIT_GPT_DIR, "subtokenizer", "perm", "perm_32768.pt")
    perm = torch.load(perm_path, map_location="cpu")
    subtokenizer = BasebShufflingLayer(
        base=base, target_length=target_length, perm=perm, vocab_size=vocab_size,
    )

    config = Config.from_name("Diff_LLaMA_1028M")
    model = TransEncoder(
        config, target_length=target_length, base=base, sum_emb=True,
    ).to(device)

    if ckpt_path is None:
        if model_name is None:
            model_name = "chen-hao-chao/mdm-prime-v2-slimpajama"
        ckpt_path = hf_hub_download(
            repo_id=model_name,
            filename="mdm-prime-v2-3300flops-weight-only.pth",
        )

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.eval()

    return torch, model, tokenizer, subtokenizer, device


def _bg_load():
    try:
        _model_cache["result"] = load_model()
    except Exception as e:
        _model_cache["error"] = str(e)


if not TEST_MODE:
    _model_cache["loading"] = True
    threading.Thread(target=_bg_load, daemon=True).start()
    print("Loading model in background...")


# ================================ Gradio UI ================================

# ---------------------------------------------------------------------------
# Content moderation — regex pre-filter + KoalaAI/Text-Moderation (local)
# ---------------------------------------------------------------------------
import re
from transformers import pipeline as hf_pipeline

# --- Regex pre-filter (fast, catches obvious leet-speak bypasses) ----------

_LEET_TABLE = str.maketrans({
    "0": "o", "1": "i", "3": "e", "4": "a",
    "5": "s", "7": "t", "@": "a", "$": "s",
    "!": "i", "+": "t", "|": "i",
})

def _normalize(text: str) -> str:
    text = text.lower().translate(_LEET_TABLE)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

_BLOCKED_REGEXES: list[re.Pattern] = [re.compile(p) for p in [
    r'\bn[i\*]+g+[ae]r\b',
    r'\bf[a\*]+g+[oi]*t?\b',
    r'\bsp[i\*]+c\b',
    r'\bch[i\*]+nk\b',
    r'\bk[i\*]+ke\b',
    r'\btr[a\*]+nn[yi]\b',
    r'\b(how\s+to\s+)?(make|build|create|synthesize)\s+(a\s+)?(bomb|explosive|poison)\b',
    r'\b(how\s+to\s+)?kill\s+(my|a|your)self\b',
    r'\bsuicide\s+(method|instruction|guide|how)\b',
    r'\bchild\s*(porn|sex|nude|exploit)\b',
    r'\b(cp|csam)\b',
]]

# --- Local neural moderation model ----------------------------------------

_mod_pipe = None

def _load_mod_pipe():
    global _mod_pipe
    import torch
    device = 0 if torch.cuda.is_available() else -1
    for model_id in [
        "unitary/toxic-bert",
        "martin-ha/toxic-comment-model",
    ]:
        try:
            _mod_pipe = hf_pipeline("text-classification", model=model_id, device=device)
            print(f"Moderation model loaded: {model_id}")
            return
        except Exception as e:
            print(f"Moderation model {model_id} failed: {e}, trying next...")
    print("All moderation models failed, falling back to regex only.")

threading.Thread(target=_load_mod_pipe, daemon=True).start()

# --- Combined check -------------------------------------------------------

def _is_blocked(message: str) -> bool:
    # 1. Fast regex pre-filter
    if any(p.search(_normalize(message)) for p in _BLOCKED_REGEXES):
        return True
    # 2. Neural model (if loaded)
    if _mod_pipe is not None:
        try:
            result = _mod_pipe(message, truncation=True, max_length=512)[0]
            label = result["label"].lower()
            score = result["score"]
            # Only block if confidently harmful (threshold avoids false positives)
            is_harmful = label not in ("ok", "non-toxic", "non_toxic")
            return is_harmful and score > 0.85
        except Exception:
            pass
    return False

_REFUSAL_MESSAGE = "I'm sorry, I can't respond to that kind of request."


def respond(
    message,
    history,
    nfe,
    max_tokens,
    suffix_text,
    deterministic,
    progress=gr.Progress(),
):
    if _is_blocked(message):
        return _REFUSAL_MESSAGE

    if re.match(r'^\s*(hello|hi|hey|howdy|greetings)[!?.\s]*$', message.strip(), re.IGNORECASE):
        return "Hello! How can I help you? Type 'help' for the user guideline."

    if re.match(r'^\s*help\s*$', message.strip(), re.IGNORECASE):
        return """\
## MDM-Prime-v2 User Guide

**What is this model?**
MDM-Prime-v2 is a **masked discrete diffusion language model** pretrained on web text data — it has *not* been instruction-finetuned. \
It works best as a **text completion / few-shot prompting** tool, not a general Q&A assistant.

**Generation settings** (click the 'Generation settings' above the chat box):
| Setting | Description |
|---|---|
| **NFE** | Number of diffusion steps — higher = better quality, slower |
| **Max new tokens** | How many tokens to generate |
| **Suffix context** | Optional ending — model will infill between prompt and suffix |

**How to get good results (Examples):**
- For few-shot prompting, give 2–3 examples before your query ([SciQ](https://huggingface.co/datasets/allenai/sciq/)):
```
Question: What type of organism is commonly used in preparation of foods such as cheese and yogurt?
(A) viruses
(B) protozoa
(C) gymnosperms
(D) mesophilic organisms
Answer: D

Question: What kind of a reaction occurs when a substance reacts quickly with oxygen?
(A) nitrogen reaction
(B) invention reaction
(C) fluid Reaction
(D) combustion reaction
Answer:
```
- Use the **Suffix context** field to guide infilling between your prompt and a target ending.
```
A combustion reaction occurs when a substance reacts quickly with oxygen (O2). 
For example, in the Figure below , charcoal is combining with oxygen. Combustion is commonly called burning, and the substance that burns is usually referred to as fuel. 
The products of a complete combustion reaction include carbon dioxide (CO2) and water vapor (H2O). The reaction typically gives off heat and light as well.
```

- Adjust 'NFE' and 'Max new tokens' (e.g., 100) and provide a clear prefix and let the model complete it.

**Tips:**
- Start with NFE=10 for quick results; increase to 200+ for quality.
- This demo is for a quick check of the model's responses, not for general Q&A. We open the weights at [chen-hao-chao/mdm-prime-v2-slimpajama](https://huggingface.co/chen-hao-chao/mdm-prime-v2-slimpajama) — one can finetune it as a chatbot.
"""

    if TEST_MODE:
        for i in range(10):
            time.sleep(0.05)
            progress((i + 1) / 10, desc="[TEST] Simulating diffusion sampling...")
        generated = (
            f"[TEST MODE] Echo: {message}\n\n"
            f"Settings: NFE={nfe}, max_tokens={int(max_tokens)}"
        )
        if suffix_text.strip():
            generated += f"\nSuffix: {suffix_text[:50]}..."
        return generated

    if _model_cache["error"]:
        return f"Model loading failed: {_model_cache['error']}"

    # Wait for model to finish loading
    while _model_cache["result"] is None:
        progress(0, desc="Waiting for model to load...")
        time.sleep(1)

    torch, model, tokenizer, subtokenizer, device = _model_cache["result"]

    prefix_ids = tokenizer(
        message, return_tensors="pt", truncation=True, max_length=2048,
    ).input_ids.to(device)

    if suffix_text.strip():
        suffix_ids = tokenizer(
            suffix_text, return_tensors="pt", truncation=True, max_length=2048,
        ).input_ids.to(device)
    else:
        suffix_ids = torch.zeros((1, 0), dtype=prefix_ids.dtype, device=device)

    seed = 42 if deterministic else None
    set_deterministic(seed, torch)

    with torch.no_grad():
        outputs = sampling_mask_diff(
            torch, model, tokenizer, subtokenizer,
            prefix_ids, suffix_ids,
            ref_token_count=int(max_tokens),
            device=device,
            step_size=1 / nfe,
            seed=seed,
            progress=progress,
            corrector_steps=nfe//10,
        )

    input_len = prefix_ids.shape[1]
    suffix_len = suffix_ids.shape[1]
    end = -suffix_len if suffix_len > 0 else None
    return tokenizer.decode(outputs[0][input_len:end], skip_special_tokens=True)


with gr.Blocks(title="MDM-Prime-v2 Demo") as demo:
    gr.Markdown("# MDM-Prime-v2 Chatbot Demo")
    gr.Markdown("Masked Discrete Diffusion Language Model — text completion / infilling")

    if TEST_MODE:
        gr.Markdown("> **TEST MODE** — no model loaded, responses are simulated.")

    with gr.Accordion("Generation settings", open=False):
        nfe_input = gr.Slider(10, 300, value=10, step=10, label="NFE (diffusion steps)")
        max_tokens_input = gr.Slider(2, 512, value=2, step=1, label="Max new tokens")
        suffix_input = gr.Textbox(
            label="Suffix context (optional)",
            placeholder="If provided, the model will infill between your prompt and this suffix.",
            lines=3,
        )
        deterministic_input = gr.Checkbox(label="Deterministic (same prompt → same output)", value=True)

    gr.ChatInterface(
        fn=respond,
        chatbot=gr.Chatbot(height=700, value=[[None, "Hello! Type 'help' for the user guideline."]]),
        additional_inputs=[nfe_input, max_tokens_input, suffix_input, deterministic_input],
        undo_btn=None,
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=3000)
