import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'lit_gpt')))

import torch
import matplotlib.pyplot as plt
import numpy as np
from lit_gpt.config import Config
from lit_gpt.diffmodel import TransEncoder
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
plt.rcParams['font.family'] = 'serif'


# --- CONFIGURATION ---
# Specify which layers to plot: (start, end) inclusive, or None for all layers
# Example: (3, 12) will plot layers 3 through 12
LAYER_RANGE = None
MODEL_SMDM = "nieshen/SMDM"
MODEL_PRIME = "chen-hao-chao/mdm-prime-v2-slimpajama"
CACHE_DIR_HF = None
MODEL_SIZE = 1028

def get_singular_values(model_name_or_mode, layer_range=None):
    """
    Extract singular values from attention weight matrices.

    Args:
        model_name_or_mode: Either a HuggingFace model name (e.g., "nieshen/SMDM")
                           or 'local' to use local checkpoint paths
        layer_range: Optional tuple (start, end) to specify which layers to process
    """
    print(f"--- Processing Model: {model_name_or_mode} ---")

    N_LAYER = 20
    N_HEAD = 14
    config_name = f'Diff_LLaMA_{MODEL_SIZE}M'

    # Determine which layers to process
    if layer_range is not None:
        layer_start, layer_end = layer_range
        layers_to_process = range(layer_start, layer_end + 1)
        print(f"Processing layers {layer_start} to {layer_end}")
    else:
        layers_to_process = range(N_LAYER)
        print(f"Processing all {N_LAYER} layers")

    # Setup model configuration and checkpoint path
    try:
        config = Config.from_name(config_name)
    except Exception:
        config = Config(block_size=2048, vocab_size=32000, n_embd=1024)

    config.n_layer = N_LAYER
    config.n_head = N_HEAD

    # Determine model type and load checkpoint
    if model_name_or_mode == "nieshen/SMDM":
        print("Loading SMDM model from HuggingFace...")
        init_kwargs = dict(target_length=1, base=32000, sum_emb=True)
        model = TransEncoder(config, **init_kwargs)

        # Download from HuggingFace
        ckpt_path = hf_hub_download(
            repo_id="nieshen/SMDM",
            filename="mdm_safetensors/mdm-1028M-3300e18-rsl-0.01-bs-1024.safetensors",
            cache_dir=CACHE_DIR_HF
        )
        print(f"Checkpoint downloaded to: {ckpt_path}")
        state_dict = load_file(ckpt_path)
        model.load_state_dict(state_dict)

    elif model_name_or_mode == "chen-hao-chao/mdm-prime-v2-slimpajama":
        print("Loading MDM-Prime-v2 model from HuggingFace...")
        init_kwargs = dict(target_length=15, base=2, sum_emb=True)
        model = TransEncoder(config, **init_kwargs)

        # Download from HuggingFace
        ckpt_path = hf_hub_download(
            repo_id="chen-hao-chao/mdm-prime-v2-slimpajama",
            filename="mdm-prime-v2-3300flops-weight-only.pth",
            cache_dir=CACHE_DIR_HF
        )
        print(f"Checkpoint downloaded to: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(checkpoint["model"])

    else:
        raise ValueError(f"Unknown model: {model_name_or_mode}")

    model.eval()

    # Extract singular values
    sv_data = {}
    print(f"Extracting singular values from attention weights...")
    for i, block in enumerate(model.transformer.h):
        if i in layers_to_process:
            weight_matrix = block.attn.attn.weight.detach().float().cpu()
            S = torch.linalg.svdvals(weight_matrix)
            sv_data[i] = S.numpy()

    del model
    return sv_data

def visualize_combined_grid_log():
    # 1. Determine layers to plot
    if LAYER_RANGE is not None:
        layer_start, layer_end = LAYER_RANGE
        layers_to_plot = list(range(layer_start, layer_end + 1))
        print(f"Plotting layers {layer_start} to {layer_end}")
    else:
        N_LAYER = 20
        layers_to_plot = list(range(N_LAYER))
        print(f"Plotting all {N_LAYER} layers")

    # 2. Get Data
    print("\n" + "="*50)
    data_prime = get_singular_values(MODEL_PRIME, LAYER_RANGE)
    print("\n" + "="*50)
    data_mdm = get_singular_values(MODEL_SMDM, LAYER_RANGE)
    print("="*50 + "\n")

    # 3. Setup Plot - dynamically calculate grid layout
    print("Generating combined LOG-SCALE plot...")
    num_layers = len(layers_to_plot)

    # Calculate optimal grid layout (aim for roughly square grid)
    cols = min(5, num_layers)  # Max 5 columns
    rows = (num_layers + cols - 1) // cols  # Ceiling division

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.8, rows * 3), sharex=True, sharey=True)
    
    # --- KEY CHANGE: log=True ---
    # We also increase bins slightly to see finer detail in the log view
    hist_kwargs = dict(bins=60, density=True, alpha=0.5, histtype='stepfilled', log=True)

    # Handle single subplot case (axes is not an array)
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1 or cols == 1:
        axes = axes.reshape(rows, cols)

    for plot_idx, layer_idx in enumerate(layers_to_plot):
        r, c = divmod(plot_idx, cols)
        ax = axes[r, c]

        # Plot MDM (Orange)
        if layer_idx in data_mdm:
            s_mdm = data_mdm[layer_idx]
            rank_mdm = (s_mdm**2).sum() / (s_mdm.max()**2)
            ax.hist(s_mdm, color='tab:orange', label=f'MDM (R={rank_mdm:.1f})', **hist_kwargs)

        # Plot Prime (Blue)
        if layer_idx in data_prime:
            s_prime = data_prime[layer_idx]
            rank_prime = (s_prime**2).sum() / (s_prime.max()**2)
            ax.hist(s_prime, color='tab:blue', label=f'Prime (R={rank_prime:.1f})', **hist_kwargs)

        ax.set_title(f"Layer {layer_idx}", fontsize=14, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=16)

    # Hide any unused subplots
    total_cells = rows * cols
    for plot_idx in range(num_layers, total_cells):
        r, c = divmod(plot_idx, cols)
        axes[r, c].axis('off')

    # Formatting
    fig.text(0.5, 0.04, 'Singular Value Magnitude', ha='center', fontsize=26)
    fig.text(0.04, 0.5, 'Frequency', va='center', rotation='vertical', fontsize=26)

    # Add figure-level legend at the top center
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=2, fontsize=20)

    plt.tight_layout(rect=[0.05, 0.05, 1, 0.90])
    plt.savefig(f'assets/spectra.png', dpi=150)

if __name__ == "__main__":
    visualize_combined_grid_log()