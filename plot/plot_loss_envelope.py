import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
plt.rcParams['font.family'] = 'serif' 

# ---------- Shared Smoothing Helpers ----------
def remove_outliers_rolling(x, y, window=20, sigma=3.0):
    y_series = pd.Series(y)
    rolling = y_series.rolling(window=window, center=True, min_periods=5)
    med = rolling.median()
    std = rolling.std()
    diff = np.abs(y_series - med)
    mask = (diff <= (sigma * std)).fillna(True).to_numpy()
    return x[mask], y[mask]

def _gaussian_weights(dlogx, sigma):
    return np.exp(-0.5 * (dlogx / sigma) ** 2)

def smooth_in_logx(x, y, sigma_log=0.002, min_pts=15):
    x, y = np.asarray(x), np.asarray(y)
    logx = np.log10(x)
    order = np.argsort(x)
    x, y, logx = x[order], y[order], logx[order]
    ys = np.empty_like(y, dtype=float)
    for i in range(len(x)):
        dlog = logx - logx[i]
        w = _gaussian_weights(dlog, sigma_log)
        mask = (~np.isnan(y)) & (w > 1e-6)
        if mask.sum() >= min_pts:
            ys[i] = np.average(y[mask], weights=w[mask])
        else:
            ys[i] = y[i]
    return x, ys

def parse_params(s):
    s = str(s).strip().upper()
    return float(s[:-1]) * (1e9 if s.endswith('B') else 1e6 if s.endswith('M') else 1.0)

# ---------- Integrated Configuration ----------
plotting_configs = {
    'arm': {
        'csv': 'assets/envelope_arm.csv',
        'sigma_log': 0.001,
        'models': {
            'arm_param_3426M_iter_18000': {'params': '3426M', 'final_flops': 3.880e+20, 'truncate_at': 3.0e+20},
            'arm_param_2359M_iter_21000': {'params': '2359M', 'final_flops': 3.117e+20, 'truncate_at': 3.0e+20},
            'arm_param_1529M_iter_40000': {'params': '1529M', 'final_flops': 3.848e+20, 'truncate_at': 3.0e+20},
            'arm_param_1107M_iter_87500': {'params': '1107M', 'final_flops': 3.048e+20, 'truncate_at': 3.0e+20},
            'arm_param_771M_iter_160000': {'params': '771M', 'final_flops': 3.879e+20, 'truncate_at': 3.0e+20},
            'arm_param_571M_iter_170000': {'params': '571M', 'final_flops': 3.053e+20, 'truncate_at': 3.0e+20},
            'arm_param_413M_iter_300000': {'params': '413M', 'final_flops': 3.897e+20, 'truncate_at': 6.0e+19},
            'arm_param_354M_iter_30000':  {'params': '354M', 'final_flops': 3.340e+19, 'truncate_at': 3.0e+19},
            'arm_param_315M_iter_110000': {'params': '315M', 'final_flops': 1.089e+20, 'truncate_at': 3.0e+19},
            'arm_param_252M_iter_40000':  {'params': '252M', 'final_flops': 3.167e+19, 'truncate_at': 3.0e+19},
            'arm_param_201M_iter_160000': {'params': '201M', 'final_flops': 1.013e+20, 'truncate_at': 3.0e+19},
            'arm_param_154M_iter_65000':  {'params': '154M', 'final_flops': 3.152e+19, 'truncate_at': 3.0e+19},
            'arm_param_106M_iter_180000': {'params': '106M', 'final_flops': 6.012e+19, 'truncate_at': 1.0e+19},
            'arm_param_79M_iter_140000':  {'params': '79M', 'final_flops': 1.675e+19, 'truncate_at': 1.0e+19},
            'arm_param_49M_iter_40000':   {'params': '49M', 'final_flops': 6.186e+18, 'truncate_at': 6.0e+18},
            'arm_param_25M_iter_80000':   {'params': '25M', 'final_flops': 6.335e+18, 'truncate_at': 2.0e+18},
        }
    },
    'mdm': {
        'csv': 'assets/envelope_mdm.csv',
        'sigma_log': 0.002,
        'models': {
            'mdm_param_3426M_iter_18000': {'params': '3426M', 'final_flops': 3.880e+20, 'truncate_at': 3.0e+20},
            'mdm_param_2359M_iter_21000': {'params': '2359M', 'final_flops': 3.117e+20, 'truncate_at': 3.0e+20},
            'mdm_param_1529M_iter_40000': {'params': '1529M', 'final_flops': 3.848e+20, 'truncate_at': 3.0e+20},
            'mdm_param_1107M_iter_87500': {'params': '1107M', 'final_flops': 3.048e+20, 'truncate_at': 3.0e+20},
            'mdm_param_771M_iter_160000': {'params': '771M', 'final_flops': 3.466e+20, 'truncate_at': 3.0e+20},
            'mdm_param_571M_iter_170000': {'params': '571M', 'final_flops': 3.053e+20, 'truncate_at': 3.0e+20},
            'mdm_param_413M_iter_300000': {'params': '413M', 'final_flops': 3.116e+20, 'truncate_at': 3.0e+20},
            'mdm_param_354M_iter_270000': {'params': '354M', 'final_flops': 3.006e+20, 'truncate_at': 3.0e+20},
            'mdm_param_252M_iter_380000': {'params': '252M', 'final_flops': 3.008e+20, 'truncate_at': 3.0e+20},
            'mdm_param_201M_iter_475000': {'params': '201M', 'final_flops': 1.704e+20, 'truncate_at': 1.0e+20},
            'mdm_param_106M_iter_300000': {'params': '106M', 'final_flops': 1.002e+20, 'truncate_at': 3.0e+19},
            'mdm_param_79M_iter_140000':  {'params': '79M', 'final_flops': 3.464e+19},
            'mdm_param_64M_iter_50000':   {'params': '64M', 'final_flops': 1.005e+19, 'truncate_at': 1.0e+19},
            'mdm_param_49M_iter_40000':   {'params': '49M', 'final_flops': 6.186e+18},
            'mdm_param_36M_iter_90000':   {'params': '36M', 'final_flops': 1.015e+19, 'truncate_at': 1.0e+19},
            'mdm_param_25M_iter_80000':   {'params': '25M', 'final_flops': 6.335e+18, 'truncate_at': 6.0e+18},
            'mdm_param_14M_iter_135000':  {'params': '14M', 'final_flops': 6.139e+18, 'truncate_at': 3.0e+18},
        }
    },
    'prime': {
        'csv': 'assets/envelope_prime.csv',
        'sigma_log': 0.002,
        'models': {
            'prime_param_3426M_iter_14000': {'params': '3426M', 'final_flops': 3.017e+20, 'truncate_at': 3.0e+20},
            'prime_param_2359M_iter_21000': {'params': '2359M', 'final_flops': 3.117e+20, 'truncate_at': 3.0e+20},
            'prime_param_1529M_iter_32000': {'params': '1529M', 'final_flops': 3.078e+20, 'truncate_at': 3.0e+20},
            'prime_param_1107M_iter_87500': {'params': '1107M', 'final_flops': 3.048e+20, 'truncate_at': 3.0e+20},
            'prime_param_771M_iter_125000': {'params': '771M', 'final_flops': 3.031e+20, 'truncate_at': 3.0e+20},
            'prime_param_571M_iter_170000': {'params': '571M', 'final_flops': 3.053e+20, 'truncate_at': 3.0e+20}, 
            'prime_param_413M_iter_240000': {'params': '413M', 'final_flops': 3.085e+20, 'truncate_at': 3.0e+20},
            'prime_param_354M_iter_270000': {'params': '354M', 'final_flops': 3.006e+20, 'truncate_at': 3.0e+20},
            'prime_param_252M_iter_380000': {'params': '252M', 'final_flops': 3.008e+20, 'truncate_at': 3.0e+20},
            'prime_param_154M_iter_125000': {'params': '154M', 'final_flops': 6.061e+19},
            'prime_param_106M_iter_180000': {'params': '106M', 'final_flops': 6.012e+19},
            'prime_param_79M_iter_125000':  {'params': '79M', 'final_flops': 3.092e+19, 'truncate_at': 3.0e+19},
            'prime_param_64M_iter_50000':   {'params': '64M', 'final_flops': 1.005e+19, 'truncate_at': 1.0e+19},
            'prime_param_49M_iter_70000':   {'params': '49M', 'final_flops': 1.082e+19, 'truncate_at': 1.0e+19},
            'prime_param_36M_iter_90000':   {'params': '36M', 'final_flops': 1.015e+19, 'truncate_at': 1.0e+19},
            'prime_param_25M_iter_80000':   {'params': '25M', 'final_flops': 6.335e+18, 'truncate_at': 6.0e+18},
            'prime_param_14M_iter_70000':   {'params': '14M', 'final_flops': 3.182e+18, 'truncate_at': 3.0e+18},
        }
    }
}

# ---------- Plotting Logic ----------
def plot_mode(mode):
    if mode not in plotting_configs:
        print(f"Error: mode '{mode}' not found. Choose from 'arm', 'mdm', or 'prime'.")
        return

    config = plotting_configs[mode]
    df = pd.read_csv(config['csv'])
    models = config['models']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Setup color mapping
    param_vals = np.array([parse_params(m['params']) for m in models.values()])
    cmap = plt.get_cmap("plasma")
    norm = mcolors.LogNorm(vmin=param_vals.min(), vmax=param_vals.max())
    
    # Sort models by size for better layering
    ordered_models = sorted(models.items(), key=lambda kv: parse_params(kv[1]['params']))

    for rank, (model_name, model_info) in enumerate(ordered_models, start=1):
        loss_col = f"{model_name} - lm loss"
        if loss_col not in df.columns:
            continue
            
        model_data = df[['Step', loss_col]].dropna().sort_values('Step')
        steps, losses = model_data['Step'].to_numpy(), model_data[loss_col].to_numpy()

        # Processing
        steps, losses = remove_outliers_rolling(steps, losses, window=25, sigma=3.0)
        flops = (steps / steps.max()) * model_info['final_flops']

        limit = model_info.get('truncate_at') 
        if limit is not None:
            mask = flops <= limit
            flops, losses = flops[mask], losses[mask]

        color = cmap(norm(parse_params(model_info['params'])))
        flops, losses = smooth_in_logx(flops, losses, sigma_log=config['sigma_log'])

        ax.plot(flops, losses, color=color, linewidth=2, alpha=0.9, zorder=rank+5,
                label=f"{model_info['params']}")

    ax.set_xscale('log')
    ax.set_xlim(1e17, 3e20)
    ax.set_ylim(1.9, 5.3)
    ax.tick_params(axis='both', which='major', labelsize=18)
    if mode == 'arm':
        title = "ARM"
    elif mode == 'mdm':
        title = "MDM"
    else:
        title = "MDM-Prime-v2"  
    ax.set_title(title, fontsize=22, fontweight='bold')
    ax.set_xlabel('FLOPs', fontsize=22)
    ax.set_ylabel('Loss (NLL)', fontsize=22)
    ax.grid(True, alpha=0.8, linestyle='-', which='both')
    plt.tight_layout()
    plt.savefig(f'envelope_{mode}.png', dpi=200)
    print(f"Saved figure: envelope_{mode}.png")

# ---------- Main Execution ----------
if __name__ == "__main__":
    choice = input("Enter model type ('arm', 'mdm', or 'prime'): ").strip().lower()
    plot_mode(choice)