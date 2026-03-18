import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
plt.rcParams['font.family'] = 'serif' 

# ---------- Shared Helpers ----------
def _quad_fit_logx(N, y):
    """Fit y ≈ a (log10 N)^2 + b (log10 N) + c and return a smooth curve."""
    N, y = np.asarray(N, dtype=float).reshape(-1), np.asarray(y, dtype=float).reshape(-1)
    x = np.log10(N)
    A = np.vstack([x**2, x, np.ones_like(x)]).T
    a, b, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return lambda Nq: a*(np.log10(Nq)**2) + b*np.log10(Nq) + c

# ---------- Integrated Data Configuration ----------
isoflop_configs = {
    'arm': {
        'budgets': ["3e18","6e18","1e19","3e19","6e19","1e20","3e20"],
        'params': {
            "3e18": [25e6, 49e6, 79e6, 106e6, 154e6, 201e6, 252e6, 315e6, 354e6, 413e6, 571e6],
            "6e18": [49e6, 79e6, 106e6, 154e6, 201e6, 252e6, 315e6, 354e6, 413e6, 571e6, 771e6],
            "1e19": [79e6, 106e6, 154e6, 201e6, 252e6, 315e6, 354e6, 413e6, 571e6, 771e6, 1107e6],
            "3e19": [106e6, 201e6, 252e6, 315e6, 413e6, 571e6, 771e6, 1107e6, 1529e6], 
            "6e19": [201e6, 315e6, 413e6, 571e6, 771e6, 1107e6, 1529e6, 2359e6], 
            "1e20": [315e6, 413e6, 571e6, 771e6, 1107e6, 1529e6, 2359e6, 3416e6],
            "3e20": [413e6, 571e6, 771e6, 1107e6, 1529e6, 2359e6, 3416e6]
        },
        'losses': {
            "3e18": [3.547, 3.458, 3.413, 3.380, 3.416, 3.421, 3.443, 3.467, 3.488, 3.518, 3.594],
            "6e18": [3.332, 3.284, 3.257, 3.252, 3.264, 3.265, 3.274, 3.282, 3.292, 3.364, 3.430],
            "1e19": [3.238, 3.177, 3.162, 3.162, 3.152, 3.170, 3.173, 3.175, 3.228, 3.255, 3.355],
            "3e19": [3.087, 3.020, 2.997, 2.992, 2.963, 2.979, 2.987, 3.061, 3.13], 
            "6e19": [2.954, 2.930, 2.906, 2.891, 2.894, 2.912, 2.959, 3.025], 
            "1e20": [2.859, 2.821, 2.795, 2.772, 2.790, 2.835, 2.891, 2.927],
            "3e20": [2.734, 2.676, 2.649, 2.619, 2.634, 2.669, 2.720]
        }
    },
    'mdm': {
        'budgets': ["3e18","6e18","1e19","3e19","6e19","1e20","3e20"],
        'params': {
            "3e18": [14e6, 25e6, 36e6, 49e6, 64e6, 79e6, 106e6, 154e6, 201e6, 252e6],
            "6e18": [25e6, 36e6, 49e6, 64e6, 79e6, 106e6, 154e6, 201e6, 252e6, 354e6, 413e6],
            "1e19": [36e6, 64e6, 79e6, 106e6, 154e6, 201e6, 252e6, 354e6, 413e6, 571e6],
            "3e19": [79e6, 106e6, 154e6, 201e6, 252e6, 354e6, 413e6, 571e6, 771e6, 1107e6],
            "6e19": [106e6, 201e6, 252e6, 354e6, 413e6, 571e6, 771e6, 1107e6, 1529e6],
            "1e20": [201e6, 252e6, 354e6, 413e6, 571e6, 771e6, 1107e6, 1529e6, 2359e6],
            "3e20": [252e6, 354e6, 413e6, 571e6, 771e6, 1107e6, 1529e6, 2359e6, 3416e6]
        },
        'losses': {
            "3e18": [4.144, 4.009, 3.993, 3.971, 3.964, 3.961, 3.997, 4.060, 4.120, 4.197],
            "6e18": [3.899, 3.846, 3.829, 3.780, 3.789, 3.805, 3.838, 3.889, 3.907, 4.026, 4.064],
            "1e19": [3.770, 3.697, 3.691, 3.677, 3.686, 3.719, 3.766, 3.842, 3.891, 3.939],
            "3e19": [3.514, 3.481, 3.478, 3.451, 3.468, 3.488, 3.498, 3.588, 3.664, 3.758],
            "6e19": [3.424, 3.408, 3.382, 3.398, 3.405, 3.443, 3.480, 3.536, 3.656],
            "1e20": [3.335, 3.299, 3.297, 3.289, 3.294, 3.327, 3.351, 3.409, 3.473],
            "3e20": [3.144, 3.110, 3.095, 3.085, 3.081, 3.098, 3.106, 3.150, 3.212]
        }
    },
    'prime': {
        'budgets': ["3e18","6e18","1e19","3e19","6e19","1e20","3e20"],
        'params': {
            "3e18": [14e6, 25e6, 36e6, 49e6, 64e6, 79e6, 106e6, 154e6, 201e6],
            "6e18": [25e6, 36e6, 49e6, 64e6, 79e6, 106e6, 154e6, 201e6, 252e6],
            "1e19": [36e6, 49e6, 64e6, 79e6, 106e6, 154e6, 201e6, 252e6, 354e6, 413e6],
            "3e19": [79e6, 106e6, 154e6, 201e6, 252e6, 354e6, 413e6, 571e6, 771e6],
            "6e19": [106e6, 154e6, 201e6, 252e6, 354e6, 413e6, 571e6, 771e6, 1107e6],
            "1e20": [201e6, 252e6, 354e6, 413e6, 571e6, 771e6, 1107e6, 1529e6],
            "3e20": [252e6, 354e6, 413e6, 571e6, 771e6, 1107e6, 1529e6, 2359e6]
        },
        'losses': {
            "3e18": [2.980, 2.889, 2.846, 2.830, 2.836, 2.826, 2.891, 2.988, 3.091],
            "6e18": [2.786, 2.715, 2.708, 2.703, 2.698, 2.735, 2.803, 2.862, 2.885],
            "1e19": [2.654, 2.613, 2.613, 2.601, 2.593, 2.606, 2.660, 2.708, 2.797, 2.847],
            "3e19": [2.446, 2.435, 2.412, 2.428, 2.429, 2.451, 2.465, 2.572, 2.620],
            "6e19": [2.349, 2.338, 2.345, 2.347, 2.333, 2.336, 2.391, 2.438, 2.525],
            "1e20": [2.261, 2.244, 2.231, 2.230, 2.249, 2.280, 2.338, 2.402],
            "3e20": [2.124, 2.092, 2.087, 2.076, 2.097, 2.137, 2.190, 2.252]
        }
    }
}

# ---------- Plotting Logic ----------
def plot_isoflop(mode, 
                 x_tick_vals=(1e7, 1e8, 3e8, 1e9, 3e9, 6e9),
                 x_tick_labels=("10M", "100M","300M","1B","3B","6B"),
                 ylim=(1.8, 4.7),
                 figsize=(8,6)):
    
    if mode not in isoflop_configs:
        print(f"Mode {mode} not found.")
        return

    config = isoflop_configs[mode]
    cmap = plt.get_cmap("viridis")
    colors = [cmap(t) for t in np.linspace(0.2, 0.95, len(config['budgets']))]

    fig, ax = plt.subplots(figsize=figsize)
    x_min, x_max = x_tick_vals[0]/1.5, x_tick_vals[-1]*1.3

    for label, color in zip(config['budgets'], colors):
        N = np.array(config['params'][label])
        L = np.array(config['losses'][label])
        fit_fn = _quad_fit_logx(N, L)

        # Draw smooth fit line
        log_lo, log_hi = np.log10(N.min()) - 0.08, np.log10(N.max()) + 0.08
        Nq = np.logspace(max(log_lo, np.log10(x_min)), min(log_hi, np.log10(x_max)), 300)
        ax.plot(Nq, fit_fn(Nq), "--", color=color, linewidth=2.5, alpha=0.8)
        
        # Plot data points
        ax.scatter(N, L, s=180, color=color, label=label, zorder=10)

    ax.set_xscale("log")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(ylim)
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
    ax.set_xticks(x_tick_vals)
    ax.set_xticklabels(x_tick_labels)
    ax.grid(True, which="both", linestyle="-", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"isoflop_{mode}.png", dpi=200)

if __name__ == "__main__":
    choice = input("Enter mode ('arm', 'mdm', or 'prime'): ").strip().lower()
    plot_isoflop(choice)