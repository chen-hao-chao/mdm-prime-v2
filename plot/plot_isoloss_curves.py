import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.rcParams['font.family'] = 'serif' 

# --- 1. Core Fitting Function ---
def chinchilla_lbfgs_huber(data_dict, delta=1e-3):
    """
    Fits: L(N, D) = E + A/N^alpha + B/D^beta
    where D = Budget / (6 * N)
    """
    all_N, all_D, all_L_obs = [], [], []
    
    # data_dict here is the specific architecture's sub-dict
    budgets = data_dict['budgets']
    for b_str in budgets:
        C = float(b_str)
        n_arr = np.array(data_dict['params'][b_str])
        l_arr = np.array(data_dict['losses'][b_str])
        
        d_arr = C / (6 * n_arr)
        all_N.append(n_arr)
        all_D.append(d_arr)
        all_L_obs.append(l_arr)
        
    N = np.concatenate(all_N)
    D = np.concatenate(all_D)
    L_obs = np.concatenate(all_L_obs)
    log_L_obs = np.log(L_obs)

    def objective_function(params):
        E, A, B, alpha, beta = params
        epsilon = 1e-9
        L_pred = E + (A / (N**alpha + epsilon)) + (B / (D**beta + epsilon))
        log_L_pred = np.log(np.maximum(L_pred, epsilon))
        residuals = log_L_obs - log_L_pred
        abs_r = np.abs(residuals)
        # Huber Loss
        loss = np.where(abs_r <= delta, 0.5 * residuals**2, delta * (abs_r - 0.5 * delta))
        return np.sum(loss)

    # Initial guess and bounds
    x0 = [1.69, 400.0, 400.0, 0.34, 0.34]
    bounds = [(0.1, None), (1.0, None), (1.0, None), (0.01, 1.0), (0.01, 1.0)]
    res = minimize(objective_function, x0, method='L-BFGS-B', bounds=bounds)
    return res.x if res.success else None

# --- 2. Real Dataset ---
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

# --- 3. Plotting Logic ---
def plot_isoflop(mode):
    if mode not in isoflop_configs:
        print(f"Error: {mode} not in dataset.")
        return
    
    config = isoflop_configs[mode]
    params_fit = chinchilla_lbfgs_huber(config)
    
    if params_fit is None:
        print("Optimization failed.")
        return
        
    E, A, B, alpha, beta = params_fit
    
    # Projection
    TARGET_FLOP = 2.887e20
    scaling_a = beta / (alpha + beta)
    G = ((alpha * A) / (beta * B))**(1/(alpha + beta))
    N_star = G * ((TARGET_FLOP/6) ** scaling_a)
    
    # Visualization
    fig, ax = plt.subplots(figsize=(8, 6))
    C_grid = np.logspace(18, 22, 100)
    N_grid = np.logspace(7, 10, 100)
    X_N, Y_C = np.meshgrid(N_grid, C_grid)
    Z_L = E + (A / X_N**alpha) + (B / (Y_C/(6*X_N))**beta)
    
    ax.contour(Y_C, X_N, Z_L, levels=60, cmap='RdBu', alpha=0.9, linewidths=2)
    ax.plot(C_grid, G * ((C_grid/6)**scaling_a), color='#377eb8', lw=3.5, label='Efficient Frontier')
    ax.scatter(TARGET_FLOP, N_star, s=200, marker='o', color='#FF0000', 
                edgecolor='black', linewidth=1, zorder=10, 
                label=f"Target ({str(TARGET_FLOP)} FLOPs)")
    ax.scatter(TARGET_FLOP, 92e6, s=250, marker='^', color='#FF0000',  #92e6
                edgecolor='black', linewidth=1, zorder=10, 
                label=f"Baseline ({str(TARGET_FLOP)} FLOPs)")
    ax.axvline(x=TARGET_FLOP, color='#FF0000', linestyle='--', linewidth=1.5, alpha=0.8, zorder=2)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.tick_params(axis='both', which='major', labelsize=18)
    if mode == 'arm':
        title = "ARM"
    elif mode == 'mdm':
        title = "MDM"
    else:
        title = "MDM-Prime-v2"  
    ax.set_title(title, fontsize=22, fontweight='bold')
    ax.set_xlabel('FLOPs', fontsize=22)
    ax.set_ylabel('Parameters', fontsize=22)
    legend_elements = [
        Line2D([0], [0], color='#377eb8', lw=2.5, label='Efficient frontier'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', framealpha=0.9, fontsize=10)
    plt.tight_layout()
    plt.savefig(f'isoloss_{mode}.png')

# --- 4. Main Loop ---
if __name__ == "__main__":
    mode = input("Enter mode ('arm', 'mdm', or 'prime'): ").strip().lower()
    plot_isoflop(mode)