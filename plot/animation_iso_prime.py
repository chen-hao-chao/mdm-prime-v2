import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.ticker import LogLocator

# 1. SETUP: Data Generation & Fitting
# ==========================================
def _quad_fit_logx(N, y):
    """Fit y ≈ a (log10 N)^2 + b (log10 N) + c."""
    x = np.log10(N)
    A = np.vstack([x**2, x, np.ones_like(x)]).T
    a, b, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return lambda Nq: a*(np.log10(Nq)**2) + b*np.log10(Nq) + c

def make_isoflop_data():
    budgets = ["3e18","6e18","1e19","3e19","6e19","1e20","3e20"]
    
    # Raw Data
    budget_parameters = {
        "3e18": np.array([14e6, 25e6, 36e6, 49e6, 64e6, 79e6, 106e6, 154e6, 201e6], dtype=float), # (201)
        "6e18": np.array([25e6, 36e6, 49e6, 64e6, 79e6, 106e6, 154e6, 201e6, 252e6], dtype=float), # (201)
        "1e19": np.array([36e6, 49e6, 64e6, 79e6, 106e6, 154e6, 201e6, 252e6, 354e6, 413e6], dtype=float), # (201)
        "3e19": np.array([79e6, 106e6, 154e6, 201e6, 252e6, 354e6, 413e6, 571e6, 771e6], dtype=float), # (201)
        "6e19": np.array([106e6, 154e6, 201e6, 252e6, 354e6, 413e6, 571e6, 771e6, 1107e6,], dtype=float), 
        "1e20": np.array([201e6, 252e6, 354e6, 413e6, 571e6, 771e6, 1107e6, 1529e6], dtype=float), # (413)
        "3e20": np.array([252e6, 354e6, 413e6, 571e6, 771e6, 1107e6, 1529e6, 2359e6], dtype=float) # (413)
    }
    budget_parameters_loss = {
        "3e18": np.array([2.980, 2.889, 2.846, 2.830, 2.836, 2.826, 2.891, 2.988, 3.091], dtype=float),
        "6e18": np.array([2.786, 2.715, 2.708, 2.703, 2.698, 2.735, 2.803, 2.862, 2.885], dtype=float),
        "1e19": np.array([2.654, 2.613, 2.613, 2.601, 2.593, 2.606, 2.660, 2.708, 2.797, 2.847], dtype=float),
        "3e19": np.array([2.446, 2.435, 2.412, 2.428, 2.429, 2.451, 2.465, 2.572, 2.620], dtype=float), 
        "6e19": np.array([2.349, 2.338, 2.345, 2.347, 2.333, 2.336, 2.391, 2.438, 2.525], dtype=float), 
        "1e20": np.array([2.261, 2.244, 2.231, 2.230, 2.249, 2.280, 2.338, 2.402], dtype=float),
        "3e20": np.array([2.124, 2.092, 2.087, 2.076, 2.097, 2.137, 2.190, 2.252], dtype=float)
    }
    
    data = {}
    for lab in budgets:
        data[lab] = (budget_parameters[lab], budget_parameters_loss[lab])
    return data

# 2. PRE-COMPUTATION
# ==========================================
raw_data = make_isoflop_data()
keys = list(raw_data.keys()) 

# Color Ramp
cmap = plt.get_cmap("viridis")
t0, t1 = 0.20, 0.95
colors = [cmap(t) for t in np.linspace(t0, t1, len(keys))]

# 3. STATIC PLOTTING SETUP
# ==========================================
plt.rcParams['font.family'] = 'serif' 

fig, ax = plt.subplots(figsize=(10, 7.5), dpi=200)

# Axes Limits
x_tick_vals = (1e7, 1e8, 3e8, 1e9, 3e9, 6e9)
x_min, x_max = float(x_tick_vals[0])/1.5, float(x_tick_vals[-1])*1.3
ax.set_xlim(x_min, x_max)
ax.set_ylim(1.8, 4.7)
ax.set_xscale("log")

# Grid & Ticks
ax.grid(True, which="both", linestyle="-", linewidth=0.8, alpha=0.8)
ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 3.0)))
ax.set_xticks(x_tick_vals)
ax.set_xticklabels(["10M", "100M","300M","1B","3B","6B"], fontsize=20)
ax.tick_params(axis='y', labelsize=20)

# Labels
ax.set_xlabel("Parameters", fontsize=24)
ax.set_ylabel("Loss", fontsize=24)
ax.set_title("MDM-Prime-v2", fontsize=24, fontweight='bold')

# --- INITIALIZE ALL CURVES (HIDDEN) ---
anim_objects = []

for label, color in zip(keys, colors):
    N, L = raw_data[label]
    budget_val = float(label)
    
    fit_fn = _quad_fit_logx(N, L)
    
    nmin, nmax = N.min(), N.max()
    fit_extend_frac = (0.08, 0.08)
    log_lo = np.log10(nmin) - fit_extend_frac[0]*(np.log10(nmax) - np.log10(nmin) + 1e-12)
    log_hi = np.log10(nmax) + fit_extend_frac[1]*(np.log10(nmax) - np.log10(nmin) + 1e-12)
    N_smooth = np.logspace(log_lo, log_hi, 300)
    L_smooth = fit_fn(N_smooth)
    
    # Initialize with alpha=0 (Invisible)
    line, = ax.plot(N_smooth, L_smooth, linestyle="--", linewidth=2.9, color=color, alpha=0)
    scat = ax.scatter(N, L, s=250, color=color, edgecolor="none", zorder=10, alpha=0)
    
    anim_objects.append({
        'budget': budget_val,
        'line': line,
        'scat': scat
    })

progress_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=24, 
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# 4. ANIMATION LOGIC (UPDATED)
# ==========================================
min_flop = 1e17
max_flop = 3e20 
total_frames = 100
fps_val = 20

# A. Generate the standard log-space frames
frames = np.logspace(np.log10(min_flop), np.log10(max_flop), total_frames)

# B. Append the final frame multiple times for the End Pause
pause_frames = [max_flop] * (2 * fps_val)
frames = np.concatenate([frames, pause_frames])

def update(current_budget):
    progress_text.set_text(f'Current FLOPs: {current_budget:.1e}')
    artists = [progress_text]
    
    for obj in anim_objects:
        # --- NEW FADE LOGIC ---
        # Instead of fading AFTER the budget is reached, we fade AS WE APPROACH it.
        # We start the fade 0.5 log-units BEFORE the curve's budget.
        # This ensures that when current_budget == obj['budget'], alpha is exactly 1.0.
        
        log_curr = np.log10(current_budget)
        log_target = np.log10(obj['budget'])
        
        # Determine fade window (starts 0.5 before target, ends at target)
        fade_start = log_target - 0.5
        fade_end = log_target
        
        if log_curr <= fade_start:
            new_alpha = 0.0
        elif log_curr >= fade_end:
            new_alpha = 1.0
        else:
            # Linear interpolation between 0 and 1 within the window
            new_alpha = (log_curr - fade_start) / (fade_end - fade_start)
        
        obj['line'].set_alpha(new_alpha)
        obj['scat'].set_alpha(new_alpha)
            
        artists.append(obj['line'])
        artists.append(obj['scat'])
        
    return artists

# 5. RENDER
# ==========================================
print("Animating...")
anim = animation.FuncAnimation(
    fig, 
    update, 
    frames=frames, 
    interval=1000/fps_val, 
    blit=True
)

output_file = 'animation_iso_prime.gif'
anim.save(output_file, writer='pillow', fps=fps_val)
print(f"Done! Saved to {output_file}")
plt.close()