import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors

plt.rcParams['font.family'] = 'serif' 

# 1. SETUP: Load Data and Define Config
# ==========================================
df = pd.read_csv('envelope_prime.csv')

models = {
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


def parse_params(p_str):
    if 'M' in p_str: return float(p_str.replace('M','')) * 1e6
    if 'B' in p_str: return float(p_str.replace('B','')) * 1e9
    return float(p_str)

# ---------- Smoothing Helpers ----------
def remove_outliers_rolling(x, y, window=20, sigma=3.0):
    y_series = pd.Series(y)
    rolling = y_series.rolling(window=window, center=True, min_periods=5)
    med = rolling.median()
    std = rolling.std()
    diff = np.abs(y_series - med)
    mask = (diff <= (sigma * std)).fillna(True).to_numpy()
    return x[mask], y[mask]

# 2. PRE-COMPUTATION
# ==========================================
processed_curves = []

# Setup Colormap
all_params = [parse_params(m['params']) for m in models.values()]
norm = mcolors.LogNorm(vmin=min(all_params), vmax=max(all_params))
cmap = plt.cm.plasma

print("Pre-computing curves...")
suffix = " - lm loss" # Suffix identified in CSV

for key, model_info in models.items():
    # Attempt to find the column (try with suffix first, then raw key)
    col_name = key + suffix
    if col_name not in df.columns:
        if key in df.columns:
            col_name = key
        else:
            # Flexible search
            found = False
            for c in df.columns:
                if key in c:
                    col_name = c
                    found = True
                    break
            if not found:
                print(f"Skipping {key}: Column not found.")
                continue

    # Extract data
    temp_df = df[['Step', col_name]].dropna()
    if temp_df.empty: continue
    
    steps = temp_df['Step'].values
    losses = temp_df[col_name].values
    
    # Calculate FLOPs
    max_step = steps.max()
    flops = (steps / max_step) * model_info['final_flops']

    # Apply Truncation
    limit = model_info.get('truncate_at')
    if limit is not None:
        mask = flops <= limit
        flops = flops[mask]
        losses = losses[mask]

    # Smoothing
    if len(flops) > 20:
        flops, losses = remove_outliers_rolling(flops, losses)

    # Determine Color
    param_val = parse_params(model_info['params'])
    color = cmap(norm(param_val))

    # Store for Animation
    processed_curves.append({
        'x': flops,
        'y': losses,
        'label': f"{model_info['params']}",
        'color': color,
        'params': param_val 
    })

# Sort curves so larger models are drawn on top (z-order)
processed_curves.sort(key=lambda k: k['params'])

# 3. ANIMATION SETUP
# ==========================================
fig, ax = plt.subplots(figsize=(10, 7.5), dpi=200)

# Static Chart Settings
ax.set_xscale('log')
ax.set_xlim(1e17, 3e20) 
ax.set_ylim(1.9, 5.3)   
ax.grid(True, alpha=0.8, which='both')
ax.set_xlabel('FLOPs', fontsize=24)
ax.set_ylabel('Loss', fontsize=24)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.set_title('MDM-Prime-v2', fontsize=24, fontweight='bold')

# Initialize lines AND dots
lines = []
dots = []
for curve in processed_curves:
    # 1. Plot the line
    line, = ax.plot([], [], color=curve['color'], lw=1.5, alpha=0.9)
    lines.append(line)
    
    # 2. Plot the dot (markersize sets the size)
    dot, = ax.plot([], [], 'o', color=curve['color'], markersize=8)
    dots.append(dot)

# Add a text label for current progress
progress_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=24, 
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# 4. DEFINE ANIMATION FRAMES
# ==========================================
min_flop_view = 1e17
max_flop_view = 3e20

# CHANGE 1: Update frame count and FPS
total_frames = 100
fps_val = 20

# Logarithmically spaced frames
frame_limits = np.logspace(np.log10(min_flop_view), np.log10(max_flop_view), total_frames)

# CHANGE 2: Append the final frame multiple times for the End Pause
# We want a 2-second pause at the end (2 seconds * 20 fps = 40 frames)
pause_seconds = 2
pause_frames = [max_flop_view] * (pause_seconds * fps_val)

# Concatenate standard frames with pause frames
frame_limits = np.concatenate([frame_limits, pause_frames])

def update(current_limit):
    # Update title/text
    progress_text.set_text(f'Current FLOPs: {current_limit:.1e}')
    
    # Update every curve
    for line, dot, data in zip(lines, dots, processed_curves):
        # Only show data points that are to the LEFT of the current limit
        mask = data['x'] <= current_limit
        
        if np.any(mask):
            x_val = data['x'][mask]
            y_val = data['y'][mask]
            
            # Update line
            line.set_data(x_val, y_val)
            
            # Update dot to be at the very last point
            dot.set_data([x_val[-1]], [y_val[-1]])
        else:
            line.set_data([], []) 
            dot.set_data([], [])
            
    return lines + dots + [progress_text]

# 5. RENDER AND SAVE
# ==========================================
print("Animating...")
anim = animation.FuncAnimation(
    fig, 
    update, 
    frames=frame_limits, 
    interval=1000/fps_val, # Use calculated interval
    blit=True
)

output_file = 'animation_prime.gif'
anim.save(output_file, writer='pillow', fps=fps_val) # Use new FPS variable
print(f"Done! Saved to {output_file}")
plt.close()