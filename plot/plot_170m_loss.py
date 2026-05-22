import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.family'] = 'serif' 

def moving_average_with_std(y, window=5):
    """Moving average smoothing with standard deviation"""
    y_padded = np.pad(y, (window//2, window//2), mode='edge')
    y_smooth = np.convolve(y_padded, np.ones(window)/window, mode='valid')

    # Calculate rolling standard deviation
    y_std = np.array([np.std(y[max(0, i-window//2):min(len(y), i+window//2+1)])
                      for i in range(len(y))])
    return y_smooth, y_std

# Read the CSV file
df = pd.read_csv('assets/170m_loss.csv')
# Include step 10, then 2000-step intervals
arc_steps = [10] + list(range(2000, 60001, 2000))
df = df[df['trainer/global_step'].isin(arc_steps)]

# Create the plot
plt.figure(figsize=(6, 6))

# Get the steps column
steps = df['trainer/global_step']

# Plot each model's loss
models = {
    'prime-170M-l15': 'prime-170M-l15 - metric/train_loss',
    'prime-170M-l3': 'prime-170M-l3 - metric/train_loss',
    'prime-170M-l1': 'prime-170M-l1 - metric/train_loss',
    'arm-170M': 'arm-170M - metric/train_loss',
}

window = 3
for label, column in models.items():
    # Filter out NaN values
    mask = df[column].notna()
    x_data = np.array(list(steps[mask]))
    y_data = np.array(list(df[column][mask]))

    # Separate step 10 from the rest
    step_10_mask = x_data == 10
    rest_mask = x_data >= 2000

    x_step_10 = x_data[step_10_mask]
    y_step_10 = y_data[step_10_mask]

    x_rest = x_data[rest_mask]
    y_rest = y_data[rest_mask]

    # Apply smoothing only to data from step 2000 onwards
    y_smooth, y_std = moving_average_with_std(y_rest, window=window)

    # Combine step 10 (unsmoothed) with smoothed data
    x_vals = np.concatenate([x_step_10, x_rest])
    y_vals_smooth = np.concatenate([y_step_10, y_smooth])
    y_vals_std = np.concatenate([np.zeros_like(y_step_10), y_std])  # No variance at step 10

    # Plot the main line
    line = plt.plot(x_vals, y_vals_smooth, marker='o', label=label, linewidth=2, markersize=4)

    # Add variance region (shaded band)
    plt.fill_between(x_vals,
                     y_vals_smooth - y_vals_std,
                     y_vals_smooth + y_vals_std,
                     alpha=0.1,
                     color=line[0].get_color())

# Customize the plot
plt.xlabel('Steps', fontsize=16)
plt.ylabel('Loss', fontsize=16)
plt.title('Training Loss vs Steps', fontsize=14, fontweight='bold')
plt.legend(fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.xlim(0, 57500)
plt.ylim(2, 7)
plt.grid(True, alpha=0.8)
plt.tight_layout()

# Save the plot
plt.savefig('assets/plot_170M_loss.png', dpi=300, bbox_inches='tight')
