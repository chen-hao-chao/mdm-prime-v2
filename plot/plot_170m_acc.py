import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.family'] = 'serif' 

# Read the CSV file
df = pd.read_csv('assets/170m_acc.csv')

# Filter data to only include steps up to 50K
df = df[df['trainer/global_step'] <= 60000]

# Create the plot
plt.figure(figsize=(6, 6))

# Get the steps column
steps = df['trainer/global_step']

# Plot each model's accuracy
models = {
    'prime-170M-l15': 'prime-170M-l15 - eval/arc_easy_acc_norm',
    'prime-170M-l3': 'prime-170M-l3 - eval/arc_easy_acc_norm',
    'prime-170M-l1': 'prime-170M-l1 - eval/arc_easy_acc_norm',
    'arm-170M': 'arm-170M - eval/arc_easy_acc_norm',
}

def moving_average_with_std(y, window=5):
    """Moving average smoothing with standard deviation"""
    y_padded = np.pad(y, (window//2, window//2), mode='edge')
    y_smooth = np.convolve(y_padded, np.ones(window)/window, mode='valid')

    # Calculate rolling standard deviation
    y_std = np.array([np.std(y[max(0, i-window//2):min(len(y), i+window//2+1)])
                      for i in range(len(y))])
    return y_smooth, y_std

for label, column in models.items():
    # Filter out NaN values
    mask = df[column].notna()
    # Get data without starting point first
    x_data = np.array(list(steps[mask]))
    y_data = np.array(list(df[column][mask]))

    # Apply moving average smoothing with standard deviation
    window = 7  # Adjust this for more or less smoothing
    y_smooth, y_std = moving_average_with_std(y_data, window=window)

    x_vals = x_data
    y_vals_smooth = y_smooth
    y_vals_std = y_std 
    y_vals_orig = y_data

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
plt.ylabel('Accuracy', fontsize=16)
plt.title('ARC Easy Accuracy vs Training Steps', fontsize=14, fontweight='bold')
plt.legend(fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.xlim(0, 57500)
plt.ylim(0.20, 0.42)
plt.grid(True, alpha=0.8)
plt.tight_layout()

# Save the plot
plt.savefig('assets/plot_170M_acc.png', dpi=300, bbox_inches='tight')
