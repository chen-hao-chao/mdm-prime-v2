import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.animation import FuncAnimation

ACTIVE_FRAME = 160
PAUSE_FRAME = 40

def create_table_diffusion_gif(output_name="table_diffusion.gif", subtoken_length_l=10):
    """
    subtoken_length_l: Controls the smoothness of the transition.
                       Higher = smoother fade. Lower = more binary/noisy.
    """
    
    # 1. Structured Table Data
    table_data = [
        [[" ", False], ["SciQ", False], ["SocialIQA", False], ["McTaco", False], ["TruthfulQA", False], ["BoolQ", False], ["ANLI", False], ["ARC-e", False], ["OBQA", False], ["Avg.", False]],
        [["GPT-Neo", False], ["77.10", False], ["41.25", False], ["42.89", False], ["23.13", False], ["61.99", False], ["33.13", False], ["50.21", False], ["33.60", False], ["45.41", False]],
        [["OPT", False], ["76.70", False], ["40.63", False], ["37.08", False], ["23.75", False], ["57.83", False], ["33.86", False], ["50.97", False], ["33.40", False], ["44.28", False]],
        [["Pythia", False], ["79.20", False], ["40.94", False], ["54.25", False], ["22.77", False], ["63.15", True], ["33.29", False], ["53.91", True], ["33.20", False], ["47.59", False]],
        [["Bloom", False], ["74.60", False], ["39.05", False], ["53.63", False], ["25.58", False], ["59.08", False], ["33.35", False], ["45.41", False], ["29.40", False], ["45.01", False]],
        [["SMDM", False], ["81.20", False], ["41.04", False], ["35.07", False], ["24.60", False], ["62.17", False], ["32.81", False], ["46.13", False], ["33.40", False], ["44.55", False]],
        [["TinyLLaMA", False], ["80.90", False], ["39.56", False], ["40.88", False], ["20.93", False], ["59.20", False], ["33.25", False], ["52.40", False], ["33.40", False], ["45.07", False]],
        [["MDM-Prime-v2", True], ["83.30", True], ["42.02", True], ["66.14", True], ["25.83", True], ["62.05", False], ["34.24", True], ["47.81", False], ["34.00", True], ["49.42", True]],
    ]

    rows = len(table_data)
    cols = len(table_data[0])
    
    # 2. Pre-calculate thresholds
    mask_thresholds = np.random.rand(rows, cols, subtoken_length_l)

    # 3. Setup Figure
    # Increased height to accommodate larger font and taller boxes
    fig, ax = plt.subplots(figsize=(19, 4.5)) 
    
    # TWEAK 1: Increased Box Height (0.65) and Width (2.1) for larger font
    box_w, box_h = 2.6, 0.8
    h_pad, v_pad = 0.1, 0.1
    
    base_mask_color = np.array([0.2, 0.1, 0.7])
    # base_mask_color = np.array([1.0, 1.0, 1.0])

    def update(frame):
        ax.clear()
        
        # Calculate table boundaries
        table_width = cols * (box_w + h_pad)
        table_height = rows * (box_h + v_pad)
        
        # Set limits
        ax.set_xlim(-0.2, table_width + 1.0)
        ax.set_ylim(-0.2, table_height + 2.0)
        
        ax.set_aspect('equal')
        ax.axis('off')

        # TWEAK 2: Slower Animation + Pause
        # Active diffusion frames: 160 (80 up, 80 down) -> Much slower than before
        # Pause frames: 40 (at 50ms interval = 2 seconds)
        active_frames = ACTIVE_FRAME
        half_active = active_frames // 2
        pause_frames = PAUSE_FRAME
        
        if frame < active_frames:
            # Animation Phase
            if frame <= half_active:
                t = frame / half_active     # 0 -> 1
            else:
                t = (active_frames - frame) / half_active # 1 -> 0
        else:
            # Pause Phase (Hold clean state)
            t = 0.0

        # --- Table Drawing Loop ---
        for r in range(rows):
            for c in range(cols):
                cell_text, is_bold = table_data[r][c]
                masks = mask_thresholds[r, c] < t
                masked_ratio = np.mean(masks)

                curr_x = c * (box_w + h_pad)
                curr_y = (rows - 1 - r) * (box_h + v_pad)
                
                # A. Background
                bg_color = "#f2f2f2" if r == 0 else "white"
                rect = patches.Rectangle(
                    (curr_x, curr_y), box_w, box_h,
                    linewidth=0.5, edgecolor="#DDDDDD", facecolor=bg_color
                )
                ax.add_patch(rect)
                
                # B. Mask Overlay
                if masked_ratio > 0:
                    mask_rect = patches.Rectangle(
                        (curr_x, curr_y), box_w, box_h,
                        facecolor=(*base_mask_color, masked_ratio) 
                    )
                    ax.add_patch(mask_rect)
                
                # C. Text
                text_alpha = max(0, 1.0 - masked_ratio)
                if text_alpha > 0.01:
                    weight = 'bold' if (is_bold or r == 0) else 'normal'
                    alignment = 'left' if c == 0 else 'center'
                    x_pos = curr_x + 0.15 if c == 0 else curr_x + box_w/2
                    
                    # TWEAK 3: Larger Font Sizes (13 for data, 14 for headers)
                    f_size = 14 if r == 0 else 13
                    
                    ax.text(
                        x_pos, curr_y + box_h/2, cell_text,
                        ha=alignment, va='center', fontsize=f_size,
                        fontfamily='monospace', fontweight=weight, color='black',
                        alpha=text_alpha
                    )
        
        # --- Legend ---
        legend_w = 6.0
        legend_h = 0.25
        center_x = table_width / 2
        legend_x = center_x - (legend_w / 2)
        legend_y = table_height + 0.8 # Moved up slightly due to taller boxes
        
        # Gradient Bar
        gradient = np.linspace(0, 1, 256)
        gradient = np.vstack((gradient, gradient))
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list("mask_cmap", ["white", (*base_mask_color, 1.0)])

        ax.imshow(gradient, aspect='auto', cmap=cmap, 
                  extent=[legend_x, legend_x + legend_w, legend_y, legend_y + legend_h], zorder=10)
        
        rect_border = patches.Rectangle((legend_x, legend_y), legend_w, legend_h, 
                                        linewidth=1, edgecolor='black', facecolor='none', zorder=11)
        ax.add_patch(rect_border)

        # Labels
        ax.text(legend_x - 0.2, legend_y + legend_h/2, "Masked Ratio", 
                ha='right', va='center', fontsize=16, fontfamily='serif') # Larger label
        
        ticks = [0.0, 0.5, 1.0]
        for tick in ticks:
            tick_x = legend_x + tick * legend_w
            ax.plot([tick_x, tick_x], [legend_y + legend_h, legend_y + legend_h + 0.1], 
                    color='black', linewidth=1, zorder=12)
            ax.text(tick_x, legend_y + legend_h + 0.2, f"{tick:.1f}", 
                    ha='center', va='bottom', fontsize=14, fontfamily='serif')

        # t value
        ax.text(legend_x + legend_w + 0.7, legend_y + legend_h/2, f"t = {t:.2f}", 
                ha='left', va='center', fontsize=16, fontweight='bold', fontfamily='serif')

    # Total frames = 160 (active) + 40 (pause) = 200
    total_frames = ACTIVE_FRAME + PAUSE_FRAME
    update(0)            # Draw the first frame so matplotlib can see the content
    plt.tight_layout()
    ani = FuncAnimation(fig, update, frames=np.arange(0, total_frames), interval=50)
    
    print(f"Generating GIF (Slower + 2s Pause)...")
    ani.save(output_name, writer='pillow')
    print(f"Done! Saved as {output_name}")

if __name__ == "__main__":
    create_table_diffusion_gif(subtoken_length_l=15)