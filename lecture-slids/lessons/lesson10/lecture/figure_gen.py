"""
figure_gen.py
Generate all figures for Lesson 10: Convolutional Neural Networks lecture notes
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.collections import PatchCollection
import warnings
warnings.filterwarnings('ignore')

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
figure_dir = os.path.join(script_dir, 'figures')

# Create figure directory if it doesn't exist
os.makedirs(figure_dir, exist_ok=True)

print(f"Saving figures to: {figure_dir}")

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10


def save_figure(fig, filename):
    """Save figure to the figure directory"""
    filepath = os.path.join(figure_dir, filename)
    fig.savefig(filepath, dpi=100, bbox_inches='tight', facecolor='white')
    print(f"Saved: {filename}")
    plt.close(fig)


# ============================================================================
# Figure 1: Parameter Explosion
# ============================================================================
def generate_fig1_parameter_explosion():
# ... (Figure 1 code remains unchanged) ...
    fig, ax = plt.subplots(figsize=(8, 4))
    
    categories = ['Grayscale\n(28×28×1)', 'RGB\n(224×224×3)']
    
    # Parameter counts
    fc_params = [78400, 15052800]  # Fully-connected with 100 neurons
    cnn_params = [320, 960]  # CNN with 32 filters, 3×3
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, fc_params, width, label='Fully-Connected (100 neurons)',
                   color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, cnn_params, width, label='CNN (32 filters, 3×3)',
                   color='#e74c3c', alpha=0.8)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 1000:
                label = f'{height/1000:.0f}K' if height < 1000000 else f'{height/1000000:.1f}M'
            else:
                label = f'{int(height)}'
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   label, ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Input Image Size', fontweight='bold')
    ax.set_ylabel('Number of Parameters', fontweight='bold')
    ax.set_title('Parameter Count: Fully-Connected vs CNN', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(loc='upper left')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    save_figure(fig, 'fig1_parameter_explosion.png')


# ============================================================================
# Figure 2: Convolution Operation
# ============================================================================
def generate_fig2_convolution_operation():
# ... (Figure 2 code remains unchanged) ...
    fig, axes = plt.subplots(1, 4, figsize=(10, 5))
    
    # Create a simple 5×5 input image
    input_image = np.array([
        [1, 2, 3, 2, 1],
        [2, 3, 4, 3, 2],
        [3, 4, 5, 4, 3],
        [2, 3, 4, 3, 2],
        [1, 2, 3, 2, 1]
    ])
    
    # 3×3 filter (vertical edge detector)
    kernel = np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ])
    
    # Four positions to visualize
    positions = [(0, 0), (0, 1), (1, 1), (2, 2)]
    
    for idx, (i, j) in enumerate(positions):
        ax = axes[idx]
        
        # Show input image with highlighted region
        ax.imshow(input_image, cmap='gray', alpha=0.6)
        
        # Highlight the 3×3 region
        rect = Rectangle((j-0.5, i-0.5), 3, 3, linewidth=3, 
                        edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        
        # Add grid
        for x in range(6):
            ax.axhline(x-0.5, color='black', linewidth=0.5)
            ax.axvline(x-0.5, color='black', linewidth=0.5)
        
        # Add numbers to cells
        for ii in range(5):
            for jj in range(5):
                ax.text(jj, ii, f'{input_image[ii, jj]:.0f}', 
                       ha='center', va='center', fontsize=8)
        
        # Calculate output
        region = input_image[i:i+3, j:j+3]
        output_val = np.sum(region * kernel)
        
        ax.set_xlim(-0.5, 4.5)
        ax.set_ylim(4.5, -0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f'Position ({i},{j})\nOutput = {output_val:.1f}', fontsize=9)
        
        # Add arrow and formula for first position
        if idx == 0:
            ax.text(2, 5.5, '⊗ Filter [-1,0,1; -1,0,1; -1,0,1]', 
                   ha='center', fontsize=7)
    
    fig.suptitle('How Convolution Works: Sliding Window', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    save_figure(fig, 'fig2_convolution_operation.png')


# ============================================================================
# Figure 3: Filter Examples
# ============================================================================
def generate_fig3_filter_examples():
# ... (Figure 3 code remains unchanged) ...
    fig, axes = plt.subplots(3, 3, figsize=(12, 4))
    
    # Create input image with geometric shapes
    input_img = np.zeros((20, 20))
    input_img[5:15, 5:7] = 1    # Vertical bar
    input_img[5:7, 5:15] = 1    # Horizontal bar
    input_img[13:15, 5:15] = 1  # Horizontal bar
    input_img[5:15, 13:15] = 1  # Vertical bar
    
    # Define filters
    filters = {
        'Vertical Edge': np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]),
        'Horizontal Edge': np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]),
        'Blur': np.ones((3, 3)) / 9
    }
    
    filter_names = ['Vertical Edge', 'Horizontal Edge', 'Blur']
    
    for idx, filter_name in enumerate(filter_names):
        kernel = filters[filter_name]
        
        # Convolve
        from scipy.ndimage import convolve
        output = convolve(input_img, kernel, mode='constant')
        
        # Plot input
        axes[idx, 0].imshow(input_img, cmap='gray', vmin=0, vmax=1)
        axes[idx, 0].set_title('Input Image' if idx == 0 else '')
        axes[idx, 0].axis('off')
        
        # Plot filter
        im = axes[idx, 1].imshow(kernel, cmap='RdBu', vmin=-1, vmax=1)
        axes[idx, 1].set_title('Filter' if idx == 0 else '')
        for i in range(3):
            for j in range(3):
                axes[idx, 1].text(j, i, f'{kernel[i, j]:.2f}', 
                                ha='center', va='center', fontsize=8)
        axes[idx, 1].axis('off')
        
        # Plot output
        axes[idx, 2].imshow(output, cmap='gray')
        axes[idx, 2].set_title('Output' if idx == 0 else '')
        axes[idx, 2].axis('off')
        
        # Add row label
        axes[idx, 0].text(-3, 10, filter_name, rotation=90, 
                         va='center', ha='right', fontweight='bold', fontsize=10)
    
    fig.suptitle('Different Filters Detect Different Patterns', 
                fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    save_figure(fig, 'fig3_filter_examples.png')


# ============================================================================
# Figure 4: Multiple Filters
# ============================================================================
def generate_fig4_multiple_filters():
    """One input image producing multiple feature maps"""
    fig = plt.figure(figsize=(10, 6))
    
    # Create a simple digit-like input
    input_img = np.zeros((28, 28))
    # Draw a "3"-like shape
    input_img[8:10, 8:20] = 1
    input_img[13:15, 8:20] = 1
    input_img[18:20, 8:20] = 1
    input_img[8:20, 18:20] = 1
    
    # Define 4 different filters
    filters = [
        np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]),  # Vertical
        np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]),   # Horizontal
        np.array([[-1, -1, 0], [-1, 0, 1], [0, 1, 1]]),   # Diagonal
        np.ones((3, 3)) / 9                                # Blur
    ]
    
    # FIX 1: Change GridSpec to 4 rows to accommodate 4 feature maps sequentially
    gs = fig.add_gridspec(4, 5, width_ratios=[3, 0.5, 1, 0.5, 3], 
                         height_ratios=[1, 1, 1, 1]) # Changed 3 rows to 4 rows
    
    # Input image
    ax_input = fig.add_subplot(gs[:, 0])
    ax_input.imshow(input_img, cmap='gray', vmin=0, vmax=1)
    ax_input.set_title('Input Image\n28×28×1', fontweight='bold', fontsize=11)
    ax_input.axis('off')
    
    # Filters (Plotting sequentially in Column 2, Rows 0 to 3)
    for i, kernel in enumerate(filters):
        # FIX 2: Simplified indexing to plot sequentially in rows 0, 1, 2, 3 of column 2
        ax = fig.add_subplot(gs[i, 2]) 
        
        ax.imshow(kernel, cmap='RdBu', vmin=-1, vmax=1)
        ax.set_title(f'Filter {i+1}', fontsize=9)
        ax.axis('off')
    
    # Feature maps (outputs)
    from scipy.ndimage import convolve
    for i, kernel in enumerate(filters):
        output = convolve(input_img, kernel, mode='constant')
        
        # FIX 3: Simplified indexing to plot sequentially in rows 0, 1, 2, 3 of column 4
        ax = fig.add_subplot(gs[i, 4])
        
        ax.imshow(output, cmap='viridis')
        ax.set_title(f'Feature Map {i+1}', fontsize=9)
        ax.axis('off')
        
        if i == 0:
            # Adjust Y coordinate slightly due to added row height
            ax.text(-3, 16, '28×28×4', rotation=90, fontsize=11, 
                   va='center', ha='right', fontweight='bold')
    
    # Arrows in middle
    # Note: The original arrow drawing logic relied on gs[:, 1] and gs[:, 3] which still exist.
    ax_arrow = fig.add_subplot(gs[:, 1])
    ax_arrow.text(0.5, 0.5, '→', fontsize=40, ha='center', va='center', 
                 transform=ax_arrow.transAxes)
    ax_arrow.axis('off')
    
    ax_arrow2 = fig.add_subplot(gs[:, 3])
    ax_arrow2.text(0.5, 0.5, '→', fontsize=40, ha='center', va='center',
                  transform=ax_arrow2.transAxes)
    ax_arrow2.axis('off')
    
    fig.suptitle('Multiple Filters → Multiple Feature Maps', 
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    
    save_figure(fig, 'fig4_multiple_filters.png')


# ============================================================================
# Figure 5: Pooling Operation
# ============================================================================
def generate_fig5_pooling_operation():
# ... (Figure 5 code remains unchanged) ...
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    
    # 4×4 input
    input_data = np.array([
        [1, 3, 2, 4],
        [5, 6, 1, 2],
        [2, 8, 3, 7],
        [1, 4, 6, 9]
    ])
    
    # Colors for pooling regions
    colors = ['#ffcccc', '#ccffcc', '#ccccff', '#ffffcc']
    
    for ax_idx, (ax, method) in enumerate(zip(axes, ['Max Pooling', 'Average Pooling'])):
        # Draw input grid
        for i in range(4):
            for j in range(4):
                # Determine color based on pooling region
                region_idx = (i // 2) * 2 + (j // 2)
                color = colors[region_idx]
                
                rect = Rectangle((j, 3-i), 1, 1, linewidth=2, 
                               edgecolor='black', facecolor=color, alpha=0.3)
                ax.add_patch(rect)
                
                # Add value
                ax.text(j+0.5, 3-i+0.5, str(input_data[i, j]), 
                       ha='center', va='center', fontsize=12, fontweight='bold')
        
        # Draw pooling region borders
        ax.plot([0, 2, 2, 0, 0], [4, 4, 2, 2, 4], 'r-', linewidth=3)
        ax.plot([2, 4, 4, 2, 2], [4, 4, 2, 2, 4], 'r-', linewidth=3)
        ax.plot([0, 2, 2, 0, 0], [2, 2, 0, 0, 2], 'r-', linewidth=3)
        ax.plot([2, 4, 4, 2, 2], [2, 2, 0, 0, 2], 'r-', linewidth=3)
        
        # Calculate pooled output
        output = np.zeros((2, 2))
        for i in range(2):
            for j in range(2):
                region = input_data[i*2:(i+1)*2, j*2:(j+1)*2]
                if method == 'Max Pooling':
                    output[i, j] = np.max(region)
                else:
                    output[i, j] = np.mean(region)
        
        # Draw output
        for i in range(2):
            for j in range(2):
                x_offset = 5.5
                y_offset = 1
                
                region_idx = i * 2 + j
                color = colors[region_idx]
                
                rect = Rectangle((x_offset + j*1.2, y_offset + (1-i)*1.2), 1, 1, 
                               linewidth=2, edgecolor='black', facecolor=color, alpha=0.3)
                ax.add_patch(rect)
                
                value_str = f'{output[i, j]:.1f}' if method == 'Average Pooling' else f'{int(output[i, j])}'
                ax.text(x_offset + j*1.2 + 0.5, y_offset + (1-i)*1.2 + 0.5, 
                       value_str, ha='center', va='center', fontsize=12, fontweight='bold')
        
        # Arrow
        ax.annotate('', xy=(5, 2), xytext=(4.3, 2), 
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))
        
        # Labels
        ax.text(2, -0.5, '4×4 Input', ha='center', fontsize=10, fontweight='bold')
        ax.text(6.1, -0.5, '2×2 Output', ha='center', fontsize=10, fontweight='bold')
        ax.text(4.65, 2.5, 'pool_size=2×2\nstride=2', ha='center', fontsize=8, 
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlim(-0.5, 8)
        ax.set_ylim(-1, 4.5)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(method, fontsize=12, fontweight='bold')
    
    fig.suptitle('Pooling Reduces Spatial Dimensions', fontsize=13, fontweight='bold')
    plt.tight_layout()
    
    save_figure(fig, 'fig5_pooling_operation.png')


# ============================================================================
# Figure 6: CNN Architecture
# ============================================================================
def generate_fig6_cnn_architecture():
# ... (Figure 6 code remains unchanged) ...
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Architecture layers with positions and sizes
    layers = [
        {'name': 'Input', 'dims': '28×28×1', 'params': '0', 'pos': 0.5, 'depth': 0.5, 'color': '#95a5a6'},
        {'name': 'Conv2D(32)', 'dims': '28×28×32', 'params': '320', 'pos': 2, 'depth': 3, 'color': '#3498db'},
        {'name': 'MaxPool2D', 'dims': '14×14×32', 'params': '0', 'pos': 3.5, 'depth': 3, 'color': '#2ecc71'},
        {'name': 'Conv2D(64)', 'dims': '14×14×64', 'params': '18,496', 'pos': 5, 'depth': 4.5, 'color': '#3498db'},
        {'name': 'MaxPool2D', 'dims': '7×7×64', 'params': '0', 'pos': 6.5, 'depth': 4.5, 'color': '#2ecc71'},
        {'name': 'Flatten', 'dims': '3,136', 'params': '0', 'pos': 8, 'depth': 0, 'color': '#9b59b6'},
        {'name': 'Dense(128)', 'dims': '128', 'params': '401,536', 'pos': 9.5, 'depth': 0, 'color': '#e67e22'},
        {'name': 'Dense(10)', 'dims': '10', 'params': '1,290', 'pos': 11, 'depth': 0, 'color': '#e67e22'},
    ]
    
    max_depth = 4.5
    
    for i, layer in enumerate(layers):
        x = layer['pos']
        
        if layer['depth'] > 0:  # 3D boxes for conv/pool layers
            # Draw 3D-ish box
            depth = layer['depth']
            height = 3
            width = 0.6
            
            # Front face
            front = Rectangle((x - width/2, 2), width, height, 
                            linewidth=2, edgecolor='black', 
                            facecolor=layer['color'], alpha=0.7)
            ax.add_patch(front)
            
            # Top face (for depth illusion)
            top_x = [x - width/2, x + width/2, x + width/2 + depth*0.15, x - width/2 + depth*0.15, x - width/2]
            top_y = [2 + height, 2 + height, 2 + height + depth*0.15, 2 + height + depth*0.15, 2 + height]
            ax.fill(top_x, top_y, color=layer['color'], alpha=0.5, edgecolor='black', linewidth=1.5)
            
            # Right face
            right_x = [x + width/2, x + width/2 + depth*0.15, x + width/2 + depth*0.15, x + width/2, x + width/2]
            right_y = [2, 2 + depth*0.15, 2 + height + depth*0.15, 2 + height, 2]
            ax.fill(right_x, right_y, color=layer['color'], alpha=0.4, edgecolor='black', linewidth=1.5)
            
            # Label
            ax.text(x, 1.5, layer['name'], ha='center', fontsize=9, fontweight='bold')
            ax.text(x, 1.2, layer['dims'], ha='center', fontsize=8)
            ax.text(x, 0.9, f"{layer['params']} params", ha='center', fontsize=7, style='italic')
            
        else:  # 1D bars for dense layers
            height = max(0.3, 3 * (float(layer['dims'].replace(',', '')) / 3136))
            width = 0.3
            
            bar = Rectangle((x - width/2, 2), width, height, 
                          linewidth=2, edgecolor='black', 
                          facecolor=layer['color'], alpha=0.7)
            ax.add_patch(bar)
            
            ax.text(x, 1.5, layer['name'], ha='center', fontsize=9, fontweight='bold')
            ax.text(x, 1.2, layer['dims'], ha='center', fontsize=8)
            ax.text(x, 0.9, f"{layer['params']} params", ha='center', fontsize=7, style='italic')
        
        # Draw arrows between layers
        if i < len(layers) - 1:
            next_x = layers[i+1]['pos']
            arrow = FancyArrowPatch((x + 0.5, 3.5), (next_x - 0.5, 3.5),
                                   arrowstyle='->', mutation_scale=20, 
                                   linewidth=2, color='black')
            ax.add_patch(arrow)
    
    # Add legend
    legend_elements = [
        patches.Patch(facecolor='#3498db', edgecolor='black', label='Convolutional Layer'),
        patches.Patch(facecolor='#2ecc71', edgecolor='black', label='Pooling Layer'),
        patches.Patch(facecolor='#9b59b6', edgecolor='black', label='Flatten'),
        patches.Patch(facecolor='#e67e22', edgecolor='black', label='Dense Layer')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    # Total parameters
    ax.text(6, 6.5, 'Total Parameters: 421,642', fontsize=11, 
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3),
           ha='center', fontweight='bold')
    
    ax.set_xlim(-0.5, 12)
    ax.set_ylim(0, 7)
    ax.axis('off')
    ax.set_title('CNN Architecture for MNIST Classification', fontsize=13, fontweight='bold', pad=20)
    
    save_figure(fig, 'fig6_cnn_architecture.png')


# ============================================================================
# Figure 7: Training Curves
# ============================================================================
def generate_fig7_training_curves():
# ... (Figure 7 code remains unchanged) ...
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    
    # Simulate realistic training curves
    epochs = np.arange(1, 6)
    
    # Accuracy curves
    train_acc = np.array([0.92, 0.96, 0.975, 0.985, 0.990])
    val_acc = np.array([0.95, 0.97, 0.980, 0.982, 0.983])
    
    # Loss curves
    train_loss = np.array([0.25, 0.12, 0.08, 0.05, 0.03])
    val_loss = np.array([0.15, 0.10, 0.08, 0.07, 0.065])
    
    # Plot accuracy
    ax1.plot(epochs, train_acc, 'o-', linewidth=2, markersize=8, 
            label='Training', color='#3498db')
    ax1.plot(epochs, val_acc, 's--', linewidth=2, markersize=8, 
            label='Validation', color='#e67e22')
    ax1.set_xlabel('Epoch', fontweight='bold')
    ax1.set_ylabel('Accuracy', fontweight='bold')
    ax1.set_title('Model Accuracy', fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.9, 1.0)
    ax1.set_xticks(epochs)
    
    # Plot loss
    ax2.plot(epochs, train_loss, 'o-', linewidth=2, markersize=8, 
            label='Training', color='#3498db')
    ax2.plot(epochs, val_loss, 's--', linewidth=2, markersize=8, 
            label='Validation', color='#e67e22')
    ax2.set_xlabel('Epoch', fontweight='bold')
    ax2.set_ylabel('Loss', fontweight='bold')
    ax2.set_title('Model Loss', fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(epochs)
    
    fig.suptitle('Training Progress: MNIST CNN', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    save_figure(fig, 'fig7_training_curves.png')


# ============================================================================
# Figure 8: Hierarchical Features
# ============================================================================
def generate_fig8_hierarchical_features():
# ... (Figure 8 code remains unchanged) ...
    fig, axes = plt.subplots(3, 6, figsize=(10, 6))
    
    np.random.seed(42)
    
    layer_names = ['Conv1: Simple Edges', 'Conv2: Textures', 'Conv3: Parts']
    
    for layer_idx, layer_name in enumerate(layer_names):
        for feature_idx in range(6):
            ax = axes[layer_idx, feature_idx]
            
            # Generate different types of features for each layer
            if layer_idx == 0:  # Layer 1: Simple edges
                # Create Gabor-like filters (edges at different orientations)
                angle = feature_idx * 30
                x, y = np.meshgrid(np.linspace(-1, 1, 16), np.linspace(-1, 1, 16))
                
                # Rotate coordinates
                angle_rad = np.deg2rad(angle)
                x_rot = x * np.cos(angle_rad) - y * np.sin(angle_rad)
                
                # Gabor-like pattern
                feature = np.cos(5 * x_rot) * np.exp(-(x**2 + y**2) / 0.5)
                
            elif layer_idx == 1:  # Layer 2: Textures
                # Create texture-like patterns
                x, y = np.meshgrid(np.linspace(-2, 2, 16), np.linspace(-2, 2, 16))
                
                if feature_idx == 0:
                    feature = np.sin(3*x) * np.cos(3*y)
                elif feature_idx == 1:
                    feature = np.cos(4*x) * np.cos(4*y)
                elif feature_idx == 2:
                    r = np.sqrt(x**2 + y**2)
                    feature = np.cos(5*r) * np.exp(-r**2/2)
                elif feature_idx == 3:
                    feature = np.sin(3*x + 3*y) 
                elif feature_idx == 4:
                    feature = (np.cos(3*x) + np.cos(3*y)) / 2
                else:
                    feature = np.sin(2*x) * np.sin(2*y) * np.cos(x*y)
                    
            else:  # Layer 3: More abstract
                # More complex, abstract patterns
                x, y = np.meshgrid(np.linspace(-2, 2, 16), np.linspace(-2, 2, 16))
                r = np.sqrt(x**2 + y**2)
                theta = np.arctan2(y, x)
                
                if feature_idx == 0:
                    feature = np.cos(3*theta) * np.exp(-r**2/2)
                elif feature_idx == 1:
                    feature = (np.sin(2*r) * np.cos(3*theta) + 
                             np.cos(2*r) * np.sin(3*theta)) * np.exp(-r**2/3)
                elif feature_idx == 2:
                    feature = np.sin(x**2 + y**2) * np.cos(3*theta)
                elif feature_idx == 3:
                    feature = (np.cos(2*x) * np.sin(2*y) + 
                             np.sin(x) * np.cos(y)) / 2
                elif feature_idx == 4:
                    feature = np.sin(3*r) * np.cos(5*theta) * np.exp(-r**2/4)
                else:
                    feature = (np.cos(2*r + 3*theta) + 
                             np.sin(3*r - 2*theta)) * np.exp(-r**2/3)
            
            # Plot
            ax.imshow(feature, cmap='gray', vmin=-1, vmax=1)
            ax.axis('off')
            
            # Add title only for first column
            if feature_idx == 0:
                ax.text(-2, 8, layer_name, rotation=90, va='center', ha='right',
                       fontsize=11, fontweight='bold')
    
    fig.suptitle('Hierarchical Feature Learning in CNNs', fontsize=13, fontweight='bold')
    fig.text(0.5, 0.02, 'Features learned become more complex in deeper layers', 
            ha='center', fontsize=10, style='italic')
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    save_figure(fig, 'fig8_hierarchical_features.png')


# ============================================================================
# Main execution
# ============================================================================
if __name__ == '__main__':
    print("\n" + "="*70)
    print("Generating figures for Lesson 10: Convolutional Neural Networks")
    print("="*70 + "\n")
    
    print("Generating Figure 1: Parameter Explosion...")
    generate_fig1_parameter_explosion()
    
    print("Generating Figure 2: Convolution Operation...")
    generate_fig2_convolution_operation()
    
    print("Generating Figure 3: Filter Examples...")
    generate_fig3_filter_examples()
    
    print("Generating Figure 4: Multiple Filters...")
    generate_fig4_multiple_filters()
    
    print("Generating Figure 5: Pooling Operation...")
    generate_fig5_pooling_operation()
    
    print("Generating Figure 6: CNN Architecture...")
    generate_fig6_cnn_architecture()
    
    print("Generating Figure 7: Training Curves...")
    generate_fig7_training_curves()
    
    print("Generating Figure 8: Hierarchical Features...")
    generate_fig8_hierarchical_features()
    
    print("\n" + "="*70)
    print("All figures generated successfully!")
    print(f"Figures saved to: {figure_dir}")
    print("="*70 + "\n")
