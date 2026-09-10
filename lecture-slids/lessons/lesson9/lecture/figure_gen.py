"""
Figure Generation Script for Lesson 9 Neural Networks Lecture
(PyTorch Version)
Generates all figures needed for the lecture notes.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.patches import FancyBboxPatch
import os
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
warnings.filterwarnings('ignore')

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
figure_dir = os.path.join(script_dir, 'figures')

# Create figures directory if it doesn't exist
os.makedirs(figure_dir, exist_ok=True)

print("Generating figures for Lesson 9 lecture notes (PyTorch Version)...")
print(f"Saving figures to: {figure_dir}")

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# ============================================================================
# Figure 1: XOR Problem Visualization
# ============================================================================
def generate_fig1_xor_problem_viz():
    """
    2D scatter plot showing the XOR problem cannot be separated by a single line.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # XOR data points
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])
    
    # Plot points
    for i in range(4):
        if y[i] == 0:
            ax.scatter(X[i, 0], X[i, 1], s=200, c='blue', marker='o', 
                      edgecolors='black', linewidths=2, label='Class 0' if i == 0 else '')
        else:
            ax.scatter(X[i, 0], X[i, 1], s=200, c='red', marker='x', 
                      linewidths=3, label='Class 1' if i == 1 else '')
    
    # Attempt to draw a decision boundary (it fails!)
    x_line = np.array([-0.2, 1.2])
    y_line = 0.5 * np.ones_like(x_line)
    ax.plot(x_line, y_line, 'g--', linewidth=2, alpha=0.7, 
            label='Attempted Decision Boundary')
    
    # Add annotations showing the problem
    ax.annotate('Cannot separate\nall points!', xy=(0.5, 0.5), xytext=(0.5, -0.3),
                fontsize=12, ha='center', color='green',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    
    ax.set_xlabel('$x_1$', fontsize=14)
    ax.set_ylabel('$x_2$', fontsize=14)
    ax.set_title('XOR Problem: Not Linearly Separable', fontsize=16, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylim(-0.3, 1.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, 'fig1_xor_problem_viz.png'), dpi=100, bbox_inches='tight')
    plt.close()
    print("✓ Generated fig1_xor_problem_viz.png")

# ============================================================================
# Figure 2: Single vs Multiple Units
# ============================================================================
def generate_fig2_single_vs_multiple_units():
    """
    Side-by-side comparison: single logistic unit vs neural network.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    # Helper function to draw nodes and connections
    def draw_network(ax, title, layers):
        ax.set_xlim(0, 4)
        ax.set_ylim(0, 4)
        ax.axis('off')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Draw nodes
        nodes_pos = {}
        for layer_idx, (layer_name, n_nodes, color) in enumerate(layers):
            x = 1 + layer_idx * 1.5
            y_start = 2 - (n_nodes - 1) * 0.4
            for i in range(n_nodes):
                y = y_start + i * 0.8
                circle = Circle((x, y), 0.25, color=color, ec='black', linewidth=2, zorder=3)
                ax.add_patch(circle)
                nodes_pos[(layer_idx, i)] = (x, y)
                
                # Add labels
                if layer_idx == 0:
                    ax.text(x - 0.5, y, f'$x_{i+1}$', fontsize=12, va='center', ha='right')
                elif layer_idx == len(layers) - 1:
                    ax.text(x + 0.5, y, r'$\hat{y}$', fontsize=12, va='center', ha='left')
        
        # Draw connections
        for layer_idx in range(len(layers) - 1):
            n_current = layers[layer_idx][1]
            n_next = layers[layer_idx + 1][1]
            for i in range(n_current):
                for j in range(n_next):
                    x1, y1 = nodes_pos[(layer_idx, i)]
                    x2, y2 = nodes_pos[(layer_idx + 1, j)]
                    arrow = FancyArrowPatch((x1 + 0.25, y1), (x2 - 0.25, y2),
                                          arrowstyle='->', mutation_scale=15,
                                          color='gray', linewidth=1.5, alpha=0.6, zorder=1)
                    ax.add_patch(arrow)
        
        # Add layer labels
        for layer_idx, (layer_name, _, _) in enumerate(layers):
            x = 1 + layer_idx * 1.5
            ax.text(x, 0.3, layer_name, fontsize=11, ha='center', style='italic')
    
    # Left panel: Single logistic regression
    draw_network(ax1, 'Logistic Regression', [
        ('Input', 2, 'lightblue'),
        ('Output', 1, 'orange')
    ])
    
    # Right panel: Neural network
    draw_network(ax2, 'Neural Network', [
        ('Input', 2, 'lightblue'),
        ('Hidden', 2, 'lightgreen'),
        ('Output', 1, 'orange')
    ])
    
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, 'fig2_single_vs_multiple_units.png'), dpi=100, bbox_inches='tight')
    plt.close()
    print("✓ Generated fig2_single_vs_multiple_units.png")

# ============================================================================
# Figure 3: XOR Solution Network
# ============================================================================
def generate_fig3_xor_solution_network():
    """
    Detailed network architecture diagram for XOR solution.
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 5)
    ax.axis('off')
    
    # Define layer positions
    layers = [
        ('Input Layer', [(1, 2.5), (1, 1.5)], 'lightblue', ['$x_1$', '$x_2$']),
        ('Hidden Layer\n(ReLU)', [(3, 3), (3, 1)], 'lightgreen', ['$h_1$', '$h_2$']),
        ('Output Layer\n(sigmoid)', [(5, 2)], 'orange', [r'$\hat{y}$'])
    ]
    
    nodes_pos = {}
    
    # Draw nodes
    for layer_idx, (layer_name, positions, color, labels) in enumerate(layers):
        for node_idx, (x, y) in enumerate(positions):
            circle = Circle((x, y), 0.3, color=color, ec='black', linewidth=2.5, zorder=3)
            ax.add_patch(circle)
            ax.text(x, y, labels[node_idx], fontsize=12, ha='center', va='center', 
                   fontweight='bold', zorder=4)
            nodes_pos[(layer_idx, node_idx)] = (x, y)
        
        # Add layer label
        avg_y = np.mean([pos[1] for pos in positions])
        ax.text(positions[0][0], avg_y - 1.2, layer_name, fontsize=12, 
               ha='center', style='italic', fontweight='bold')
    
    # Draw connections
    for layer_idx in range(len(layers) - 1):
        n_current = len(layers[layer_idx][1])
        n_next = len(layers[layer_idx + 1][1])
        for i in range(n_current):
            for j in range(n_next):
                x1, y1 = nodes_pos[(layer_idx, i)]
                x2, y2 = nodes_pos[(layer_idx + 1, j)]
                arrow = FancyArrowPatch((x1 + 0.3, y1), (x2 - 0.3, y2),
                                      arrowstyle='->', mutation_scale=15,
                                      color='darkgray', linewidth=2, alpha=0.7, zorder=1)
                ax.add_patch(arrow)
    
    # Add annotations for what hidden units learn
    ax.text(3, 3.6, '$x_1$ AND NOT $x_2$', fontsize=10, ha='center',
           bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.8))
    ax.text(3, 0.4, '$x_2$ AND NOT $x_1$', fontsize=10, ha='center',
           bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.8))
    
    ax.set_title('Neural Network Architecture for XOR', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, 'fig3_xor_solution_network.png'), dpi=100, bbox_inches='tight')
    plt.close()
    print("✓ Generated fig3_xor_solution_network.png")

# ============================================================================
# Figure 4: General NN Architecture
# ============================================================================
def generate_fig4_nn_architecture_labeled():
    """
    Clean, professional neural network architecture diagram.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # Define layers
    input_layer = [(1.5, 4), (1.5, 3), (1.5, 2)]
    hidden_layer = [(3.5, 4.5), (3.5, 3.5), (3.5, 2.5), (3.5, 1.5)]
    output_layer = [(5.5, 3)]
    
    all_layers = [
        (input_layer, 'lightblue', 'Input Layer\n(n=3)', ['$x_1$', '$x_2$', '$x_3$']),
        (hidden_layer, 'lightgreen', 'Hidden Layer\n(n=4)', ['$h_1$', '$h_2$', '$h_3$', '$h_4$']),
        (output_layer, 'lightsalmon', 'Output Layer\n(n=1)', [r'$\hat{y}$'])
    ]
    
    nodes_pos = {}
    
    # Draw nodes
    for layer_idx, (positions, color, label, node_labels) in enumerate(all_layers):
        for node_idx, (x, y) in enumerate(positions):
            circle = Circle((x, y), 0.25, color=color, ec='black', linewidth=2, zorder=3)
            ax.add_patch(circle)
            ax.text(x, y, node_labels[node_idx], fontsize=11, ha='center', 
                   va='center', zorder=4)
            nodes_pos[(layer_idx, node_idx)] = (x, y)
        
        # Layer label
        avg_y = np.mean([pos[1] for pos in positions])
        ax.text(positions[0][0], 0.5, label, fontsize=12, ha='center', 
               fontweight='bold', bbox=dict(boxstyle='round,pad=0.5', 
               facecolor='white', edgecolor='black', linewidth=1.5))
    
    # Draw all connections
    for layer_idx in range(len(all_layers) - 1):
        n_current = len(all_layers[layer_idx][0])
        n_next = len(all_layers[layer_idx + 1][0])
        for i in range(n_current):
            for j in range(n_next):
                x1, y1 = nodes_pos[(layer_idx, i)]
                x2, y2 = nodes_pos[(layer_idx + 1, j)]
                ax.plot([x1 + 0.25, x2 - 0.25], [y1, y2], 
                       color='gray', linewidth=0.8, alpha=0.5, zorder=1)
    
    # Add weight and bias annotations
    ax.text(2.5, 5.2, '$W^{(1)}, b^{(1)}$', fontsize=11, ha='center',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='lavender', alpha=0.8))
    ax.text(4.5, 5.2, '$W^{(2)}, b^{(2)}$', fontsize=11, ha='center',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='lavender', alpha=0.8))
    
    ax.set_title('General Neural Network Architecture', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, 'fig4_nn_architecture_labeled.png'), dpi=100, bbox_inches='tight')
    plt.close()
    print("✓ Generated fig4_nn_architecture_labeled.png")

# ============================================================================
# Figure 5: Activation Functions
# ============================================================================
def generate_fig5_activation_functions():
    """
    Plot of sigmoid, ReLU, and tanh activation functions.
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    
    z = np.linspace(-5, 5, 200)
    
    # Sigmoid
    sigmoid = 1 / (1 + np.exp(-z))
    ax.plot(z, sigmoid, 'b-', linewidth=2.5, label='Sigmoid: $\sigma(z) = \\frac{1}{1+e^{-z}}$')
    
    # ReLU
    relu = np.maximum(0, z)
    ax.plot(z, relu, 'r-', linewidth=2.5, label='ReLU: $f(z) = \max(0, z)$')
    
    # Tanh
    tanh = np.tanh(z)
    ax.plot(z, tanh, 'g-', linewidth=2.5, label='Tanh: $f(z) = \\frac{e^z - e^{-z}}{e^z + e^{-z}}$')
    
    ax.axhline(y=0, color='black', linewidth=0.8, linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='black', linewidth=0.8, linestyle='-', alpha=0.3)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    ax.set_xlabel('Input (z)', fontsize=13)
    ax.set_ylabel('Activation Output', fontsize=13)
    ax.set_title('Common Activation Functions', fontsize=16, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-1.5, 1.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, 'fig5_activation_functions.png'), dpi=100, bbox_inches='tight')
    plt.close()
    print("✓ Generated fig5_activation_functions.png")

# ============================================================================
# Figure 6: XOR Solution (PyTorch Replacement)
# ============================================================================
def generate_fig6_xor_torch_solution():
    """
    Decision boundary visualization showing neural network solving XOR.
    Replaced Keras with PyTorch.
    """
    # XOR dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y = np.array([[0], [1], [1], [0]], dtype=np.float32)
    
    # Convert to PyTorch tensors
    X_torch = torch.from_numpy(X)
    y_torch = torch.from_numpy(y)
    
    # Build a simple neural network using PyTorch
    # 2 inputs -> 8 hidden (ReLU) -> 1 output (Sigmoid)
    model = nn.Sequential(
        nn.Linear(2, 8),
        nn.ReLU(),
        nn.Linear(8, 1),
        nn.Sigmoid()
    )
    
    # Optimizer and Loss
    optimizer = optim.Adam(model.parameters(), lr=0.05)
    criterion = nn.BCELoss()
    
    # Training loop
    epochs = 1000
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_torch)
        loss = criterion(outputs, y_torch)
        loss.backward()
        optimizer.step()
    
    # Create mesh for decision boundary
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    
    # Predict for all points in the mesh
    grid_points = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)
    grid_tensor = torch.from_numpy(grid_points)
    
    # Evaluation mode
    model.eval()
    with torch.no_grad():
        Z = model(grid_tensor).numpy()
    
    Z = Z.reshape(xx.shape)
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Decision boundary background
    contour = ax.contourf(xx, yy, Z, levels=np.linspace(0, 1, 20), cmap='RdBu', alpha=0.6)
    # Add decision boundary line (where prob = 0.5)
    ax.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2, linestyles='--')
    
    # Plot original data points
    y_flat = y.flatten()
    for i in range(4):
        if y_flat[i] == 0:
            ax.scatter(X[i, 0], X[i, 1], s=250, c='blue', marker='o', 
                      edgecolors='black', linewidths=3, label='Class 0' if i == 0 else '', zorder=3)
        else:
            ax.scatter(X[i, 0], X[i, 1], s=250, c='red', marker='x', 
                      linewidths=4, label='Class 1' if i == 1 else '', zorder=3)
    
    # Colorbar
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label('Predicted Probability', fontsize=11)
    
    ax.set_xlabel('$x_1$', fontsize=14)
    ax.set_ylabel('$x_2$', fontsize=14)
    ax.set_title('Neural Network Decision Boundary for XOR\n(Generated via PyTorch)', fontsize=16, fontweight='bold')
    ax.legend(loc='upper left', fontsize=12)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, 'fig6_xor_solution.png'), dpi=100, bbox_inches='tight')
    plt.close()
    print("✓ Generated fig6_xor_solution.png (PyTorch)")

# ============================================================================
# Figure 7: Training Curves
# ============================================================================
def generate_fig7_training_curves():
    """
    Training and validation loss curves showing good training vs overfitting.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6))
    
    epochs = np.arange(0, 100)
    
    # Top panel: Good training
    train_loss_good = 0.5 * np.exp(-epochs / 20) + 0.05
    ax1.plot(epochs, train_loss_good, 'b-', linewidth=2.5, label='Training Loss')
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Good Training: Loss Decreases and Converges', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 0.6)
    
    # Bottom panel: Overfitting
    train_loss_overfit = 0.5 * np.exp(-epochs / 15) + 0.03
    val_loss_overfit = 0.5 * np.exp(-epochs / 20) + 0.1
    val_loss_overfit[50:] = val_loss_overfit[50:] + (epochs[50:] - 50) * 0.002
    
    ax2.plot(epochs, train_loss_overfit, 'b-', linewidth=2.5, label='Training Loss')
    ax2.plot(epochs, val_loss_overfit, color='orange', linewidth=2.5, label='Validation Loss')
    
    # Highlight overfitting region
    ax2.axvline(x=50, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax2.text(52, 0.4, 'Overfitting\nbegins here', fontsize=10, color='red', fontweight='bold')
    
    ax2.set_xlabel('Epochs', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Overfitting: Validation Loss Increases While Training Loss Decreases', 
                  fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 0.6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, 'fig7_training_curves.png'), dpi=100, bbox_inches='tight')
    plt.close()
    print("✓ Generated fig7_training_curves.png")

# ============================================================================
# Figure 8: Backpropagation Flow
# ============================================================================
def generate_fig8_backprop_flow():
    """
    Network diagram showing forward and backward propagation.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 5)
    ax.axis('off')
    
    # Define layer positions
    input_layer = [(1.5, 3), (1.5, 2)]
    hidden_layer = [(4, 3.5), (4, 1.5)]
    output_layer = [(6.5, 2.5)]
    
    all_layers = [input_layer, hidden_layer, output_layer]
    colors = ['lightblue', 'lightgreen', 'lightsalmon']
    labels = [['$x_1$', '$x_2$'], ['$h_1$', '$h_2$'], [r'$\hat{y}$']]
    
    nodes_pos = {}
    
    # Draw nodes
    for layer_idx, (positions, color, node_labels) in enumerate(zip(all_layers, colors, labels)):
        for node_idx, (x, y) in enumerate(positions):
            circle = Circle((x, y), 0.3, color=color, ec='black', linewidth=2, zorder=3)
            ax.add_patch(circle)
            ax.text(x, y, node_labels[node_idx], fontsize=12, ha='center', 
                   va='center', fontweight='bold', zorder=4)
            nodes_pos[(layer_idx, node_idx)] = (x, y)
    
    # Draw forward pass (green solid arrows)
    for layer_idx in range(len(all_layers) - 1):
        for i in range(len(all_layers[layer_idx])):
            for j in range(len(all_layers[layer_idx + 1])):
                x1, y1 = nodes_pos[(layer_idx, i)]
                x2, y2 = nodes_pos[(layer_idx + 1, j)]
                arrow = FancyArrowPatch((x1 + 0.3, y1), (x2 - 0.3, y2),
                                      arrowstyle='->', mutation_scale=20,
                                      color='green', linewidth=3, alpha=0.7, zorder=2)
                ax.add_patch(arrow)
                
                # Add annotation for first connection
                if layer_idx == 0 and i == 0 and j == 0:
                    mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                    ax.text(mid_x, mid_y + 0.3, '$a^{(1)}$', fontsize=10, 
                           ha='center', color='green', fontweight='bold')
    
    # Draw backward pass (red dashed arrows)
    for layer_idx in range(len(all_layers) - 1, 0, -1):
        for i in range(len(all_layers[layer_idx])):
            for j in range(len(all_layers[layer_idx - 1])):
                x1, y1 = nodes_pos[(layer_idx, i)]
                x2, y2 = nodes_pos[(layer_idx - 1, j)]
                arrow = FancyArrowPatch((x1 - 0.3, y1), (x2 + 0.3, y2),
                                      arrowstyle='->', mutation_scale=20,
                                      color='red', linewidth=3, alpha=0.7, 
                                      linestyle='--', zorder=1)
                ax.add_patch(arrow)
                
                # Add annotation for first backward connection
                if layer_idx == 2 and i == 0 and j == 0:
                    mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                    ax.text(mid_x, mid_y - 0.3, '$\delta^{(2)}$', fontsize=10, 
                           ha='center', color='red', fontweight='bold')
    
    # Add layer labels
    ax.text(1.5, 0.5, 'Input\nLayer', fontsize=11, ha='center', fontweight='bold')
    ax.text(4, 0.5, 'Hidden\nLayer', fontsize=11, ha='center', fontweight='bold')
    ax.text(6.5, 0.5, 'Output\nLayer', fontsize=11, ha='center', fontweight='bold')
    
    # Add legend
    forward_patch = plt.Line2D([0], [0], color='green', linewidth=3, label='Forward Pass')
    backward_patch = plt.Line2D([0], [0], color='red', linewidth=3, 
                               linestyle='--', label='Backward Pass: Gradients')
    ax.legend(handles=[forward_patch, backward_patch], loc='upper center', 
             fontsize=12, ncol=2, bbox_to_anchor=(0.5, 1.05))
    
    ax.set_title('Forward and Backward Propagation', fontsize=16, fontweight='bold', pad=30)
    
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, 'fig8_backprop_flow.png'), dpi=100, bbox_inches='tight')
    plt.close()
    print("✓ Generated fig8_backprop_flow.png")

# ============================================================================
# Main execution
# ============================================================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("GENERATING FIGURES FOR LESSON 9: NEURAL NETWORKS (TORCH)")
    print("="*70 + "\n")
    
    generate_fig1_xor_problem_viz()
    generate_fig2_single_vs_multiple_units()
    generate_fig3_xor_solution_network()
    generate_fig4_nn_architecture_labeled()
    generate_fig5_activation_functions()
    generate_fig6_xor_torch_solution() # <-- Updated function call
    generate_fig7_training_curves()
    generate_fig8_backprop_flow()
    
    print("\n" + "="*70)
    print(f"✓ ALL FIGURES GENERATED SUCCESSFULLY!")
    print(f"✓ Saved to: {figure_dir}")
    print("="*70 + "\n")