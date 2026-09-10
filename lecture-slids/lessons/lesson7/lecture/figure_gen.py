"""
figure_gen.py
Generate all figures for Lesson 7 PCA Lecture Notes

This script creates 6 figures demonstrating PCA concepts and saves them
to a ./figures/ directory in the same location as this script.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
from matplotlib.patches import Ellipse
import matplotlib.patches as mpatches

# Set random seed for reproducibility
np.random.seed(42)

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Create figures directory if it doesn't exist
figures_dir = os.path.join(script_dir, 'figures')
os.makedirs(figures_dir, exist_ok=True)

print(f"Saving figures to: {figures_dir}")


def generate_correlated_data(n_samples=300, mean=[0, 0], cov=[[2, 1.5], [1.5, 2]]):
    """Generate 2D correlated data for PCA demonstrations."""
    return np.random.multivariate_normal(mean, cov, n_samples)


# =============================================================================
# Figure 1: fig_pca_intuition
# Simple 2D scatter showing data varying along one main direction
# =============================================================================
def create_fig_pca_intuition():
    """Create simple scatter plot showing elliptical data distribution."""
    print("Generating fig_pca_intuition...")
    
    # Generate data
    data = generate_correlated_data(n_samples=300)
    
    # Create figure
    plt.figure(figsize=(7, 6))
    plt.scatter(data[:, 0], data[:, 1], alpha=0.5, s=30, color='steelblue')
    plt.xlabel('Feature 1', fontsize=12)
    plt.ylabel('Feature 2', fontsize=12)
    plt.title('Data Varies More in One Direction', fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Save
    filepath = os.path.join(figures_dir, 'fig_pca_intuition.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")


# =============================================================================
# Figure 2: fig_variance_directions
# 2D scatter with multiple candidate directions, one highlighted
# =============================================================================
def create_fig_variance_directions():
    """Create scatter plot with candidate projection directions."""
    print("Generating fig_variance_directions...")
    
    # Generate data
    data = generate_correlated_data(n_samples=300)
    
    # Compute PCA to find true maximum variance direction
    pca = PCA(n_components=2)
    pca.fit(data)
    pc1 = pca.components_[0]
    
    # Define several candidate directions
    angles = [30, 45, 60, 90]  # degrees
    candidate_dirs = [np.array([np.cos(np.radians(a)), np.sin(np.radians(a))]) 
                      for a in angles]
    
    # True maximum variance direction (from PCA)
    max_var_dir = pc1
    
    # Create figure
    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], alpha=0.4, s=30, color='steelblue')
    
    # Plot candidate directions in gray
    scale = 3
    for direction in candidate_dirs:
        plt.arrow(0, 0, direction[0]*scale, direction[1]*scale,
                 head_width=0.2, head_length=0.15, fc='gray', ec='gray',
                 alpha=0.4, linewidth=1.5, label='_nolegend_')
    
    # Plot maximum variance direction in red
    plt.arrow(0, 0, max_var_dir[0]*scale, max_var_dir[1]*scale,
             head_width=0.25, head_length=0.2, fc='red', ec='red',
             linewidth=2.5, label='Maximum variance direction (PC1)')
    
    # Add some projection lines for the max variance direction
    # Project a few points onto the max variance direction
    sample_indices = [50, 100, 150]
    for idx in sample_indices:
        point = data[idx]
        projection = np.dot(point, max_var_dir) * max_var_dir
        plt.plot([point[0], projection[0]], [point[1], projection[1]], 
                'k--', alpha=0.3, linewidth=0.8, label='_nolegend_')
    
    plt.xlabel('Feature 1', fontsize=12)
    plt.ylabel('Feature 2', fontsize=12)
    plt.title('Finding the Direction of Maximum Variance', fontsize=13)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Save
    filepath = os.path.join(figures_dir, 'fig_variance_directions.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")


# =============================================================================
# Figure 3: fig_covariance_ellipse
# 2D scatter with covariance ellipse and eigenvector arrows
# =============================================================================
def create_fig_covariance_ellipse():
    """Create scatter plot with covariance ellipse and eigenvectors."""
    print("Generating fig_covariance_ellipse...")
    
    # Generate data
    data = generate_correlated_data(n_samples=300)
    
    # Compute PCA
    pca = PCA(n_components=2)
    pca.fit(data)
    
    # Get eigenvectors and eigenvalues
    eigenvectors = pca.components_
    eigenvalues = pca.explained_variance_
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(data[:, 0], data[:, 1], alpha=0.4, s=30, color='steelblue')
    
    # Draw covariance ellipse (2 standard deviations)
    # Calculate angle of first eigenvector
    angle = np.degrees(np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0]))
    width, height = 2 * 2 * np.sqrt(eigenvalues)  # 2 std devs
    ellipse = Ellipse(xy=(0, 0), width=width, height=height, angle=angle,
                     facecolor='none', edgecolor='purple', linewidth=2, 
                     linestyle='--', label='Covariance ellipse')
    ax.add_patch(ellipse)
    
    # Draw eigenvectors scaled by eigenvalues
    colors = ['red', 'blue']
    for i, (eigvec, eigval) in enumerate(zip(eigenvectors, eigenvalues)):
        scale = np.sqrt(eigval) * 2
        ax.arrow(0, 0, eigvec[0]*scale, eigvec[1]*scale,
                head_width=0.2, head_length=0.15, fc=colors[i], ec=colors[i],
                linewidth=2.5, label=f'PC{i+1} (λ={eigval:.2f})')
    
    ax.set_xlabel('Feature 1', fontsize=12)
    ax.set_ylabel('Feature 2', fontsize=12)
    ax.set_title('Covariance Structure and Principal Components', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Save
    filepath = os.path.join(figures_dir, 'fig_covariance_ellipse.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")


# =============================================================================
# Figure 4: fig_variance_explained
# Bar chart showing variance explained by each component
# =============================================================================
def create_fig_variance_explained():
    """Create bar chart of explained variance per component."""
    print("Generating fig_variance_explained...")
    
    # Load digits dataset
    digits = load_digits()
    X = digits.data
    
    # Apply PCA with all components
    pca = PCA()
    pca.fit(X)
    
    # Get explained variance ratios for first 15 components
    n_show = 15
    variance_ratios = pca.explained_variance_ratio_[:n_show]
    components = np.arange(1, n_show + 1)
    
    # Create figure
    plt.figure(figsize=(10, 6))
    bars = plt.bar(components, variance_ratios, color='steelblue', 
                   edgecolor='black', alpha=0.7)
    
    # Add percentage labels on top of first 5 bars
    for i in range(min(5, len(bars))):
        height = bars[i].get_height()
        plt.text(bars[i].get_x() + bars[i].get_width()/2., height,
                f'{variance_ratios[i]:.1%}',
                ha='center', va='bottom', fontsize=9)
    
    # Add reference line
    plt.axhline(y=0.1, color='red', linestyle='--', linewidth=1, 
                alpha=0.5, label='10% threshold')
    
    plt.xlabel('Principal Component', fontsize=12)
    plt.ylabel('Explained Variance Ratio', fontsize=12)
    plt.title('Variance Explained by Each Principal Component', fontsize=13)
    plt.xticks(components)
    plt.grid(True, alpha=0.3, axis='y')
    plt.legend(fontsize=10)
    
    # Save
    filepath = os.path.join(figures_dir, 'fig_variance_explained.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")


# =============================================================================
# Figure 5: fig_cumulative_variance
# Line plot showing cumulative explained variance
# =============================================================================
def create_fig_cumulative_variance():
    """Create line plot of cumulative explained variance."""
    print("Generating fig_cumulative_variance...")
    
    # Load digits dataset
    digits = load_digits()
    X = digits.data
    
    # Apply PCA with all components
    pca = PCA()
    pca.fit(X)
    
    # Calculate cumulative variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.arange(1, len(cumulative_variance) + 1)
    
    # Find where we cross 90% and 95%
    idx_90 = np.argmax(cumulative_variance >= 0.90) + 1
    idx_95 = np.argmax(cumulative_variance >= 0.95) + 1
    
    # Create figure
    plt.figure(figsize=(10, 6))
    plt.plot(n_components, cumulative_variance, 'b-', linewidth=2.5, 
             label='Cumulative variance')
    
    # Add threshold lines
    plt.axhline(y=0.90, color='green', linestyle='--', linewidth=1.5, 
                label='90% threshold')
    plt.axhline(y=0.95, color='orange', linestyle='--', linewidth=1.5, 
                label='95% threshold')
    
    # Add annotations
    plt.plot(idx_90, 0.90, 'go', markersize=8)
    plt.annotate(f'90% at k={idx_90}', 
                xy=(idx_90, 0.90), xytext=(idx_90+5, 0.85),
                fontsize=10, color='green',
                arrowprops=dict(arrowstyle='->', color='green', lw=1.5))
    
    plt.plot(idx_95, 0.95, 'o', color='orange', markersize=8)
    plt.annotate(f'95% at k={idx_95}', 
                xy=(idx_95, 0.95), xytext=(idx_95+5, 0.98),
                fontsize=10, color='orange',
                arrowprops=dict(arrowstyle='->', color='orange', lw=1.5))
    
    plt.xlabel('Number of Components', fontsize=12)
    plt.ylabel('Cumulative Explained Variance', fontsize=12)
    plt.title('Cumulative Variance Explained by Principal Components', fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.xlim(0, len(n_components) + 1)
    plt.ylim(0, 1.05)
    
    # Save
    filepath = os.path.join(figures_dir, 'fig_cumulative_variance.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")


# =============================================================================
# Figure 6: fig_2d_projection
# 2D scatter of digits dataset in PC1-PC2 space
# =============================================================================
def create_fig_2d_projection():
    """Create 2D PCA projection of digits dataset."""
    print("Generating fig_2d_projection...")
    
    # Load digits dataset
    digits = load_digits()
    X = digits.data
    y = digits.target
    
    # Apply PCA to 2 dimensions
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='tab10',
                         alpha=0.6, s=40, edgecolors='k', linewidth=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ticks=range(10))
    cbar.set_label('Digit Class', fontsize=11)
    
    # Calculate variance explained
    var_pc1 = pca.explained_variance_ratio_[0]
    var_pc2 = pca.explained_variance_ratio_[1]
    
    plt.xlabel(f'First Principal Component (PC1: {var_pc1:.1%} variance)', 
              fontsize=12)
    plt.ylabel(f'Second Principal Component (PC2: {var_pc2:.1%} variance)', 
              fontsize=12)
    plt.title('Digits Dataset: 2D PCA Projection', fontsize=13)
    plt.grid(True, alpha=0.3)
    
    # Save
    filepath = os.path.join(figures_dir, 'fig_2d_projection.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")


# =============================================================================
# BONUS Figure: fig_reconstruction (mentioned in lecture notes)
# Side-by-side comparison of original vs reconstructed digits
# =============================================================================
def create_fig_reconstruction():
    """Create comparison of original vs reconstructed digit images."""
    print("Generating fig_reconstruction (bonus)...")
    
    # Load digits dataset
    digits = load_digits()
    X = digits.data
    
    # Apply PCA with 20 components
    pca = PCA(n_components=20)
    X_reduced = pca.fit_transform(X)
    X_reconstructed = pca.inverse_transform(X_reduced)
    
    # Select a few examples to show
    n_examples = 10
    indices = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    
    # Create figure with two rows
    fig, axes = plt.subplots(2, n_examples, figsize=(12, 3))
    
    for i, idx in enumerate(indices):
        # Original
        axes[0, i].imshow(X[idx].reshape(8, 8), cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original', fontsize=10, loc='left')
        
        # Reconstructed
        axes[1, i].imshow(X_reconstructed[idx].reshape(8, 8), cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Reconstructed\n(20 components)', 
                                fontsize=10, loc='left')
    
    plt.suptitle('Original vs Reconstructed Digits (PCA with 20 components)', 
                fontsize=13, y=1.02)
    plt.tight_layout()
    
    # Save
    filepath = os.path.join(figures_dir, 'fig_reconstruction.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")


# =============================================================================
# Main execution
# =============================================================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("Generating PCA Lecture Figures")
    print("="*60 + "\n")
    
    # Generate all figures
    create_fig_pca_intuition()
    create_fig_variance_directions()
    create_fig_covariance_ellipse()
    create_fig_variance_explained()
    create_fig_cumulative_variance()
    create_fig_2d_projection()
    create_fig_reconstruction()  # Bonus figure
    
    print("\n" + "="*60)
    print("All figures generated successfully!")
    print(f"Location: {figures_dir}")
    print("="*60 + "\n")