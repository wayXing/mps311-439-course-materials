"""
Figure Generation Script for Lesson 8: K-means and Hierarchical Clustering
Generates all figures used in the lecture notes.

Author: Dr. Wei Xing
Course: MPS311/439 Machine Learning
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Set random seed for reproducibility
np.random.seed(42)

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
figure_dir = os.path.join(script_dir, 'figures')

# Create figure directory if it doesn't exist
os.makedirs(figure_dir, exist_ok=True)

print(f"Saving figures to: {figure_dir}")

# Set style for consistent, professional-looking figures
plt.style.use('seaborn-v0_8-darkgrid')
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']


# ============================================================================
# Figure 1: Clustering Challenge - Unlabeled data points
# ============================================================================
print("Generating Figure 1: clustering_challenge.png...")

fig, ax = plt.subplots(figsize=(8, 6))

# Generate data with 3 clear clusters
X_challenge, _ = make_blobs(n_samples=300, centers=3, n_features=2, 
                             cluster_std=0.8, random_state=42)

# Plot all points in the same color (unlabeled)
ax.scatter(X_challenge[:, 0], X_challenge[:, 1], 
           c='steelblue', s=50, alpha=0.7, edgecolors='k', linewidth=0.5)

ax.set_xlabel('Feature 1 (e.g., Annual Spending)', fontsize=12)
ax.set_ylabel('Feature 2 (e.g., Visit Frequency)', fontsize=12)
ax.set_title('The Clustering Challenge: Can You Identify Natural Groups?', 
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(figure_dir, 'clustering_challenge.png'), 
            dpi=300, bbox_inches='tight')
plt.close()


# ============================================================================
# Figure 2: K-means Iterations - Show algorithm progress
# ============================================================================
print("Generating Figure 2: kmeans_iterations.png...")

# Generate data
X_iter, _ = make_blobs(n_samples=300, centers=3, n_features=2,
                       cluster_std=0.7, random_state=42)

# Custom K-means to track iterations
def kmeans_track_iterations(X, K, max_iters=20):
    """K-means with iteration tracking"""
    n, d = X.shape
    
    # Initialize randomly
    idx = np.random.choice(n, K, replace=False)
    centroids = X[idx].copy()
    
    history = {'centroids': [centroids.copy()], 'labels': []}
    
    for i in range(max_iters):
        # Assignment
        distances = np.sqrt(((X[:, np.newaxis, :] - centroids) ** 2).sum(axis=2))
        labels = np.argmin(distances, axis=1)
        history['labels'].append(labels.copy())
        
        # Update
        new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(K)])
        
        history['centroids'].append(new_centroids.copy())
        
        # Check convergence
        if np.allclose(centroids, new_centroids):
            break
            
        centroids = new_centroids
    
    return history

# Run K-means with tracking
np.random.seed(42)
history = kmeans_track_iterations(X_iter, K=3, max_iters=20)

# Create 4-panel figure
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

iterations_to_show = [0, 1, 4, len(history['centroids']) - 1]
titles = ['Iteration 1: Initial Random Centroids', 
          'Iteration 2: First Update',
          'Iteration 5: Clusters Forming',
          f'Final: Converged (Iteration {len(history["centroids"])})']

for idx, (iter_num, title) in enumerate(zip(iterations_to_show, titles)):
    ax = axes[idx]

    # Clamp iter_num to valid range for centroids
    iter_num_cent = min(iter_num, len(history['centroids']) - 1)
    centroids = history['centroids'][iter_num_cent]

    # For labels, use min to avoid IndexError (labels are always one less than centroids)
    if iter_num > 0 and iter_num - 1 < len(history['labels']):
        labels = history['labels'][iter_num - 1]
        # Plot points colored by cluster
        for k in range(3):
            mask = labels == k
            ax.scatter(X_iter[mask, 0], X_iter[mask, 1],
                       c=[colors[k]], s=50, alpha=0.6,
                       edgecolors='k', linewidth=0.5)
    else:
        # First iteration: all points same color
        ax.scatter(X_iter[:, 0], X_iter[:, 1],
                   c='lightgray', s=50, alpha=0.6,
                   edgecolors='k', linewidth=0.5)

    # Plot centroids
    ax.scatter(centroids[:, 0], centroids[:, 1],
               marker='X', s=400, c='red',
               edgecolors='black', linewidth=2,
               label='Centroids', zorder=5)

    ax.set_xlabel('Feature 1', fontsize=10)
    ax.set_ylabel('Feature 2', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

plt.suptitle('K-means Algorithm: Iterative Convergence', 
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(os.path.join(figure_dir, 'kmeans_iterations.png'), 
            dpi=300, bbox_inches='tight')
plt.close()


# ============================================================================
# Figure 3: K-means Failures - When the algorithm struggles
# ============================================================================
print("Generating Figure 3: kmeans_failures.png...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left panel: Non-convex shapes (moons)
X_moons, y_moons = make_moons(n_samples=300, noise=0.08, random_state=42)
kmeans_moons = KMeans(n_clusters=2, random_state=42, n_init=10)
labels_moons = kmeans_moons.fit_predict(X_moons)

axes[0].scatter(X_moons[:, 0], X_moons[:, 1], 
                c=labels_moons, cmap='viridis', s=50, 
                alpha=0.7, edgecolors='k', linewidth=0.5)
axes[0].scatter(kmeans_moons.cluster_centers_[:, 0],
                kmeans_moons.cluster_centers_[:, 1],
                marker='X', s=400, c='red', edgecolors='black', 
                linewidth=2, zorder=5)
axes[0].set_xlabel('Feature 1', fontsize=11)
axes[0].set_ylabel('Feature 2', fontsize=11)
axes[0].set_title('Failure Case 1: Non-Convex Clusters (Half-Moons)', 
                  fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].text(0.5, -1.2, 'K-means incorrectly splits the crescent shapes',
             ha='center', fontsize=10, style='italic',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Right panel: Different sizes and densities
# Create clusters with very different characteristics
np.random.seed(42)
X1 = np.random.randn(200, 2) * 0.3 + np.array([0, 0])  # Small, dense
X2 = np.random.randn(50, 2) * 0.3 + np.array([3, 3])   # Small, dense
X3 = np.random.randn(300, 2) * 1.5 + np.array([5, -2])  # Large, sparse
X_varied = np.vstack([X1, X2, X3])
y_varied = np.array([0]*200 + [1]*50 + [2]*300)

kmeans_varied = KMeans(n_clusters=3, random_state=42, n_init=10)
labels_varied = kmeans_varied.fit_predict(X_varied)

axes[1].scatter(X_varied[:, 0], X_varied[:, 1], 
                c=labels_varied, cmap='viridis', s=50,
                alpha=0.7, edgecolors='k', linewidth=0.5)
axes[1].scatter(kmeans_varied.cluster_centers_[:, 0],
                kmeans_varied.cluster_centers_[:, 1],
                marker='X', s=400, c='red', edgecolors='black',
                linewidth=2, zorder=5)
axes[1].set_xlabel('Feature 1', fontsize=11)
axes[1].set_ylabel('Feature 2', fontsize=11)
axes[1].set_title('Failure Case 2: Different Sizes and Densities',
                  fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].text(2.5, -6, 'K-means assumes similar variance across clusters',
             ha='center', fontsize=10, style='italic',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('When K-means Fails: Understanding Limitations',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(figure_dir, 'kmeans_failures.png'),
            dpi=300, bbox_inches='tight')
plt.close()


# ============================================================================
# Figure 4: Elbow Plot - Choosing the optimal K
# ============================================================================
print("Generating Figure 4: elbow_plot.png...")

# Generate data with clear 3 clusters
X_elbow, _ = make_blobs(n_samples=400, centers=3, n_features=2,
                        cluster_std=0.9, random_state=42)

# Compute inertia for different K values
K_range = range(1, 11)
inertias = []

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_elbow)
    inertias.append(kmeans.inertia_)

# Create the elbow plot
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(K_range, inertias, marker='o', linewidth=2, markersize=10,
        color='steelblue', markerfacecolor='orange', markeredgewidth=2,
        markeredgecolor='steelblue')

# Highlight the elbow point (K=3)
ax.plot(3, inertias[2], marker='o', markersize=20, 
        color='red', markeredgewidth=3, markeredgecolor='darkred',
        zorder=5)
ax.annotate('Elbow Point\n(Optimal K = 3)', 
            xy=(3, inertias[2]), xytext=(5, inertias[2] + 300),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=12, fontweight='bold', color='red',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

ax.set_xlabel('Number of Clusters (K)', fontsize=12, fontweight='bold')
ax.set_ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=12, fontweight='bold')
ax.set_title('The Elbow Method: Choosing Optimal K', fontsize=14, fontweight='bold')
ax.set_xticks(K_range)
ax.grid(True, alpha=0.3, linestyle='--')

# Add text explanation
ax.text(7, inertias[0] * 0.9, 
        'Steep decrease:\nCapturing real structure',
        fontsize=10, ha='center',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.6))
ax.text(7, inertias[-1] * 3, 
        'Gradual decrease:\nDiminishing returns',
        fontsize=10, ha='center',
        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.6))

plt.tight_layout()
plt.savefig(os.path.join(figure_dir, 'elbow_plot.png'),
            dpi=300, bbox_inches='tight')
plt.close()


# ============================================================================
# Figure 5: Dendrogram - Hierarchical clustering visualization
# ============================================================================
print("Generating Figure 5: dendrogram.png...")

# Generate data with clear hierarchical structure
np.random.seed(42)
X_hier, _ = make_blobs(n_samples=50, centers=3, n_features=2,
                       cluster_std=0.6, random_state=42)

# Compute linkage matrix
Z = linkage(X_hier, method='average')

# Create dendrogram
fig, ax = plt.subplots(figsize=(12, 6))

# Plot dendrogram with customization
dendrogram(Z, ax=ax, color_threshold=5, above_threshold_color='gray')

ax.set_xlabel('Data Point Index', fontsize=12, fontweight='bold')
ax.set_ylabel('Distance (Dissimilarity)', fontsize=12, fontweight='bold')
ax.set_title('Hierarchical Clustering Dendrogram', fontsize=14, fontweight='bold')

# Add horizontal lines to show where to cut for different K
ax.axhline(y=5, color='red', linestyle='--', linewidth=2, label='Cut for K=3 clusters')
ax.axhline(y=2.5, color='blue', linestyle='--', linewidth=2, label='Cut for K=5 clusters')

# Add annotations
ax.annotate('Cutting here gives 3 clusters', 
            xy=(25, 5), xytext=(25, 6.5),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=10, color='red', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='red', alpha=0.8))

ax.annotate('Cutting here gives 5 clusters', 
            xy=(35, 2.5), xytext=(35, 4),
            arrowprops=dict(arrowstyle='->', color='blue', lw=2),
            fontsize=10, color='blue', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='blue', alpha=0.8))

# Add text box explaining dendrograms
textstr = 'Reading Dendrograms:\n• Y-axis: distance at merge\n• Vertical lines: merged clusters\n• Cut horizontally for K clusters'
props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

ax.legend(loc='upper right', fontsize=10)
ax.grid(True, axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(figure_dir, 'dendrogram.png'),
            dpi=300, bbox_inches='tight')
plt.close()


# ============================================================================
# Print completion message
# ============================================================================
print("\n" + "="*70)
print("All figures generated successfully!")
print("="*70)
print(f"\nFigures saved in: {figure_dir}")
print("\nGenerated figures:")
print("  1. clustering_challenge.png - The clustering problem setup")
print("  2. kmeans_iterations.png - K-means convergence process")
print("  3. kmeans_failures.png - When K-means fails")
print("  4. elbow_plot.png - Choosing optimal K")
print("  5. dendrogram.png - Hierarchical clustering tree")
print("\n" + "="*70)