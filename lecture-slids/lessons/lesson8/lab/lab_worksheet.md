---
pagetitle: "Lab 8: K-means and Hierarchical Clustering"
---

# Lab 8: K-means and Hierarchical Clustering
## MPS311/439 - Machine Learning
**Lesson 8 Lab Session | Duration: 50 minutes**

---

## Introduction

Welcome to Lab 8! Today you'll explore **Clustering** - finding natural groups in data without labels. This is unsupervised learning at its finest!

**What you'll learn today:**
- How to apply K-means clustering using sklearn
- How to find the optimal number of clusters (elbow method)
- How to visualize high-dimensional clusters
- How to use hierarchical clustering and dendrograms
- When to use K-means vs hierarchical clustering

**Remember:** This lab uses a fill-in-the-blanks approach. Don't write code from scratch - just fill in the blanks marked with `____`. Focus on understanding the concepts!

---

## Setup: Import Libraries and Load Data

We'll use the **Iris dataset** - it has 150 flowers with 4 measurements each. Although we know there are 3 species, we'll pretend we don't and use clustering to discover them!

**IMPORTANT:** Clustering uses distances, so we must standardize our features first!

**Copy and run this code:**

```python
# Import packages
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage

# Load iris dataset
iris = load_iris()
X = iris.data
y_true = iris.target  # True labels (we won't use these for clustering!)

# CRITICAL: Standardize features before clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Samples: {X.shape[0]}")
print(f"Features: {X.shape[1]}")
print(f"Feature names: {iris.feature_names}")
print(f"True number of species: {len(np.unique(y_true))}")
print(f"\nFeatures standardized! Ready for clustering.")
```

**What's happening here?**
- We have 150 flowers with 4 measurements each
- We standardized so all features have mean=0 and std=1
- This ensures no feature dominates due to its scale
- We know there are 3 true species, but clustering will discover them

---

## Part 1: Your First K-means Clustering (8 minutes)

**Background:** K-means groups data into K clusters by iteratively:
1. Assigning each point to its nearest centroid
2. Updating centroids as the mean of assigned points
3. Repeating until convergence

Let's start with K=3 (since we suspect 3 species exist).

### Task 1.1: Create and fit K-means

**Fill in the blanks below:**

```python
# Create K-means with 3 clusters
kmeans = KMeans(n_clusters=____, random_state=42, n_init=10)

# Fit the model and get cluster assignments
clusters = kmeans.____(X_scaled)

print(f"Cluster assignments: {clusters}")
print(f"Unique clusters: {np.unique(clusters)}")
print(f"Cluster sizes: {np.bincount(clusters)}")
```

**Hints:**
- Use `n_clusters=3`
- Method to fit and predict: `.fit_predict(X_scaled)`

**AI Help:** Ask ChatGPT: "How do I use KMeans from sklearn to cluster data?"

### Task 1.2: Examine cluster centers

**Fill in the blanks:**

```python
# Get the centroids
centroids = kmeans.____

print(f"\nCentroid positions (standardized features):")
print(centroids)
print(f"Centroids shape: {centroids.shape}")
```

**Hint:**
- Attribute name: `.cluster_centers_`

### Task 1.3: Compute inertia

**Fill in the blanks:**

```python
# Inertia = sum of squared distances to nearest centroid
inertia = kmeans.____

print(f"\nInertia (within-cluster sum of squares): {inertia:.2f}")
```

**Hint:**
- Attribute name: `.inertia_`

**Question:** What does inertia measure? Lower is better or worse?

```
Your answer: _______________________________________________
```

---

## Part 2: Finding the Optimal K (Elbow Method) (10 minutes)

**Background:** We chose K=3 because we suspected 3 species. But what if we didn't know? The **elbow method** helps us find the optimal K by plotting inertia vs K and looking for the "elbow" where improvement slows down.

### Task 2.1: Compute inertia for different K values

**Fill in the blanks:**

```python
# Try K from 1 to 10
K_range = range(1, 11)
inertias = []

for k in K_range:
    kmeans_temp = KMeans(n_clusters=____, random_state=42, n_init=10)
    kmeans_temp.fit(____)
    inertias.append(kmeans_temp.____)

print("K values:", list(K_range))
print("Inertias:", [f"{i:.2f}" for i in inertias])
```

**Hints:**
- Use the loop variable `k` for n_clusters
- Fit on `X_scaled`
- Append the inertia: `.inertia_`

**AI Help:** Ask ChatGPT: "What is the elbow method for choosing K in K-means?"

### Task 2.2: Plot the elbow curve

**Copy and run this code:**

```python
# Plot elbow curve
plt.figure(figsize=(8, 5))
plt.plot(K_range, inertias, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.grid(True)
plt.show()
```

### Task 2.3: Identify the elbow

Look at your plot carefully. The "elbow" is where the curve bends - where adding more clusters gives diminishing returns.

**Questions:**

1. At which K value does the elbow appear?
```
K = ____
```

2. Does this match the true number of species (3)?
```
Answer: _______________________________________________
```

3. What happens to inertia as K increases?
```
Answer: _______________________________________________
```

**Key insight:** Inertia always decreases as K increases, but we want the K where it stops decreasing rapidly!

---

## Part 3: Visualizing Clusters in 2D (8 minutes)

**Background:** Our data has 4 dimensions (features), so we can't plot it directly. We'll use **PCA** to reduce it to 2D for visualization, then color points by their cluster.

### Task 3.1: Use PCA to reduce dimensions

**Fill in the blanks:**

```python
# Reduce to 2 dimensions using PCA
pca = PCA(n_components=____)
X_pca = pca.____(X_scaled)

print(f"Original dimensions: {X_scaled.shape}")
print(f"Reduced dimensions: {X_pca.shape}")
print(f"Variance explained: {pca.explained_variance_ratio_}")
```

**Hints:**
- Use `n_components=2`
- Method to transform: `.fit_transform(X_scaled)`

**AI Help:** Ask ChatGPT: "How do I use PCA to reduce data to 2D for visualization?"

### Task 3.2: Plot clusters

**Fill in the blanks:**

```python
# Plot the clusters
plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, ____], X_pca[:, ____], c=____, cmap='viridis', s=50)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('K-means Clustering (K=3) - PCA Visualization')
plt.colorbar(label='Cluster')
plt.show()
```

**Hints:**
- First column: `0`
- Second column: `1`
- Color by cluster assignments: `clusters`

### Task 3.3: Plot with true labels for comparison

**Copy and run this code:**

```python
# Compare with true labels
plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_true, cmap='viridis', s=50)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('True Species Labels - PCA Visualization')
plt.colorbar(label='Species')
plt.show()
```

**Question:** Do the K-means clusters match the true species well? Where do they differ?

```
Your answer: _______________________________________________
___________________________________________________________
```

---

## Part 4: Hierarchical Clustering (10 minutes)

**Background:** Hierarchical clustering builds a tree (dendrogram) showing how clusters merge. Unlike K-means, we don't need to choose K upfront - we can explore different numbers of clusters by cutting the tree at different heights!

### Task 4.1: Create a dendrogram

**Fill in the blanks:**

```python
# Compute linkage matrix
Z = linkage(X_scaled, method='____')

# Plot dendrogram
plt.figure(figsize=(12, 6))
dendrogram(____, no_labels=True)
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.title('Hierarchical Clustering Dendrogram (Average Linkage)')
plt.show()
```

**Hints:**
- Use `method='average'` for balanced clustering
- Pass the linkage matrix `Z` to dendrogram

**AI Help:** Ask ChatGPT: "How do I create a dendrogram for hierarchical clustering in Python?"

### Task 4.2: Extract clusters from dendrogram

**Fill in the blanks:**

```python
# Use hierarchical clustering to get 3 clusters
hc = AgglomerativeClustering(n_clusters=____, linkage='average')
clusters_hc = hc.____(X_scaled)

print(f"Hierarchical cluster assignments: {clusters_hc}")
print(f"Cluster sizes: {np.bincount(clusters_hc)}")
```

**Hints:**
- Use `n_clusters=3`
- Method to fit and predict: `.fit_predict(X_scaled)`

### Task 4.3: Visualize hierarchical clusters

**Fill in the blanks:**

```python
# Plot hierarchical clustering results
plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=____, cmap='viridis', s=50)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('Hierarchical Clustering (K=3) - PCA Visualization')
plt.colorbar(label='Cluster')
plt.show()
```

**Hint:**
- Color by hierarchical cluster assignments: `clusters_hc`

**Question:** Do hierarchical clusters look different from K-means clusters?

```
Your answer: _______________________________________________
```

---

## Part 5: K-means vs Hierarchical Comparison (8 minutes)

**Background:** Both methods can give different results. Let's compare them and understand when to use each.

### Task 5.1: Count agreement between methods

**Fill in the blanks:**

```python
# How often do both methods assign same cluster?
# Note: cluster labels might be permuted, so we check if points
# that are together in K-means are also together in hierarchical

# Simple check: are cluster sizes similar?
print("K-means cluster sizes:", np.bincount(____))
print("Hierarchical cluster sizes:", np.bincount(clusters_hc))
```

**Hint:**
- Use `clusters` for K-means assignments

### Task 5.2: Side-by-side comparison

**Copy and run this code:**

```python
# Plot both side by side
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', s=50)
axes[0].set_xlabel('PC1')
axes[0].set_ylabel('PC2')
axes[0].set_title('K-means (K=3)')

axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=clusters_hc, cmap='viridis', s=50)
axes[1].set_xlabel('PC1')
axes[1].set_ylabel('PC2')
axes[1].set_title('Hierarchical (K=3)')

plt.tight_layout()
plt.show()
```

### Task 5.3: Try different linkage methods

**Fill in the blanks:**

```python
# Try single linkage
hc_single = AgglomerativeClustering(n_clusters=3, linkage='____')
clusters_single = hc_single.fit_predict(X_scaled)

# Try complete linkage
hc_complete = AgglomerativeClustering(n_clusters=3, linkage='____')
clusters_complete = hc_complete.fit_predict(____)

# Plot all three linkage methods
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=clusters_single, cmap='viridis', s=50)
axes[0].set_title('Single Linkage')

axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=clusters_hc, cmap='viridis', s=50)
axes[1].set_title('Average Linkage')

axes[2].scatter(X_pca[:, 0], X_pca[:, 1], c=clusters_complete, cmap='viridis', s=50)
axes[2].set_title('Complete Linkage')

plt.tight_layout()
plt.show()
```

**Hints:**
- Use `linkage='single'` and `linkage='complete'`
- Fit on `X_scaled`

**AI Help:** Ask ChatGPT: "What are the differences between single, complete, and average linkage in hierarchical clustering?"

**Questions:**

1. Which linkage method gives the most similar results to K-means?
```
Answer: _______________________________________________
```

2. Do you see any major differences between linkage methods?
```
Answer: _______________________________________________
```

---

## Reflection Questions (4 minutes)

Answer these questions based on what you've learned today:

**Question 1:** Why did we standardize the features before clustering? What would happen if we didn't?

```
Your answer: _______________________________________________
___________________________________________________________
```

**Question 2:** The elbow method showed K=3 is optimal. If the elbow was at K=5, what would that mean?

```
Your answer: _______________________________________________
___________________________________________________________
```

**Question 3:** When would you choose K-means over hierarchical clustering in a real project?

```
Your answer: _______________________________________________
___________________________________________________________
```

**Question 4:** You have a dataset with 100,000 samples. Which clustering method would you use and why?

```
Your answer: _______________________________________________
___________________________________________________________
```

---

## Summary

**What you learned today:**
- ✅ How to apply K-means clustering using sklearn
- ✅ How to use the elbow method to find optimal K
- ✅ How to visualize high-dimensional clusters using PCA
- ✅ How to create dendrograms and use hierarchical clustering
- ✅ How to compare different clustering methods

**Key takeaways:**
- Always standardize features before clustering!
- K-means is fast but needs K specified upfront
- Hierarchical builds a tree and lets you explore different K values
- The elbow method helps choose K, but use domain knowledge too
- Different methods can give different results - try multiple approaches!

---

## Part 6: Advanced Challenge - For MPS439 Students Only (30 minutes)

**Background:** Now you'll implement K-means from scratch to understand exactly how it works! You'll write the core algorithm using only NumPy.

### Task 6.1: Initialize centroids

The first step is to randomly select K data points as initial centroids.

**Fill in the blanks:**

```python
def initialize_centroids(X, K):
    """
    Randomly select K data points as initial centroids.
    
    X: array of shape (n, d)
    K: number of clusters
    Returns: array of shape (K, d)
    """
    n = X.shape[0]
    indices = np.random.choice(____, K, replace=False)
    centroids = X[____]
    return centroids

# Test it
np.random.seed(42)
initial_centroids = initialize_centroids(X_scaled, 3)
print("Initial centroids shape:", initial_centroids.shape)
print("Initial centroids:\n", initial_centroids)
```

**Hints:**
- Choose from `n` samples
- Index X with `indices`

### Task 6.2: Compute distances

Now compute the distance from each point to each centroid.

**Fill in the blanks:**

```python
def compute_distances(X, centroids):
    """
    Compute Euclidean distance from each point to each centroid.
    
    X: array of shape (n, d)
    centroids: array of shape (K, d)
    Returns: array of shape (n, K)
    """
    # Expand dimensions for broadcasting
    X_expanded = X[:, np.newaxis, :]  # Shape: (n, 1, d)
    centroids_expanded = ____         # Already shape: (K, d)
    
    # Compute squared differences
    diff = X_expanded - centroids_expanded  # Shape: (n, K, d)
    
    # Sum over features and take square root
    distances = np.sqrt(np.sum(____**2, axis=2))
    
    return distances

# Test it
distances = compute_distances(X_scaled, initial_centroids)
print("Distances shape:", distances.shape)
print("First 5 distances:\n", distances[:5])
```

**Hints:**
- `centroids` is already the right shape
- Square the `diff` array
- Sum over axis=2 (features)

**AI Help:** Ask ChatGPT: "How does NumPy broadcasting work for computing pairwise distances?"

### Task 6.3: Assignment step

Assign each point to its nearest centroid.

**Fill in the blanks:**

```python
def assign_clusters(distances):
    """
    Assign each point to nearest centroid.
    
    distances: array of shape (n, K)
    Returns: array of shape (n,) with values in {0, 1, ..., K-1}
    """
    return np.argmin(____, axis=1)

# Test it
labels = assign_clusters(distances)
print("Cluster assignments:", labels)
print("Cluster sizes:", np.bincount(labels))
```

**Hint:**
- Find minimum along axis=1 (across centroids)

### Task 6.4: Update step

Recompute centroids as the mean of assigned points.

**Fill in the blanks:**

```python
def update_centroids(X, labels, K):
    """
    Recompute centroids as mean of assigned points.
    
    X: array of shape (n, d)
    labels: array of shape (n,)
    K: number of clusters
    Returns: array of shape (K, d)
    """
    n, d = X.shape
    centroids = np.zeros((K, d))
    
    for k in range(K):
        # Get points assigned to cluster k
        cluster_points = X[____ == k]
        
        if len(cluster_points) > 0:
            centroids[k] = cluster_points.mean(axis=____)
        else:
            # Empty cluster: reinitialize randomly
            centroids[k] = X[np.random.randint(n)]
    
    return centroids

# Test it
new_centroids = update_centroids(X_scaled, labels, 3)
print("Updated centroids shape:", new_centroids.shape)
print("Updated centroids:\n", new_centroids)
```

**Hints:**
- Filter where `labels == k`
- Take mean over axis=0 (average over points)

### Task 6.5: Check convergence

Determine if centroids have stopped moving.

**Fill in the blanks:**

```python
def has_converged(old_centroids, new_centroids, tol=1e-4):
    """
    Check if centroids have converged.
    
    old_centroids: array of shape (K, d)
    new_centroids: array of shape (K, d)
    tol: convergence tolerance
    Returns: bool
    """
    return np.allclose(____, ____, atol=tol)

# Test it
converged = has_converged(initial_centroids, new_centroids)
print("Has converged?", converged)
```

**Hints:**
- Compare `old_centroids` and `new_centroids`

### Task 6.6: Put it all together

Now combine all functions into the complete K-means algorithm!

**Fill in the blanks:**

```python
def kmeans_from_scratch(X, K, max_iters=100, tol=1e-4, random_state=None):
    """
    Complete K-means implementation.
    
    X: array of shape (n, d)
    K: number of clusters
    max_iters: maximum iterations
    tol: convergence tolerance
    Returns: labels, centroids, num_iters
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Initialize
    centroids = initialize_centroids(____, ____)
    
    for iteration in range(max_iters):
        # Assignment step
        distances = compute_distances(X, ____)
        labels = assign_clusters(____)
        
        # Update step
        new_centroids = update_centroids(X, ____, K)
        
        # Check convergence
        if has_converged(centroids, ____):
            print(f"Converged after {iteration + 1} iterations")
            return labels, new_centroids, iteration + 1
        
        centroids = ____
    
    print(f"Max iterations ({max_iters}) reached")
    return labels, centroids, max_iters

# Run your implementation
labels_scratch, centroids_scratch, num_iters = kmeans_from_scratch(
    X_scaled, K=3, random_state=42
)

print(f"\nYour K-means converged in {num_iters} iterations")
print("Cluster sizes:", np.bincount(labels_scratch))
```

**Hints:**
- Initialize with `X` and `K`
- Use `centroids` for distance computation
- Use `distances` for assignment
- Use `labels` for update
- Check convergence with `new_centroids`
- Update `centroids` to `new_centroids`

### Task 6.7: Compare with sklearn

**Fill in the blanks:**

```python
# Run sklearn K-means for comparison
kmeans_sklearn = KMeans(n_clusters=3, random_state=42, n_init=10, max_iter=100)
labels_sklearn = kmeans_sklearn.____(X_scaled)

# Compute inertia for your implementation
def compute_inertia(X, labels, centroids):
    """Compute within-cluster sum of squares."""
    inertia = 0
    for k in range(len(centroids)):
        cluster_points = X[labels == k]
        if len(cluster_points) > 0:
            inertia += np.sum((cluster_points - centroids[____])**2)
    return inertia

inertia_scratch = compute_inertia(X_scaled, labels_scratch, centroids_scratch)
inertia_sklearn = kmeans_sklearn.____

print(f"\nYour inertia: {inertia_scratch:.4f}")
print(f"Sklearn inertia: {inertia_sklearn:.4f}")
print(f"Difference: {abs(inertia_scratch - inertia_sklearn):.6f}")
```

**Hints:**
- Fit sklearn on `X_scaled`: `.fit_predict(X_scaled)`
- Use centroid index `k`
- Get sklearn inertia: `.inertia_`

### Task 6.8: Visualize your results

**Copy and run this code:**

```python
# Compare visually
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=labels_scratch, cmap='viridis', s=50)
axes[0].set_title(f'Your K-means (inertia={inertia_scratch:.2f})')
axes[0].set_xlabel('PC1')
axes[0].set_ylabel('PC2')

axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=labels_sklearn, cmap='viridis', s=50)
axes[1].set_title(f'Sklearn K-means (inertia={inertia_sklearn:.2f})')
axes[1].set_xlabel('PC1')
axes[1].set_ylabel('PC2')

plt.tight_layout()
plt.show()
```

### Task 6.9: Advanced reflection

**Question 1:** Why do we need to handle empty clusters in the update step?

```
Your answer: _______________________________________________
___________________________________________________________
```

**Question 2:** Your implementation and sklearn might give slightly different results. Why?

```
Your answer: _______________________________________________
___________________________________________________________
```

**Question 3:** What is the time complexity of your K-means implementation? (Hint: Consider the distance computation step)

```
Your answer: _______________________________________________
___________________________________________________________
```

**Question 4:** How would you modify your implementation to use Manhattan distance instead of Euclidean distance?

```
Your answer: _______________________________________________
___________________________________________________________
```

---

**Congratulations!** You've implemented K-means from scratch and understand exactly how it works under the hood. This is the same algorithm that sklearn uses (with some optimizations)!

---

**End of Lab Worksheet**

<!-- COURSE_FEEDBACK_QR:START -->
---

## 30-second feedback

What would help you learn better next time?

Scan to share anonymous feedback on today's lab.

![Feedback QR code for Lesson 08 lab](./feedback-qr.png){fig-align="center" width="180px"}
<!-- COURSE_FEEDBACK_QR:END -->
