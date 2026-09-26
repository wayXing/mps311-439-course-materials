---
pagetitle: "Lab 7: Principal Component Analysis"
---

# Lab 7: Principal Component Analysis
## MPS311/439 - Machine Learning
**Lesson 7 Lab Session | Duration: 50 minutes**

***

## Introduction

Welcome to Lab 7! Today you'll explore **Principal Component Analysis (PCA)** - a powerful technique for reducing dimensionality and understanding the structure of high-dimensional data.

**What you'll learn today:**
- How to apply PCA to reduce dimensions while preserving information
- How to choose the optimal number of components
- How to visualize high-dimensional data in 2D
- How PCA enables image compression and reconstruction
- The trade-off between compression and information loss

**Remember:** This lab uses a fill-in-the-blanks approach. Don't write code from scratch - just fill in the blanks marked with `____`. Focus on understanding the concepts!

***

## Setup: Import Libraries and Load Data

We'll use the **digits dataset** - handwritten digit images (0-9). Each image is 8×8 pixels = 64 dimensions. This is perfect for seeing how PCA reduces dimensionality!

**Copy and run this code:**

```python
# Import packages
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

# Load digits dataset
digits = load_digits()
X = digits.data
y = digits.target

print(f"Number of samples: {X.shape[0]}")
print(f"Number of features (pixels): {X.shape[1]}")
print(f"Feature shape: Each image is {int(np.sqrt(X.shape[1]))}x{int(np.sqrt(X.shape[1]))} pixels")
print(f"Classes: {np.unique(y)}")

# Visualize a few digits
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.flat):
    ax.imshow(X[i].reshape(8, 8), cmap='gray')
    ax.set_title(f'Label: {y[i]}')
    ax.axis('off')
plt.tight_layout()
plt.show()
```

**What's happening here?**
- We have 1,797 images, each with 64 features (8×8 pixels)
- Each image is a handwritten digit (0-9)
- We'll use PCA to reduce from 64 dimensions to much fewer!

***

## Part 1: Your First PCA (8 minutes)

**Background:** PCA finds new axes (principal components) that capture maximum variance in the data. The first principal component (PC1) captures the most variance, PC2 captures the second-most, and so on. Let's apply PCA and see what happens!

### Task 1.1: Apply PCA to reduce dimensions

**Fill in the blanks below:**

```python
# Create PCA model to keep top 20 components
pca = PCA(n_components=____)

# Fit and transform the data
X_reduced = pca.____(____)

# Check the new shape
print(f"Original shape: {X.shape}")
print(f"Reduced shape: {____}")
```

**Hints:**
- Use 20 components
- Method to fit and transform: `.fit_transform(X)`
- Access shape: `X_reduced.shape`

**AI Help:** Ask ChatGPT: "How do I use PCA in sklearn to reduce dimensions?"

### Task 1.2: Examine variance explained

Each principal component captures some portion of the total variance. Let's see how much!

**Fill in the blanks:**

```python
# Get variance explained by each component
variance_ratios = pca.____

# Print first 5 components
print("Variance explained by first 5 components:")
for i in range(5):
    print(f"PC{i+1}: {variance_ratios[____]:.3f}")

# Total variance explained by 20 components
total_variance = ____.sum(variance_ratios)
print(f"\nTotal variance with 20 components: {total_variance:.3f}")
```

**Hints:**
- Attribute: `.explained_variance_ratio_`
- Array indexing: `variance_ratios[i]`
- Use numpy to sum: `np.sum()` or `.sum()`

**AI Help:** Ask ChatGPT: "What is explained_variance_ratio_ in PCA?"

### Task 1.3: Interpret the results

Look at the variance explained by each component.

**Answer these questions:**

1. Which component captures the most variance?
```
Answer: _______________________________________________
```

2. Does PC5 capture more or less variance than PC1?
```
Answer: _______________________________________________
```

3. With 20 components, what percentage of total variance do we keep?
```
Answer: _______________________________________________
```

**Key insight:** Often, a small number of components capture most of the variance! This is why PCA is so powerful for compression.

***

## Part 2: Choosing the Number of Components (10 minutes)

**Background:** How many components should we keep? A common approach is to keep enough components to retain 90% (or 95%) of the variance. Let's find this threshold!

### Task 2.1: Calculate cumulative variance

**Fill in the blanks:**

```python
# Fit PCA with all components
pca_full = PCA()
pca_full.fit(____)

# Get explained variance ratios
all_variances = pca_full.____

# Calculate cumulative variance
cumulative_variance = ____.cumsum(all_variances)

print(f"Total components available: {len(cumulative_variance)}")
print(f"Cumulative variance with all components: {cumulative_variance[-1]:.3f}")
```

**Hints:**
- Fit on: `X`
- Attribute: `.explained_variance_ratio_`
- Numpy cumulative sum: `np.cumsum()`

**AI Help:** Ask ChatGPT: "How do I calculate cumulative sum in numpy?"

### Task 2.2: Find 90% threshold

**Fill in the blanks:**

```python
# Find how many components needed for 90% variance
threshold = 0.90
n_components_90 = np.argmax(____ >= threshold) + 1

print(f"Components needed for {threshold:.0%} variance: {n_components_90}")
print(f"Actual variance with {n_components_90} components: {cumulative_variance[____]:.3f}")

# Also find for 95%
threshold_95 = ____
n_components_95 = np.argmax(cumulative_variance >= ____) + 1
print(f"Components needed for {threshold_95:.0%} variance: {n_components_95}")
```

**Hints:**
- Check cumulative_variance against threshold
- Array indexing: `cumulative_variance[n_components_90 - 1]`
- 95% threshold: `0.95`

**AI Help:** Ask ChatGPT: "How do I find the index where an array exceeds a threshold in numpy?"

### Task 2.3: Visualize the scree plot

**Fill in the blanks:**

```python
# Create scree plot
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(____) + 1), cumulative_variance, 'b-', linewidth=2)
plt.axhline(y=____, color='green', linestyle='--', label='90% threshold')
plt.axhline(y=0.95, color='orange', linestyle='--', label='95% threshold')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Variance Explained')
plt.title('Cumulative Variance vs Number of Components')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

**Hints:**
- Length of: `cumulative_variance`
- 90% threshold: `0.90`

### Task 2.4: Interpret the plot

Look at your scree plot.

**Answer these questions:**

1. How many components do we need to keep 90% of variance?
```
Answer: _______________________________________________
```

2. How much dimension reduction is this? (From 64 to how many?)
```
Answer: _______________________________________________
```

3. Is there an "elbow" point where adding more components gives little extra variance?
```
Answer: _______________________________________________
```

**Key insight:** We can often reduce dimensions dramatically (64 → ~20) while keeping most information (90%+)!

***

## Part 3: Visualizing High-Dimensional Data in 2D (10 minutes)

**Background:** One of PCA's most powerful uses is visualization. We can't plot 64 dimensions, but we can plot 2! Let's project our digits into 2D and see if different digits cluster together.

### Task 3.1: Project to 2D

**Fill in the blanks:**

```python
# Create PCA for 2D projection
pca_2d = PCA(n_components=____)
X_2d = pca_2d.____(____)

print(f"2D shape: {X_2d.shape}")
print(f"Variance captured by PC1: {pca_2d.explained_variance_ratio_[0]:.3f}")
print(f"Variance captured by PC2: {pca_2d.explained_variance_ratio_[____]:.3f}")
print(f"Total variance in 2D: {pca_2d.explained_variance_ratio_.sum():.3f}")
```

**Hints:**
- 2 components
- Method: `.fit_transform(X)`
- PC2 index: `1`

**AI Help:** Ask ChatGPT: "How do I reduce data to 2D using PCA?"

### Task 3.2: Create a scatter plot

**Fill in the blanks:**

```python
# Create scatter plot colored by digit
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_2d[:, ____], X_2d[:, 1], c=____, cmap='tab10', alpha=0.6, s=30)
plt.colorbar(scatter, label='Digit')
plt.xlabel('First Principal Component (PC1)')
plt.ylabel('Second Principal Component (PC2)')
plt.title('Digits Dataset: 2D PCA Projection')
plt.grid(True, alpha=0.3)
plt.show()
```

**Hints:**
- PC1 is column: `0`
- Color by: `y` (the labels)

**AI Help:** Ask ChatGPT: "How do I create a scatter plot with colors based on labels in matplotlib?"

### Task 3.3: Interpret the visualization

Look at your 2D scatter plot.

**Answer these questions:**

1. Can you see any clusters? Do different digits group together?
```
Answer: _______________________________________________
```

2. What percentage of total variance do these 2 components capture?
```
Answer: _______________________________________________
```

3. Even with only ~25-30% variance, can we see structure? What does this tell you?
```
Answer: _______________________________________________
___________________________________________________________
```

**Key insight:** Even with just 2 components (capturing ~25-30% variance), we can see meaningful structure! Different digits naturally cluster in different regions of the 2D space.

***

## Part 4: Image Reconstruction - Seeing Information Loss (12 minutes)

**Background:** When we reduce dimensions, we lose some information. But we can **reconstruct** the original data from the reduced version using PCA's `.inverse_transform()`. Let's see how reconstruction quality changes with different numbers of components!

### Task 4.1: Reconstruct with few components

**Fill in the blanks:**

```python
# Try reconstruction with only 5 components
pca_5 = PCA(n_components=____)
X_reduced_5 = pca_5.fit_transform(____)
X_reconstructed_5 = pca_5.____(X_reduced_5)

print(f"Original shape: {X.shape}")
print(f"Reduced shape (5 components): {X_reduced_5.shape}")
print(f"Reconstructed shape: {X_reconstructed_5.shape}")
print(f"Variance kept: {pca_5.explained_variance_ratio_.sum():.3f}")
```

**Hints:**
- 5 components
- Fit on: `X`
- Method to go back: `.inverse_transform()`

**AI Help:** Ask ChatGPT: "What does inverse_transform do in PCA?"

### Task 4.2: Reconstruct with more components

**Fill in the blanks:**

```python
# Try different numbers of components
n_components_list = [1, 5, 10, 20, 40]
reconstructions = []

for n in n_components_list:
    pca_temp = ____(n_components=n)
    X_reduced_temp = pca_temp.____(X)
    X_recon_temp = pca_temp.inverse_transform(____)
    reconstructions.append(X_recon_temp)
    
    variance = pca_temp.explained_variance_ratio_.sum()
    print(f"n={n:2d}: Variance kept = {variance:.3f}")
```

**Hints:**
- Create: `PCA(n_components=n)`
- Fit and transform: `.fit_transform(X)`
- Reduced data: `X_reduced_temp`

### Task 4.3: Visualize reconstructions

**Fill in the blanks:**

```python
# Pick one digit to visualize
digit_idx = 0

# Create subplots
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.flatten()

# Original
axes[0].imshow(X[digit_idx].reshape(8, 8), cmap='gray')
axes[0].set_title('Original')
axes[0].axis('off')

# Reconstructions with different components
for i, n in enumerate(n_components_list):
    axes[i+1].imshow(reconstructions[____].reshape(____, ____), cmap='gray')
    axes[i+1].set_title(f'{n} components')
    axes[i+1].axis('off')

plt.suptitle(f'Digit {y[digit_idx]} Reconstructed with Different Numbers of Components')
plt.tight_layout()
plt.show()
```

**Hints:**
- Index reconstructions list: `i`
- Reshape to: `8, 8`

### Task 4.4: Compare reconstruction quality

Let's look at the reconstruction error:

```python
# Calculate mean squared error for each reconstruction
for i, n in enumerate(n_components_list):
    mse = np.mean((X[digit_idx] - reconstructions[i][____]) ** 2)
    print(f"n={n:2d} components: MSE = {mse:.4f}")
```

**Hint:**
- Get same digit from reconstruction: `digit_idx`

**Answer these questions:**

1. With 1 component, can you recognize the digit?
```
Answer: _______________________________________________
```

2. How many components do you need before the digit looks "good"?
```
Answer: _______________________________________________
```

3. What happens to reconstruction error (MSE) as we add more components?
```
Answer: _______________________________________________
```

4. Is perfect reconstruction (MSE=0) possible? With how many components?
```
Answer: _______________________________________________
```

**Key insight:** There's a trade-off between compression (fewer components) and quality (lower error). Often, moderate compression (10-20 components) gives acceptable quality!

***
## Part 5: Real Image Compression (8 minutes)

**Background:** PCA isn't just for datasets - it's used for real image compression! Let's apply PCA to compress a grayscale photograph. Instead of compressing across samples (like digits), we'll compress the image itself by treating each row as a "sample".

### Task 5.1: Load and display a real image

**Copy and run this code:**

```python
# Load a real sample image bundled with scikit-learn
from sklearn.datasets import load_sample_image

image = load_sample_image("china.jpg")
gray_image = image[..., :3].mean(axis=2) / 255.0

plt.figure(figsize=(8, 8))
plt.imshow(gray_image, cmap='gray')
plt.title('Original Grayscale Image')
plt.axis('off')
plt.show()

print(f"Image shape: {gray_image.shape}")
print(f"Image size: {gray_image.shape[0]} rows × {gray_image.shape[1]} columns")
```

**What's happening here?**
- We load a sample photograph bundled with scikit-learn
- Convert it to grayscale using the mean of the RGB channels
- Each row becomes a "sample", each column becomes a "feature"

**Hint:** You can also try `load_sample_image("flower.jpg")`, or replace the sample with your own RGB image array.

**AI Help:** Ask ChatGPT: "How do I load my own image as a NumPy array for PCA?"

### Task 5.2: Apply PCA to compress the image

For image compression, we treat each row as a "sample" and each column as a "feature".

**Fill in the blanks:**

```python
# Apply PCA to compress image
n_components = ____  # Try 50 components

pca_img = PCA(n_components=n_components)
image_reduced = pca_img.____(gray_image)
image_reconstructed = pca_img.inverse_transform(____)

print(f"Original size: {gray_image.shape}")
print(f"Compressed size: {image_reduced.shape}")
print(f"Variance kept: {pca_img.explained_variance_ratio_.sum():.3f}")
```

**Hints:**
- Try 50 components first
- Method to fit and transform: `.fit_transform(gray_image)`
- Reconstruct from: `image_reduced`

**AI Help:** Ask ChatGPT: "How do I use PCA for image compression in Python?"

### Task 5.3: Compare original and compressed images

**Fill in the blanks:**

```python
# Visualize comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].imshow(____, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(____, cmap='gray')
axes[1].set_title(f'Compressed ({n_components} components, {pca_img.explained_variance_ratio_.sum():.1%} variance)')
axes[1].axis('off')

plt.tight_layout()
plt.show()
```

**Hints:**
- Original: `gray_image`
- Reconstructed: `image_reconstructed`

### Task 5.4: Calculate compression ratio

**Fill in the blanks:**

```python
# Calculate how much compression we achieved
original_size = gray_image.shape[0] * gray_image.shape[____]
compressed_size = image_reduced.shape[0] * image_reduced.shape[____] + \
                  pca_img.components_.shape[0] * pca_img.components_.shape[1]

compression_ratio = original_size / ____

print(f"Original elements: {original_size}")
print(f"Compressed elements: {compressed_size}")
print(f"Compression ratio: {compression_ratio:.2f}×")
print(f"Space saved: {(1 - 1/compression_ratio)*100:.1f}%")
```

**Hints:**
- Number of columns: `1`
- Compressed total: `compressed_size`

**Note:** We count both the reduced data AND the components matrix, since we need both to reconstruct!

### Task 5.5: Experiment with different compression levels

**Fill in the blanks:**

```python
# Try different numbers of components
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

component_counts = [10, 25, 50, 100, 150, 200]

for i, n_comp in enumerate(component_counts):
    pca_temp = PCA(n_components=____)
    img_reduced = pca_temp.fit_transform(____)
    img_recon = pca_temp.____(img_reduced)
    
    axes[i].imshow(img_recon, cmap='gray')
    variance = pca_temp.explained_variance_ratio_.sum()
    axes[i].set_title(f'{n_comp} components\n({variance:.1%} variance)')
    axes[i].axis('off')

plt.suptitle('Image Compression with Different Numbers of Components', fontsize=14)
plt.tight_layout()
plt.show()
```

**Hints:**
- Components: `n_comp`
- Fit on: `gray_image`
- Reconstruct: `.inverse_transform()`

**AI Help:** Ask ChatGPT: "How do I create a subplot grid in matplotlib?"

### Task 5.6: Analyze compression quality

**Answer these questions by looking at your results:**

1. With 10 components, can you still recognize the image? What's lost?
```
Answer: _______________________________________________
```

2. How many components do you need before the image looks "good enough"?
```
Answer: _______________________________________________
```

3. What's the compression ratio with 50 components?
```
Answer: _______________________________________________
```

4. Look at the 200 component image. Is it much better than 100 components? What does this tell you about diminishing returns?
```
Answer: _______________________________________________
___________________________________________________________
```

**Key insight:** PCA enables real image compression! With just 50-100 components (out of 640 columns in this example), we can substantially reduce the representation while keeping good visual quality. This is the basis for many compression algorithms!

***

## Summary

**What you learned today:**
- ✅ How to apply PCA with `PCA()` from sklearn
- ✅ How to interpret variance explained and choose number of components
- ✅ How to visualize high-dimensional data in 2D
- ✅ How PCA enables compression and reconstruction
- ✅ The trade-off between compression (fewer components) and information preservation

**Key takeaway:** PCA finds new axes that capture maximum variance, enabling dimensionality reduction while preserving information. It's incredibly useful for visualization, compression, and preprocessing!

***

## Part 6: Advanced Challenge - For MPS439 Students Only (30 minutes)

**Background:** Now you'll understand the mathematics behind PCA! You'll calculate the covariance matrix, compute eigenvalues/eigenvectors, and implement PCA from scratch using Singular Value Decomposition (SVD).

### Task 6.1: Calculate the Covariance Matrix

The covariance matrix is the foundation of PCA. For centered data **X**, it's: **C** = (1/n) **X**ᵀ**X**

**Fill in the blanks:**

```python
# Use a small subset for computation
X_small = X[:100, :]  # First 100 samples

# Step 1: Center the data
X_mean = ____.mean(axis=0)
X_centered = X_small - ____

print(f"Mean of centered data: {X_centered.mean():.10f}")  # Should be ~0

# Step 2: Calculate covariance matrix manually
n = X_centered.shape[____]
C = (1/n) * X_centered.____ @ X_centered

print(f"Covariance matrix shape: {C.shape}")
print(f"Covariance matrix is symmetric: {np.allclose(C, C.T)}")
```

**Hints:**
- Mean: `X_small.mean(axis=0)`
- Subtract mean: `X_mean`
- Number of samples: `0`
- Transpose: `.T`

**AI Help:** Ask ChatGPT: "How do I calculate a covariance matrix in numpy?"

### Task 6.2: Compute Eigenvalues and Eigenvectors

PCA components are the eigenvectors of the covariance matrix!

**Fill in the blanks:**

```python
# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(____)

# Sort by eigenvalue (descending)
idx = eigenvalues.argsort()[::-1]
eigenvalues_sorted = eigenvalues[____]
eigenvectors_sorted = eigenvectors[:, ____]

print(f"Top 5 eigenvalues: {eigenvalues_sorted[:5]}")
print(f"Eigenvector shape: {eigenvectors_sorted[:, 0].shape}")

# Verify they're unit vectors
print(f"First eigenvector norm: {np.linalg.norm(eigenvectors_sorted[:, ____]):.6f}")
```

**Hints:**
- Compute eigenvalues of: `C`
- Sort both by idx: `idx`
- First eigenvector: `0`

**AI Help:** Ask ChatGPT: "How do I compute eigenvalues and eigenvectors in numpy?"

### Task 6.3: Verify against sklearn

Let's check if our eigenvalues match sklearn's!

**Fill in the blanks:**

```python
# Fit sklearn PCA
pca_sklearn = PCA(n_components=10)
pca_sklearn.fit(____)

# Compare eigenvalues (variance)
# Our eigenvalues need to be scaled by (n-1)/n for sklearn's formula
our_variance = eigenvalues_sorted[:10] * (n-1) / ____
sklearn_variance = pca_sklearn.____

print("Comparison of eigenvalues (variance):")
for i in range(10):
    print(f"PC{i+1}: Our={our_variance[i]:.6f}, sklearn={sklearn_variance[____]:.6f}")
```

**Hints:**
- Fit on: `X_centered`
- Scale by n: `n`
- Attribute: `.explained_variance_`
- Index: `i`

### Task 6.4: Implement PCA using SVD

SVD is more numerically stable than eigendecomposition. Let's implement PCA from scratch!

**Background:** For centered data **X** = **U** **Σ** **V**ᵀ, the principal components are the columns of **V**, and the eigenvalues are σ²/(n-1).

**Fill in the blanks:**

```python
def pca_from_scratch(X, n_components):
    """
    Implement PCA using SVD.
    
    Parameters:
    X : array (n_samples, n_features) - data to transform
    n_components : int - number of components to keep
    
    Returns:
    X_transformed : array (n_samples, n_components)
    components : array (n_components, n_features)
    explained_variance : array (n_components,)
    """
    # Step 1: Center the data
    X_mean = X.mean(axis=____)
    X_centered = X - ____
    
    # Step 2: Compute SVD
    U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
    
    # Step 3: Extract principal components (rows of Vt)
    components = Vt[:____]
    
    # Step 4: Transform data (project onto components)
    X_transformed = X_centered @ components.____
    
    # Step 5: Calculate explained variance
    n = X.shape[0]
    explained_variance = (s[:n_components] ** 2) / (n - ____)
    
    return X_transformed, components, explained_variance

# Test your implementation
X_transformed_ours, components_ours, variance_ours = pca_from_scratch(X_small, n_components=10)

print(f"Transformed shape: {X_transformed_ours.shape}")
print(f"Components shape: {components_ours.shape}")
print(f"Variance shape: {variance_ours.shape}")
```

**Hints:**
- Mean axis: `0`
- Subtract: `X_mean`
- First n_components rows: `n_components`
- Transpose of components: `.T`
- Divide by: `1`

**AI Help:** Ask ChatGPT: "How does SVD relate to PCA? Explain the mathematical connection."

### Task 6.5: Verify your implementation

**Fill in the blanks:**

```python
# Compare with sklearn
pca_sklearn = PCA(n_components=10)
X_transformed_sklearn = pca_sklearn.fit_transform(____)

# Check if results match (allowing for sign flips)
print("\nVerification against sklearn:")
print("Variance comparison:")
for i in range(10):
    print(f"PC{i+1}: Ours={variance_ours[i]:.6f}, sklearn={pca_sklearn.explained_variance_[____]:.6f}")

# Check transformation (absolute values, since sign can flip)
transformation_match = np.allclose(np.abs(____), np.abs(X_transformed_sklearn))
print(f"\nTransformations match: {transformation_match}")

# Check components (absolute values)
components_match = np.allclose(np.abs(____), np.abs(pca_sklearn.components_))
print(f"Components match: {components_match}")
```

**Hints:**
- Fit on: `X_small`
- Index: `i`
- Our transformation: `X_transformed_ours`
- Our components: `components_ours`

### Task 6.6: Understanding the Math

**Fill in the blanks to answer these questions:**

**Question 1:** Why do we use SVD instead of eigendecomposition in practice?

```python
# Hint: Think about numerical stability and computation speed
# Try computing eigenvalues of X.T @ X for large matrices

Your answer: _______________________________________________
___________________________________________________________
```

**Question 2:** The relationship between singular values (s) and eigenvalues (λ):

```python
# Complete this formula
# lambda_i = s_i^2 / ____

Your answer: _______________________________________________
```

**Question 3:** Why must data be centered before PCA?

```python
# Try running PCA on non-centered data and explain what happens

# Test: Non-centered data
X_not_centered = X_small  # Not centered!
pca_test = PCA(n_components=2)
pca_test.fit(X_not_centered)

# Manually center
pca_test2 = PCA(n_components=2)
pca_test2.fit(X_not_centered - X_not_centered.mean(axis=0))

# Compare first principal components
print("PC1 without centering:", pca_test.components_[0][:5])
print("PC1 with centering:", pca_test2.components_[0][:5])

Your explanation: _______________________________________________
___________________________________________________________
```

### Task 6.7: Advanced Exploration (Optional)

**Challenge 1:** Implement whitening transformation

Whitening transforms data so all components have unit variance:

```python
def pca_whitening(X, n_components):
    """
    Apply PCA whitening transformation.
    """
    # Your implementation here
    # Hint: After PCA transformation, divide by sqrt(eigenvalues)
    pass

# Test your implementation
```

**Challenge 2:** Implement Kernel PCA

Research and implement a simple version of Kernel PCA for non-linear dimensionality reduction:

```python
# Hint: Use RBF kernel and perform PCA on the kernel matrix
from sklearn.metrics.pairwise import rbf_kernel

# Your implementation here
```

**Challenge 3:** Compare PCA with other dimensionality reduction methods

```python
# Try t-SNE and UMAP, compare with PCA
from sklearn.manifold import TSNE
# from umap import UMAP  # May need: pip install umap-learn

# Your comparison code here
```

---

**Congratulations!** You've completed the advanced section. You now understand PCA from both a practical and mathematical perspective - you can use it AND build it from scratch!

---

**End of Lab Worksheet**

<!-- COURSE_FEEDBACK_QR:START -->
---

## 30-second feedback

Scan the code to share anonymous feedback or post a question for this lab. It opens the correct **Lesson 07 lab** record automatically.

![Feedback QR code for Lesson 07 lab](./feedback-qr.png){fig-align="center" width="180px"}
<!-- COURSE_FEEDBACK_QR:END -->
