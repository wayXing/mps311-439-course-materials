---
theme: default
background: https://cover.sli.dev
class: text-center
highlighter: shiki
lineNumbers: false
info: |
  ## Lesson 8: K-means and Hierarchical Clustering
  MPS311/439 Machine Learning - Dr. Wei Xing
drawings:
  persist: false
transition: slide-left
title: 'Lesson 8: K-means and Hierarchical Clustering'
routerMode: hash
mdc: true
---

# Lesson 8: K-means and Hierarchical Clustering

## Discovering Groups in Unlabeled Data

<div class="pt-12">
  <span class="text-xl">
    MPS311/439 Machine Learning<br>
    Dr. Wei Xing<br>
    2026–27
  </span>
</div>

---

# Recap & Today's Challenge

<div class="grid grid-cols-2 gap-4">

<div>

## Last Week: PCA

- Found hidden structure in data
- Maximized variance in fewer dimensions
- Reduced dimensionality while preserving information

**Key insight**: Discovered structure through variance

</div>

<div>

## This Week: Clustering

<div class="bg-blue-100 p-4 rounded-lg mt-4">

**New Challenge**: How do we **group similar data** when we have **no labels**?

</div>

This is **unsupervised learning**!

**Two methods today**:
- **K-means**: Fast, needs K
- **Hierarchical**: Exploratory, builds tree

</div>

</div>

---

# The Clustering Challenge

<div class="text-center">

<!-- ### Can You Identify Natural Groups? -->

<img src="./figures/clustering_challenge.png" class="mx-auto" style="width: 55%; margin-top: 1rem; margin-bottom: 1rem;">

<div class="text-l font-bold text-blue-600 mt--4">

If I asked you to organize these points into 3 groups, how would you do it?

</div>

<div class="bg-yellow-100 p-0 rounded-lg mt-0 inline-block">

Your brain does this naturally - but how do we teach a computer?

</div>

</div>

---

# Real-World Applications

<div class="grid grid-cols-3 gap-6 mt-8">

<div class="text-left bg-blue-50">

### 🛒 Customer Segmentation

**Scenario**: E-commerce with millions of users

**Data**: browsing, purchases, time on site

**Goal**: Discover customer types (VIP, window shoppers, budget buyers)

**Use**: Tailor marketing strategies

</div>

<div class="text-left bg-yellow-50">

### 🖼️ Image Compression

**Scenario**: Color images with millions of pixels

**Data**: RGB values for each pixel

**Goal**: Group similar colors together

**Use**: Reduce storage, maintain quality

</div>

<div class="text-left bg-red-50">

### 🧬 Gene Expression

**Scenario**: Thousands of genes measured

**Data**: Expression levels across conditions

**Goal**: Find genes that work together

**Use**: Biological insights (immune response, cell division)

</div>

</div>

<div class="bg-green-100 p-1 rounded-lg mt-4 text-center text-lg font-semibold">

Common Pattern: Find natural groupings without predefined categories

</div>

---

# K-means: A Concrete Example

<div class="mt-0">

## Imagine: Grouping Students in This Class

<div class="bg-blue-50 p-1 rounded-lg mb-3">

**Scenario**: We measure everyone's height and weight, plot them as points

- No labels (not told who is male/female)
- Just x-y coordinates: (height, weight)
- Task: **Can we find 2 natural groups?**

</div>

<div class="grid grid-cols-2 gap-6">

<div class="bg-purple-50 p-1 rounded-lg">

### 🤔 If we knew the "center" of each group...

- Male center: (180cm, 75kg)
- Female center: (165cm, 60kg)

**Easy!** Just assign each person to whichever center they're closer to

</div>

<div class="bg-red-50 p-1 rounded-lg">

### 😕 But we don't know the centers!

**The Problem**:
- We have the points (measurements)
- We don't have the centers
- We don't have labels

**How do we start?**

</div>

</div>

</div>

---

# K-means: The Intuitive Idea

<div class="mt-0">

<div class="bg-green-50 p-1 rounded-lg">

<!-- ### 💡 The Brilliant Insight -->

<div class="text-l mt-1">

**Just guess and improve iteratively!**

1. **Guess**: Pick 2 random students as initial "centers"
   - Say: Alice (170cm, 65kg) and Bob (175cm, 70kg)

2. **Assign**: Each student goes to nearest center
   - Creates 2 groups

3. **Improve**: Compute actual center of each group
   - Average height and weight of Group 1; Average height and weight of Group 2

1. **Repeat**: Use new centers, reassign everyone, recompute...

</div>

</div>

<div class="text-center text-2xl font-bold text-green-700 mt-2 bg-yellow-100 p-4 rounded-lg">

This is **K-means clustering** - iterative optimization!

<div class="text-lg font-normal mt-2">

Guaranteed to converge to stable groups

</div>

</div>

</div>

<!-- ---

# K-means: The Intuitive Idea

<div class="mt-4">

<div class="bg-blue-50 p-1 rounded-lg mb-8">

## 🤔 Thought Experiment

If someone told you where the **centers** of each group are...

→ Clustering is easy! Just assign each point to nearest center

But we don't know the centers... 

</div>

<div class="bg-green-50 p-1 rounded-lg">

## 💡 The Brilliant Insight

<div class="text-xl mt-1">

1. Start with a **guess** of where centers are
2. Assign points to nearest center
3. **Improve** guess by computing actual center of each group
4. **Repeat** until stable

</div>

<div class="text-center text-2xl font-bold text-green-700 mt-6">

This is **K-means clustering** - iterative optimization!

</div>

</div>

</div> -->

---

# K-means Algorithm

<div class="mt-4">

<div class="grid grid-cols-2 gap-4">

<div class="bg-blue-50 p-4 rounded-lg">

### 1. Initialize

Randomly select K data points as centroids

<div class="mt-2">

$\mu_1, \mu_2, ..., \mu_K$

</div>

</div>

<div class="bg-green-50 p-4 rounded-lg">

### 2. Assignment

Assign each point to nearest centroid

<div class="mt-2">

$c_i = \arg\min_{k} \|x_i - \mu_k\|^2$

</div>

</div>

<div class="bg-yellow-50 p-4 rounded-lg">

### 3. Update

Recompute centroids as mean of assigned points

<div class="mt-2">

$\mu_k = \frac{1}{|C_k|} \sum_{i \in C_k} x_i$

</div>

</div>

<div class="bg-purple-50 p-4 rounded-lg">

### 4. Iterate

Repeat steps 2-3 until convergence

<div class="mt-2">

Centroids stop moving

</div>

</div>

</div>

<div class="bg-red-50 p-3 rounded-lg mt-6 text-center text-lg font-semibold">

✅ Guaranteed to converge! Each iteration reduces within-cluster variance.

</div>

</div>

---

<div class="flex flex-col md:flex-row items-center gap-x-8 gap-y-6">
    <!-- Left Column: Text (1/3 width on medium screens and up) -->
    <div class="md:w-1/3 text-center md:text-left">        
        <h2 class="text-2xl font-bold mb-4">K-means in Action</h2>
        <p class="text-lg mb-4">
            Notice how clusters form and stabilize - typically converges in 10-50 iterations.
        </p>
        <div class="bg-blue-100 p-4 rounded-lg inline-block text-lg font-semibold">
            The algorithm <strong>always makes progress</strong> — it never gets worse!
        </div>
    </div>
    <!-- Right Column: Image (2/3 width on medium screens and up) -->
    <div class="md:w-2/3">
        <img src="./figures/kmeans_iterations.png" alt="Animation showing K-means clustering converging over iterations" class="w-full rounded-lg shadow-lg">
    </div>
</div>

---

# Why "Spherical" Clusters?

<div class="grid grid-cols-2 gap-8 mt-8">

<div>

## Geometric Insight

- K-means assigns based on **Euclidean distance** to centroids
- Creates **perpendicular bisectors** between centroids
- Results in convex, roughly circular regions (**Voronoi diagram**)

<div class="bg-blue-50 p-4 rounded-lg mt-6">

**Mathematical formulation**:

Distance from point to centroid:

<div class="mt-2">

$\|x_i - \mu_k\|^2$

</div>

Decision boundaries are linear!

</div>

</div>

<div>

## Works Best When:

<div class="text-lg mt-4">

✅ Clusters are **compact and spherical**

✅ Clusters are **well-separated**

✅ Clusters have **similar sizes**

✅ Clusters have **similar variances**

</div>

<div class="bg-yellow-100 p-4 rounded-lg mt-8">

**Key takeaway**: Understanding this helps us know when K-means will struggle

</div>

</div>

</div>

---

# When K-means Fails (Understanding Limitations)

<div class="text-center">

<!-- ## Understanding Limitations -->

<img src="./figures/kmeans_failures.png" class="mx-auto" style="width: 90%; margin-top: 2rem; margin-bottom: 0rem;">

<div class="grid grid-cols-2 gap-8 mt-2">

<div class="bg-red-50 p-1 rounded-lg">

**Non-convex shapes**: Distance-based assignment can't capture curved structures

</div>

<div class="bg-orange-50 p-1 rounded-lg">

**Different sizes/densities**: K-means assumes similar variance across clusters

</div>

</div>

<div class="bg-purple-100 p-1 rounded-lg mt-2 inline-block">

If your data has these characteristics, consider other methods (DBSCAN, Gaussian Mixture Models)

</div>

</div>

---

# K-means in Python

<div class="mt-4">

## K-means is Easy in Sklearn
```python {all|1|4|5|8-9|all}
from sklearn.cluster import KMeans

# Create and fit K-means with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# Get results
labels = kmeans.labels_           # Cluster assignments
centers = kmeans.cluster_centers_ # Centroid positions
```

<div class="grid grid-cols-3 gap-4 mt-8">

<div class="bg-blue-50 p-3 rounded-lg">

**`n_clusters`**

Number of clusters K (you must specify!)

</div>

<div class="bg-green-50 p-3 rounded-lg">

**`random_state`**

For reproducible results

</div>

<div class="bg-yellow-50 p-3 rounded-lg">

**`n_init=10`**

Runs K-means 10 times, returns best

</div>

</div>

<div class="text-center text-lg font-semibold text-purple-600 mt-6">

We'll see this live in the demo! 🚀

</div>

</div>

---

# The Crucial Question: Choosing K

<div class="mt-6">

<div class="bg-red-50 p-6 rounded-lg mb-6">

## ❓ But how do we choose K?

- Unlike supervised learning, classes aren't given by data
- Sometimes domain knowledge tells us (e.g., marketing wants 4 segments)
- Often we need a **data-driven approach**

</div>

<div class="bg-green-50 p-6 rounded-lg">

## 💡 The Elbow Method

**Idea**: Plot a metric vs. K, look for the "elbow"

**Key metric**: **Inertia** (within-cluster sum of squares)

<div class="mt-4">

$\text{Inertia} = \sum_{k=1}^{K} \sum_{i \in C_k} \|x_i - \mu_k\|^2$

</div>

<div class="mt-4 text-lg">

Lower inertia = tighter clusters

</div>

</div>

</div>

---

# The Elbow Method

<div class="text-center">

## Finding the Optimal K

<img src="./figures/elbow_plot.png" class="mx-auto" style="width: 75%; margin-top: 1rem; margin-bottom: 1rem;">

<div class="grid grid-cols-3 gap-4 mt-4">

<div class="bg-green-50 p-3 rounded-lg">

**Before elbow**: Steep decrease (capturing real structure)

</div>

<div class="bg-yellow-50 p-3 rounded-lg">

**At elbow**: Rate slows sharply → optimal K

</div>

<div class="bg-red-50 p-3 rounded-lg">

**After elbow**: Gradual decrease (overfitting)

</div>

</div>

<div class="bg-blue-100 p-3 rounded-lg mt-4 inline-block">

⚠️ Not always a clear elbow - use domain knowledge too!

</div>

</div>

---

# Transition to Hierarchical

<div class="mt-4">

<div class="bg-red-50 p-2 rounded-lg mb-8">

## 🚧 K-means Limitation

K-means requires choosing K **in advance**

<div class="text-xl mt-2">

- What if we want to explore 2 groups? 5 groups? 10 groups?
- What if we want to **visualize** how clusters merge?
- What if we don't want to commit to a specific K?

</div>

</div>

<div class="bg-green-50 p-2 rounded-lg">

## 🌳 Hierarchical Clustering to the Rescue!

- Builds a **tree (dendrogram)** showing all possible groupings
- Explore clustering at multiple scales
- No need to choose K upfront
- Two approaches: **Agglomerative** (bottom-up) ← We'll focus here

</div>

</div>

---

# Hierarchical Clustering Idea

<div class="mt-1">

<div class="bg-blue-50 p-1 rounded-lg mb-6">

## Agglomerative (Bottom-up) Approach

<div class="text-xl mt-1">

1. **Start**: Each point is its own cluster (n clusters)
2. **Loop**: Find and merge the two closest clusters
3. **End**: Until everything is one cluster

</div>

<div class="text-lg font-semibold text-blue-700 mt-1">

Result: Complete merge history (dendrogram)

</div>

</div>

<div class="bg-purple-50 p-1 rounded-lg">

## How do we measure distance between clusters?

<div class="grid grid-cols-3 gap-4 mt-1">

<div class="bg-white p-1 rounded-lg border-2 border-purple-200">

**Single**

Minimum distance between any two points

→ Elongated clusters

</div>

<div class="bg-white p-1 rounded-lg border-2 border-purple-200">

**Complete**

Maximum distance between any two points

→ Compact clusters

</div>

<div class="bg-white p-1 rounded-lg border-2 border-purple-200">

**Average**

Average of all pairwise distances

→ Balanced, robust ✅

</div>

</div>

</div>

</div>

---

## The Dendrogram

<div class="text-center">

<!-- ## Understanding Dendrograms -->

<img src="./figures/dendrogram.png" class="mx-auto" style="width: 65%; margin-top: 1rem; margin-bottom: 1rem;">

<div class="grid grid-cols-2 gap-6 mt-1">

<div class="bg-blue-50 p-0 rounded-lg text-left">

### Reading Guide

- **Bottom**: Individual data points
- **Y-axis**: Distance at which merges occur
- **Vertical lines**: Clusters being merged
- **Height**: Shows dissimilarity

</div>

<div class="bg-green-50 p-0 rounded-lg text-left">

### Usage Guide

- Cut horizontally to extract K clusters
- Red line → K=3 clusters
- Blue line → K=5 clusters
- **Large vertical gaps** = natural separations!

</div>

</div>

</div>

---

# Hierarchical in Python

<div class="mt-4">

## Two Ways to Use Hierarchical Clustering

<div class="grid grid-cols-2 gap-6 mt-6">

<div>

### Approach 1: Get clusters directly
```python
from sklearn.cluster import 
  AgglomerativeClustering

hc = AgglomerativeClustering(
  n_clusters=3, 
  linkage='average'
)
labels = hc.fit_predict(X)
```

</div>

<div>

### Approach 2: Visualize dendrogram
```python
from scipy.cluster.hierarchy import 
  dendrogram, linkage

Z = linkage(X, method='average')
dendrogram(Z)
```

</div>

</div>

<div class="bg-yellow-50 p-4 rounded-lg mt-8">

### Key Parameters

- **`n_clusters`**: Where to cut the tree
- **`linkage`**: `'single'`, `'complete'`, `'average'`, or `'ward'`

</div>

</div>

---

# K-means vs Hierarchical

<div class="mt-0">

<!-- ## When to Use Which? -->

<div class="text-sm">

| Aspect | K-means | Hierarchical |
|--------|---------|--------------|
| **Speed** | Fast: O(nKt) | Slower: O(n²log n) |
| **Dataset Size** | Large (n > 10,000) ✅ | Small-medium (n < 5,000) |
| **Choose K?** | Must know K upfront | Explore multiple K ✅ |
| **Visualization** | Cluster assignments | Dendrogram tree ✅ |
| **Cluster Shape** | Spherical, similar size | Flexible shapes ✅ |
| **Best For** | Known K, speed matters | Exploration, interpretation |

</div>

<div class="grid grid-cols-2 gap-6 mt-3">

<div class="bg-blue-50 p-0 rounded-lg">

### Use K-means when:

✅ You know K
✅ Large data
✅ Speed matters

</div>

<div class="bg-green-50 p-0 rounded-lg">

### Use Hierarchical when:

✅ Exploring different K
✅ Want tree visualization
✅ Small-medium data

</div>

</div>

</div>

---

# Practical Tips

<div class="grid grid-cols-3 gap-6 mt-6">

<div class="bg-blue-50 p-5 rounded-lg">

### ⚖️ Feature Scaling

**Critical for K-means!**

- Uses Euclidean distance
- Features with larger ranges dominate

**Solution**: Standardize first
```python
from sklearn.preprocessing 
  import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

</div>

<div class="bg-green-50 p-5 rounded-lg">

### 🔄 Multiple Runs

K-means sensitive to initialization

- Can get stuck in local minima

**Solution**: sklearn's `n_init=10`

- Runs 10 times
- Returns best result
- Handles initialization sensitivity

</div>

<div class="bg-yellow-50 p-5 rounded-lg">

### 🎯 Method Selection

**Choose based on**:

- Dataset size
- Whether K is known
- Desired cluster shape
- Need for visualization

**Try both if unsure!**

</div>

</div>

---

# Summary: What We Learned

<div class="mt-6">

<div class="grid grid-cols-2 gap-6">

<div class="bg-blue-50 p-5 rounded-lg">

### 🎯 Clustering Problem

- Unsupervised learning: finding groups without labels
- Similarity measured by distance
- Real applications: customer segmentation, image compression, gene analysis

</div>

<div class="bg-green-50 p-5 rounded-lg">

### ⚡ K-means

- Iterative: Assignment → Update → Repeat
- Fast but needs K, assumes spherical clusters
- Elbow method helps choose K
- Always converges (guaranteed!)

</div>

<div class="bg-purple-50 p-5 rounded-lg">

### 🌳 Hierarchical

- Builds merge tree (dendrogram)
- Explore multiple K values
- Slower but more flexible
- Linkage criteria affect results

</div>

<div class="bg-yellow-50 p-5 rounded-lg">

### 💡 Practical Wisdom

- Scale your features!
- Understand limitations
- Choose method based on context
- Multiple initializations help K-means

</div>

</div>

<div class="bg-red-50 p-4 rounded-lg mt-8 text-center text-xl font-bold">

You now have two powerful tools for discovering hidden patterns in data!

</div>

</div>

---

# Learning Outcomes

<div class="mt-4">

## What You Should Now Be Able To Do

<div class="grid grid-cols-2 gap-6 mt-6">

<div>

### Core Skills (MPS311 & MPS439)

<div class="text-lg">

✅ Apply K-means clustering using sklearn

✅ Apply hierarchical clustering using sklearn

✅ Explain why K-means finds spherical clusters

✅ Describe when K-means vs hierarchical is better

✅ Choose optimal K using the elbow method

✅ Interpret dendrograms and extract clusters

✅ Understand why feature scaling matters

</div>

</div>

<div>

### Advanced Skills (MPS439)

<div class="text-lg">

✅ Implement K-means algorithm from scratch

✅ Understand the mathematical objective function

✅ Explain convergence guarantees

</div>

### Next Steps

<div class="bg-blue-50 p-4 rounded-lg mt-6">

- **Lab session**: Hands-on practice
- **Assignment 2**: Apply clustering to real data
- **Office hours**: Tuesday 12-1pm

</div>

</div>

</div>

</div>

<!-- COURSE_FEEDBACK_QR:START -->
---
layout: center
class: text-center
---

# 30-second feedback

<p class="text-3xl mb-4">What would help you learn better next time?</p>

<p class="text-2xl mb-4">Scan to share anonymous feedback on today's lecture.</p>

<img src="./feedback-qr.svg" alt="Feedback QR code for Lesson 08 lecture" class="w-44 mx-auto rounded-lg shadow" />

<p class="text-sm mt-4 opacity-70">MPS311/439 · 2026-27 · Lesson 08 lecture</p>
<!-- COURSE_FEEDBACK_QR:END -->
