# Lesson 8 Lecture Slides - Detailed Outline

## Slide 1: Title Slide
**Layout**: Center-aligned title slide

**Content**:
- Main title: "Lesson 8: K-means and Hierarchical Clustering"
- Subtitle: "Discovering Groups in Unlabeled Data"
- Course: MPS311/439 Machine Learning
- Instructor: Dr. Wei Xing
- School of Mathematics and Statistics, University of Sheffield

---

## Slide 2: Recap & Today's Challenge
**Layout**: Two-column layout

**Left Column (40%)**:
- Heading: "Last Week: PCA"
- Bullet points:
  - Found hidden structure in data
  - Maximized variance in fewer dimensions
  - Reduced dimensionality while preserving information
- Key insight: Discovered structure through variance

**Right Column (60%)**:
- Heading: "This Week: Clustering"
- New challenge highlighted in colored box:
  - "How do we **group similar data** when we have **no labels**?"
- This is **unsupervised learning**
- Two methods today:
  - **K-means**: Fast, needs K
  - **Hierarchical**: Exploratory, builds tree

---

## Slide 3: The Clustering Challenge
**Layout**: Full-width image with text overlay

**Content**:
- Image: `./figures/clustering_challenge.png`
- Heading above image: "Can You Identify Natural Groups?"
- Question below image in large text: "If I asked you to organize these points into 3 groups, how would you do it?"
- Bottom callout box: "Your brain does this naturally - but how do we teach a computer?"

---

## Slide 4: Real-World Applications
**Layout**: Three-column grid

**Column 1: Customer Segmentation**
- Icon/emoji: 🛒
- Scenario: E-commerce with millions of users
- Data: browsing, purchases, time on site
- Goal: Discover customer types (VIP, window shoppers, budget buyers)
- Use: Tailor marketing strategies

**Column 2: Image Compression**
- Icon/emoji: 🖼️
- Scenario: Color images with millions of pixels
- Data: RGB values for each pixel
- Goal: Group similar colors together
- Use: Reduce storage, maintain quality

**Column 3: Gene Expression**
- Icon/emoji: 🧬
- Scenario: Thousands of genes measured
- Data: Expression levels across conditions
- Goal: Find genes that work together
- Use: Biological insights (immune response, cell division)

**Bottom banner**: "Common Pattern: Find natural groupings without predefined categories"

---

## Slide 5: K-means - The Intuitive Idea
**Layout**: Two-step visual explanation

**Top Section (Problem Setup)**:
- Text box: "Thought Experiment"
- "If someone told you where the **centers** of each group are..."
- "→ Clustering is easy! Just assign each point to nearest center"
- "But we don't know the centers... 🤔"

**Bottom Section (The Solution)**:
- Text box highlighted in color: "The Brilliant Insight"
- Numbered steps in large font:
  1. Start with a **guess** of where centers are
  2. Assign points to nearest center
  3. **Improve** guess by computing actual center of each group
  4. Repeat until stable
- Bottom tagline: "This is **K-means clustering** - iterative optimization!"

---

## Slide 6: K-means Algorithm
**Layout**: Algorithm steps with math

**Heading**: "The K-means Algorithm"

**Four Steps** (each in colored box):

1. **Initialize**: Randomly select K data points as centroids $\mu_1, \mu_2, ..., \mu_K$

2. **Assignment**: Assign each point to nearest centroid
   
   <div>
   
   $c_i = \arg\min_{k} \|x_i - \mu_k\|^2$
   
   </div>

3. **Update**: Recompute centroids as mean of assigned points
   
   <div>
   
   $\mu_k = \frac{1}{|C_k|} \sum_{i \in C_k} x_i$
   
   </div>

4. **Iterate**: Repeat steps 2-3 until convergence (centroids stop moving)

**Bottom callout**: "Guaranteed to converge! Each iteration reduces within-cluster variance."

---

## Slide 7: K-means in Action
**Layout**: Full-width figure

**Content**:
- Heading: "Watching K-means Converge"
- Image: `./figures/kmeans_iterations.png` (4-panel visualization)
- Caption below: "Notice how clusters form and stabilize - typically converges in 10-50 iterations"
- Bottom highlight box: "The algorithm **always makes progress** - never gets worse!"

---

## Slide 8: Why "Spherical" Clusters?
**Layout**: Explanation with visual concept

**Left side (50%)**:
- Heading: "Geometric Insight"
- K-means assigns based on **Euclidean distance** to centroids
- Creates **perpendicular bisectors** between centroids
- Results in convex, roughly circular regions (Voronoi diagram)

**Right side (50%)**:
- Heading: "Works Best When:"
- Checkmark list:
  - ✅ Clusters are compact and spherical
  - ✅ Clusters are well-separated
  - ✅ Clusters have similar sizes
  - ✅ Clusters have similar variances

**Bottom note**: "Understanding this helps us know when K-means will struggle"

---

## Slide 9: When K-means Fails
**Layout**: Full-width figure with annotations

**Content**:
- Heading: "Understanding Limitations"
- Image: `./figures/kmeans_failures.png` (2-panel showing failure cases)
- Left panel annotation: "**Non-convex shapes**: Distance-based assignment can't capture curved structures"
- Right panel annotation: "**Different sizes/densities**: K-means assumes similar variance"
- Bottom takeaway box: "If your data has these characteristics, consider other methods (DBSCAN, Gaussian Mixture Models)"

---

## Slide 10: K-means in Python
**Layout**: Code example with explanation

**Heading**: "K-means is Easy in Sklearn"

**Code block**:
```python
from sklearn.cluster import KMeans

# Create and fit K-means with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# Get results
labels = kmeans.labels_          # Cluster assignments
centers = kmeans.cluster_centers_ # Centroid positions
```

**Key Parameters** (in colored boxes):
- `n_clusters`: Number of clusters K (you must specify!)
- `random_state`: For reproducible results
- `n_init=10`: Runs K-means 10 times, returns best (handles initialization sensitivity)

**Bottom note**: "We'll see this live in the demo!"

---

## Slide 11: The Crucial Question - Choosing K
**Layout**: Problem statement with solution preview

**Top section (The Problem)**:
- Large text: "But how do we choose K?"
- Unlike supervised learning, classes aren't given by data
- Sometimes domain knowledge tells us (e.g., marketing wants 4 segments)
- Often we need a **data-driven approach**

**Bottom section (The Solution Preview)**:
- Heading: "The Elbow Method"
- Brief description: Plot a metric vs. K, look for the "elbow"
- Key metric: **Inertia** (within-cluster sum of squares)

<div>

$\text{Inertia} = \sum_{k=1}^{K} \sum_{i \in C_k} \|x_i - \mu_k\|^2$

</div>

- Lower inertia = tighter clusters

---

## Slide 12: The Elbow Method
**Layout**: Full-width figure with explanation

**Content**:
- Heading: "Finding the Optimal K"
- Image: `./figures/elbow_plot.png`
- Top explanation box:
  - "As K increases, inertia always decreases"
  - "But after the 'right' K, we get **diminishing returns**"
- Bottom interpretation guide:
  - **Before elbow**: Steep decrease (capturing real structure)
  - **At elbow**: Rate of decrease slows sharply → optimal K
  - **After elbow**: Gradual decrease (overfitting)
- Practical note: "Not always a clear elbow - use domain knowledge too!"

---

## Slide 13: Transition to Hierarchical
**Layout**: Problem → Solution format

**Top section (K-means Limitation)**:
- Highlighted box: "K-means requires choosing K **in advance**"
- Questions in large font:
  - What if we want to explore 2 groups? 5 groups? 10 groups?
  - What if we want to **visualize** how clusters merge?
  - What if we don't want to commit to a specific K?

**Bottom section (Hierarchical Solution)**:
- Heading: "Hierarchical Clustering to the Rescue!"
- Builds a **tree (dendrogram)** showing all possible groupings
- Explore clustering at multiple scales
- No need to choose K upfront
- Two approaches: Agglomerative (bottom-up) ← We'll focus here

---

## Slide 14: Hierarchical Clustering Idea
**Layout**: Algorithm explanation with linkage criteria

**Top section (Algorithm)**:
- Heading: "Agglomerative (Bottom-up) Approach"
- Three steps in sequence:
  1. **Start**: Each point is its own cluster (n clusters)
  2. **Loop**: Find and merge the two closest clusters
  3. **End**: Until everything is one cluster
- Result: Complete merge history (dendrogram)

**Bottom section (Linkage Criteria)**:
- Heading: "How do we measure distance between clusters?"
- Three options in columns:
  - **Single**: Minimum distance between any two points (elongated clusters)
  - **Complete**: Maximum distance between any two points (compact clusters)
  - **Average**: Average of all pairwise distances (balanced, robust)

---

## Slide 15: The Dendrogram
**Layout**: Full-width figure with reading guide

**Content**:
- Heading: "Understanding Dendrograms"
- Image: `./figures/dendrogram.png`
- Reading guide (left side):
  - **Bottom**: Individual data points
  - **Y-axis**: Distance at which merges occur
  - **Vertical lines**: Clusters being merged
  - **Height**: Shows dissimilarity
- Usage guide (right side):
  - "Cut horizontally to extract K clusters"
  - Red line → K=3 clusters
  - Blue line → K=5 clusters
- Bottom insight: "Large vertical gaps = natural separations (good cut points!)"

---

## Slide 16: Hierarchical in Python
**Layout**: Code example with two approaches

**Heading**: "Two Ways to Use Hierarchical Clustering"

**Approach 1: Get clusters directly**:
```python
from sklearn.cluster import AgglomerativeClustering

hc = AgglomerativeClustering(n_clusters=3, linkage='average')
labels = hc.fit_predict(X)
```

**Approach 2: Visualize dendrogram**:
```python
from scipy.cluster.hierarchy import dendrogram, linkage

Z = linkage(X, method='average')
dendrogram(Z)
```

**Key parameters** (in box):
- `n_clusters`: Where to cut the tree
- `linkage`: 'single', 'complete', 'average', or 'ward'

---

## Slide 17: K-means vs Hierarchical
**Layout**: Comparison table

**Heading**: "When to Use Which?"

**Table** (with colored rows for clarity):

| Aspect | K-means | Hierarchical |
|--------|---------|--------------|
| **Speed** | Fast: O(nKt) | Slower: O(n²log n) |
| **Dataset Size** | Large (n > 10,000) | Small-medium (n < 5,000) |
| **Choose K?** | Must know K | Explore multiple K |
| **Visualization** | Cluster assignments | Dendrogram tree |
| **Cluster Shape** | Spherical, similar size | Flexible shapes |
| **Best For** | Known K, speed matters | Exploration, interpretation |

**Bottom decision guide**:
- Use **K-means**: When you know K, have large data, want speed
- Use **Hierarchical**: When exploring, want tree, have small-medium data

---

## Slide 18: Practical Tips
**Layout**: Three-column tips

**Column 1: Feature Scaling**
- Icon: ⚖️
- **Critical for K-means!**
- Uses Euclidean distance
- Features with larger ranges dominate
- Solution: Standardize before clustering
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

**Column 2: Multiple Runs**
- Icon: 🔄
- K-means sensitive to initialization
- Can get stuck in local minima
- Solution: sklearn's `n_init=10`
- Runs 10 times, returns best

**Column 3: Method Selection**
- Icon: 🎯
- Choose based on:
  - Dataset size
  - Whether K is known
  - Desired cluster shape
  - Need for visualization
- Try both if unsure!

---

## Slide 19: Summary - What We Learned
**Layout**: Key concepts with icons

**Heading**: "Today's Journey"

**Four main boxes**:

1. **Clustering Problem** 🎯
   - Unsupervised learning: finding groups without labels
   - Similarity measured by distance

2. **K-means** ⚡
   - Iterative: Assignment → Update → Repeat
   - Fast but needs K, assumes spherical clusters
   - Elbow method helps choose K

3. **Hierarchical** 🌳
   - Builds merge tree (dendrogram)
   - Explore multiple K values
   - Slower but more flexible

4. **Practical Wisdom** 💡
   - Scale your features!
   - Understand limitations
   - Choose method based on context

**Bottom banner**: "You now have two powerful tools for discovering hidden patterns in data!"

---

## Slide 20: Learning Outcomes
**Layout**: Checklist format

**Heading**: "What You Should Now Be Able To Do"

**Core Skills (MPS311 & MPS439)**:
- ✅ Apply K-means clustering using sklearn
- ✅ Apply hierarchical clustering using sklearn
- ✅ Explain why K-means finds spherical clusters
- ✅ Describe when K-means vs hierarchical is better
- ✅ Choose optimal K using the elbow method
- ✅ Interpret dendrograms and extract clusters
- ✅ Understand why feature scaling matters

**Advanced Skills (MPS439)**:
- ✅ Implement K-means algorithm from scratch
- ✅ Understand the mathematical objective function
- ✅ Explain convergence guarantees

**Next Steps**:
- Lab session: Hands-on practice
- Assignment 2: Apply clustering to real data
- Office hours: Tuesday 12-1pm

---

# Appendix: Interactive Code Demonstrations

## Demo 1: K-means Convergence Visualization
**Purpose**: Show students how K-means iteratively converges to final clustering

**Description**:
- Generate 2D synthetic data with 3 clear clusters (use make_blobs)
- Implement step-by-step K-means visualization
- **Interactive elements**:
  - Slider for "iteration number" (0 to max_iterations)
  - As slider moves, show:
    - Current centroid positions (marked with red X)
    - Current cluster assignments (colored points)
    - Lines connecting points to their assigned centroids (optional, can be toggled)
  - Display iteration number and current inertia value
  - Button to "auto-play" iterations (animate convergence)
  - Button to "reset" with new random initialization

**Educational goal**: Students see that K-means always makes progress, understand the assignment + update cycle, and observe that different initializations can lead to different results

**Key features**:
- Clean 2D scatter plot
- Clear color coding for clusters
- Large, visible centroid markers
- Display inertia value decreasing

---

## Demo 2: The Elbow Method Interactive
**Purpose**: Help students understand how to choose optimal K using the elbow method

**Description**:
- Generate 2D synthetic data (default: 4 natural clusters)
- **Interactive elements**:
  - Slider for "number of clusters K" (range 1-10)
  - As K changes, show TWO side-by-side plots:
    - **Left plot**: Clustering result for current K value (scatter plot with colored clusters)
    - **Right plot**: Elbow curve (inertia vs K), with current K highlighted
  - Slider for "number of true clusters" in data generation (2-6)
    - Regenerates data with different number of natural clusters
    - Students can see how elbow point matches true K
  - Button to "show elbow annotation" (adds vertical line and label at recommended K)

**Educational goal**: Students learn to read elbow plots, understand that the elbow point suggests optimal K, and see how this relates to the actual cluster structure

**Key features**:
- Real-time update as K changes
- Clear elbow annotation
- Ability to test with different datasets
- Display inertia value numerically

---

## Demo 3: K-means Failure Cases
**Purpose**: Show students when K-means struggles and why

**Description**:
- Dropdown menu to select dataset type:
  - "Well-separated spherical" (K-means works great)
  - "Non-convex (half-moons)" (K-means fails)
  - "Different densities" (K-means struggles)
  - "Elongated clusters" (K-means fails)
  - "Concentric circles" (K-means fails)
- For each dataset:
  - Show true structure (if applicable, colored differently)
  - Show K-means clustering result
  - Show cluster boundaries (Voronoi diagram)
- **Interactive elements**:
  - Slider for K (2-5)
  - Button to "re-run with new initialization"
  - Checkbox to "show Voronoi boundaries"

**Educational goal**: Students understand that K-means assumes spherical clusters, see geometric reason for failures (Voronoi partitioning), and learn to recognize when K-means is inappropriate

**Key features**:
- Multiple dataset examples
- Clear visualization of failure modes
- Optional Voronoi boundaries to show decision regions

---

## Demo 4: Dendrogram Interactive Explorer
**Purpose**: Help students understand dendrograms and how cutting at different heights gives different numbers of clusters

**Description**:
- Generate 2D hierarchical data (e.g., 50 points with clear hierarchical structure)
- Display TWO synchronized plots:
  - **Left plot**: Dendrogram (tree diagram)
  - **Right plot**: Scatter plot of actual data points
- **Interactive elements**:
  - Horizontal slider for "cutting height"
  - As slider moves, show:
    - Red horizontal line on dendrogram at current cut height
    - Number of resulting clusters displayed
    - Data points colored by cluster membership on right plot
  - Dropdown for linkage method ('single', 'complete', 'average')
    - Regenerates dendrogram when changed
    - Students see how linkage affects tree structure
  - Display current K (number of clusters) at cut height

**Educational goal**: Students learn to read dendrograms, understand that cutting height determines K, and see how different linkage methods affect clustering

**Key features**:
- Synchronized dendrogram and scatter plot
- Real-time cluster highlighting
- Clear cut line visualization
- Linkage comparison capability

---

## Demo 5: Feature Scaling Impact
**Purpose**: Demonstrate why feature scaling is critical for K-means

**Description**:
- Generate 2D data where features have very different scales
  - Feature 1: range 0-1 (e.g., normalized spending)
  - Feature 2: range 0-1000 (e.g., visit frequency in raw counts)
- **Interactive elements**:
  - Checkbox: "Apply StandardScaler"
  - When unchecked: Show K-means on raw data (Feature 2 dominates)
  - When checked: Show K-means on scaled data (both features contribute)
  - Display TWO scatter plots side-by-side for comparison
  - Slider for K (2-5)
  - Display cluster centers with coordinates

**Educational goal**: Students see dramatic difference between scaled and unscaled clustering, understand why features with larger ranges dominate distance calculations, and always remember to scale before clustering

**Key features**:
- Clear before/after comparison
- Display actual coordinate values of centroids
- Show feature ranges (min, max) for both versions
- Visual annotation explaining which feature dominates when unscaled

---

## Demo 6: K-means vs Hierarchical Comparison
**Purpose**: Show students the practical differences between K-means and hierarchical clustering

**Description**:
- Generate 2D dataset (moderate size, ~200 points)
- Display THREE panels:
  - **Left**: K-means result
  - **Middle**: Hierarchical (agglomerative) result
  - **Right**: Dendrogram from hierarchical
- **Interactive elements**:
  - Slider for K (2-7)
  - Both methods use same K for fair comparison
  - Dropdown for linkage method (affects hierarchical only)
  - Button to "randomize data" (generate new dataset)
  - Display computation time for each method
  - Checkbox to "show cluster centers" (K-means only)

**Educational goal**: Students compare results from both methods, understand that results can differ, see computation time differences, and learn when each method is more appropriate

**Key features**:
- Side-by-side comparison
- Timing information displayed
- Ability to test on different datasets
- Clear visual distinction between methods

---

## Technical Requirements for All Demos

**General specifications**:
- Use matplotlib for all visualizations
- Implement with ipywidgets for interactivity (works in Colab)
- Use clear, large fonts (14pt minimum for labels)
- Color-blind friendly color schemes (use colorblind-safe palettes)
- Add informative titles and axis labels
- Include brief text explanations on plots when helpful
- Code should be modular and well-commented for teaching purposes

**Performance requirements**:
- All interactions should update in < 1 second
- Use reasonable dataset sizes (50-500 points depending on demo)
- Pre-compute when possible (e.g., elbow values for all K)

**Educational requirements**:
- Each demo should be self-contained (can run independently)
- Include markdown cells explaining what students should observe
- Add "Try this" suggestions in markdown (e.g., "Try moving the K slider - what happens to the clusters?")
- Keep UI simple and uncluttered - focus on the concept being taught