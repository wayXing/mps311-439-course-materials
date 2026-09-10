# Lesson 7: Principal Component Analysis (PCA)
## Detailed Lecture Slides Outline

---

## **Slide 1: Title Slide**
**Layout**: Cover page

**Content**:
- Title: "Lesson 7: Principal Component Analysis"
- Subtitle: "Finding Structure in High-Dimensional Data"
- Course: MPS311/439 Machine Learning
- Instructor: Dr. Wei Xing
- University of Sheffield
- Background image or clean design

---

## **Slide 2: Where We Left Off**
**Purpose**: Recap and transition to unsupervised learning

**Content**:
- **Header**: "Our Journey So Far"
- **Left column** - What we've learned:
  - Lesson 2-3: Linear Regression (predict continuous values)
  - Lesson 4-5: Classification (predict categories)
  - Lesson 6: Decision Trees (non-linear decisions)
  - **Common theme**: All supervised - we had labels!
  
- **Right column** - Today's shift:
  - What if we don't have labels?
  - What if we just want to understand data structure?
  - Welcome to **Unsupervised Learning**
  
- **Highlight box**: "PCA is our first unsupervised technique"

---

## **Slide 3: The Challenge - High-Dimensional Data**
**Purpose**: Motivate the problem with concrete example

**Content**:
- **Header**: "The Curse of Dimensionality"

- **Example box**: 
  - "Handwritten digit images: 64×64 pixels = **4,096 dimensions**"
  - "Gene expression data: **20,000+ genes** per patient"
  - "Text documents: **10,000+ word** vocabulary"

- **Three Key Problems** (with icons or visual indicators):
  1. 📊 **Visualization**: Can't plot 4,096 dimensions
  2. ⚡ **Computation**: More features = slower algorithms
  3. 🎯 **Noise**: Many features are redundant or noisy

- **Key insight box**: "Most high-dimensional data lives in a lower-dimensional space!"

- **Figure**: `./figures/fig_pca_intuition.png`
  - Caption: "2D data that really only needs 1D"

---

## **Slide 4: Core Idea - Variance as Information**
**Purpose**: Build intuition for why variance matters

**Content**:
- **Header**: "The Central Principle: Variance = Information"

- **Thought experiment**:
  - "100 students, 3 test scores: Math, Physics, Chemistry"
  - "Goal: Summarize each student with one number"
  - "Which score matters most?"

- **Answer box**:
  - If Math scores vary a lot (50-100) → informative
  - If Chemistry scores barely vary (85-90) → less informative
  - **Directions with high variance contain more information**

- **Visual**: Simple illustration or bullet points showing:
  - High variance feature: spreads data apart
  - Low variance feature: data clustered together

- **Transition**: "PCA finds the directions of maximum variance"

---

## **Slide 5: Finding the Direction of Maximum Variance**
**Purpose**: Visual demonstration of principal components

**Content**:
- **Header**: "Which Direction Matters Most?"

- **Figure**: `./figures/fig_variance_directions.png`
  - Large, centered on slide
  - Caption: "The red arrow shows maximum variance direction (PC1)"

- **Key concepts** (beside or below figure):
  - **First Principal Component (PC1)**: Direction of maximum spread
  - **Second Principal Component (PC2)**: Perpendicular to PC1, next-most variance
  - **Third, Fourth, ...**: Continue with decreasing variance
  
- **Insight box**: "These components form a new coordinate system aligned with data structure"

---

## **Slide 6: How PCA Works - The Algorithm**
**Purpose**: Step-by-step algorithm walkthrough

**Content**:
- **Header**: "PCA in Four Steps"

- **Step-by-step visualization** (numbered, with visual flow):

  **Step 1: Center the Data**
  - Subtract mean from each feature
  - Why? PCA finds directions through origin
  
  **Step 2: Find Maximum Variance Direction**
  - This becomes PC1
  - (Math details coming next...)
  
  **Step 3: Find Orthogonal Directions**
  - PC2: perpendicular to PC1, maximum remaining variance
  - PC3: perpendicular to PC1 & PC2, etc.
  
  **Step 4: Transform and Reduce**
  - Project data onto PC directions
  - Keep only top k components

- **Bottom note**: "All this happens automatically in sklearn!"

---

## **Slide 7: The Mathematics - Covariance Matrix**
**Purpose**: Introduce mathematical foundation (accessible level)

**Content**:
- **Header**: "Mathematical Foundation"

- **The Optimization Problem**:
  - Given: Centered data matrix **X** (n samples × d features)
  - Goal: Find unit vector **w** that maximizes variance
  
  - Variance of projection = $\mathbf{w}^T \mathbf{C} \mathbf{w}$
  
  - Where $\mathbf{C} = \frac{1}{n}\mathbf{X}^T\mathbf{X}$ is the **covariance matrix**

- **The Solution**:
  - Maximize: $\mathbf{w}^T \mathbf{C} \mathbf{w}$
  - Subject to: $\mathbf{w}^T \mathbf{w} = 1$
  - Solution: $\mathbf{C}\mathbf{w} = \lambda\mathbf{w}$ (eigenvalue equation!)

- **Figure**: `./figures/fig_covariance_ellipse.png`
  - Caption: "Covariance ellipse with eigenvectors"

- **Note box**: "Details optional for MPS311, required for MPS439"

---

## **Slide 8: Understanding Eigenvalues & Eigenvectors**
**Purpose**: Intuitive explanation of eigenvalues/eigenvectors

**Content**:
- **Header**: "What Do Eigenvalues and Eigenvectors Mean?"

- **Two columns**:

  **Left: Intuitive Explanation**
  - **Eigenvectors** = Principal component directions
  - **Eigenvalues** = Amount of variance in each direction
  - Larger eigenvalue = more important direction
  
  **Right: Mathematical Result**
  - Eigenvectors of covariance matrix **C** are PCs
  - Eigenvalues = variance captured by each PC
  - Sorted: $\lambda_1 \geq \lambda_2 \geq ... \geq \lambda_d$

- **Key insight box**: 
  - "The eigenvector with largest eigenvalue is PC1"
  - "Second-largest eigenvalue gives PC2, and so on"
  - "This is optimal - no other linear projection captures more variance with k components"

- **Geometric interpretation**: Reference to ellipse figure from previous slide

---

## **Slide 9: Implementation in sklearn**
**Purpose**: Show practical implementation

**Content**:
- **Header**: "Using PCA in Python"

- **Code example** (syntax highlighted):
```python
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits

# Load data (1797 samples, 64 features)
digits = load_digits()
X = digits.data

# Apply PCA - keep top 20 components
pca = PCA(n_components=20)
X_reduced = pca.fit_transform(X)

print(X.shape)          # (1797, 64)
print(X_reduced.shape)  # (1797, 20)
```

- **Key outputs**:
  - `pca.explained_variance_ratio_`: Proportion of variance per component
  - `pca.components_`: The principal component vectors
  - `pca.explained_variance_`: Variance (eigenvalues) per component

- **Bottom note**: "That's it! 64 → 20 dimensions while keeping ~93% variance"

---

## **Slide 10: Choosing Number of Components**
**Purpose**: Strategies for selecting k

**Content**:
- **Header**: "How Many Components Should We Keep?"

- **Three Strategies**:

  **Strategy 1: Variance Threshold**
  - Keep components until 90% (or 95%) total variance
  - Most common approach
  
  **Strategy 2: Scree Plot / Elbow Method**
  - Look for "elbow" where variance drops off
  - More subjective
  
  **Strategy 3: Task-Specific**
  - Visualization: 2 or 3 components
  - Preprocessing: validate on downstream task

- **Figures side-by-side**:
  - Left: `./figures/fig_variance_explained.png`
    - Caption: "Variance per component"
  - Right: `./figures/fig_cumulative_variance.png`
    - Caption: "Cumulative variance (21 components → 90%)"

---

## **Slide 11: Visualization Example**
**Purpose**: Show power of PCA for visualization

**Content**:
- **Header**: "Seeing High-Dimensional Data"

- **Setup box**:
  - Digits dataset: 64 dimensions → 2 dimensions
  - No labels used in PCA
  - Color points by true digit class to see if structure emerges

- **Code snippet**:
```python
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)

plt.scatter(X_2d[:, 0], X_2d[:, 1], c=digits.target)
```

- **Figure**: `./figures/fig_2d_projection.png` (large, prominent)
  - Caption: "Clusters emerge even without using labels!"

- **Observation**: "PC1 and PC2 capture ~28% variance, but reveal meaningful structure"

---

## **Slide 12: When to Use PCA - Practical Considerations**
**Purpose**: Guidelines and common pitfalls

**Content**:
- **Header**: "PCA in Practice: Do's and Don'ts"

- **Two columns**:

  **✅ When PCA Works Well**:
  - Features are correlated (linear relationships)
  - Need to visualize high-D data
  - Preprocessing before ML models
  - Noise reduction
  - Features on similar scales (or standardized)
  
  **❌ Watch Out For**:
  - **Pitfall 1**: Forgetting to standardize
    - If features have different units/scales, PCA biased toward large-scale features
  - **Pitfall 2**: Blindly choosing k
    - More components ≠ always better (can overfit)
  - **Pitfall 3**: Assuming interpretability
    - PCs are linear combinations - often not interpretable
  - **Pitfall 4**: Nonlinear data
    - PCA can't find curved/complex structures

- **Bottom tip box**: "Always standardize if features have different units!"

---

## **Slide 13: What We Learned Today**
**Purpose**: Summary of core concepts

**Content**:
- **Header**: "Summary: Principal Component Analysis"

- **Core Concepts**:
  1. **What PCA Does**:
     - Finds orthogonal directions of maximum variance
     - Reduces dimensionality while preserving information
     - Transforms data to new coordinate system (principal components)
  
  2. **How It Works**:
     - Based on eigendecomposition of covariance matrix
     - Eigenvectors = PC directions, Eigenvalues = importance
     - Guaranteed optimal linear projection
  
  3. **Practical Use**:
     - `sklearn.decomposition.PCA` - easy to use!
     - Choose k by explained variance (90% threshold common)
     - Standardize features if needed

- **Key Formula** (centered):

$$\mathbf{C}\mathbf{w} = \lambda\mathbf{w}$$

- **Bottom note**: "First step into unsupervised learning - finding structure without labels"

---

## **Slide 14: Student Checklist - What You Should Know**
**Purpose**: Learning outcomes verification

**Content**:
- **Header**: "Learning Outcomes - Check Your Understanding"

- **For All Students (MPS311 & MPS439)**:
  - ☐ Can you explain why high variance = high information?
  - ☐ Can you describe what eigenvectors and eigenvalues represent?
  - ☐ Can you implement PCA using sklearn?
  - ☐ Can you choose number of components using explained variance?
  - ☐ Do you know when to standardize features before PCA?
  - ☐ Can you visualize high-D data in 2D/3D using PCA?

- **Additional for MPS439 Students**:
  - ☐ Can you derive the PCA optimization problem?
  - ☐ Can you explain the eigenvalue equation solution?
  - ☐ Do you understand the SVD approach to PCA?

- **Resources**:
  - Lecture notes on Blackboard (with full derivations)
  - Lab session this Friday: hands-on PCA
  - Office hours: Tuesday 12-1pm

- **Bottom quote**: 
  > "PCA: Simple, elegant, powerful. The foundation for understanding data structure."

---

# **APPENDIX: Interactive Code Demonstrations**

## Demo 1: Interactive 2D PCA Visualization with Covariance Control
**Purpose**: Help students understand how data covariance affects principal components

**Description**:
- **Setup**: Generate 2D correlated data from a bivariate Gaussian distribution
- **Interactive Controls**:
  - Slider 1: Variance in X direction (range: 0.5 to 5.0)
  - Slider 2: Variance in Y direction (range: 0.5 to 5.0)
  - Slider 3: Covariance/correlation (range: -0.95 to 0.95)
  - Button: "Regenerate data" (new random sample)
  
- **Visualization**:
  - Scatter plot of 300 data points
  - Overlay: Two arrows showing PC1 (red) and PC2 (blue) directions
  - Arrows scaled by eigenvalues (longer = more variance)
  - Display eigenvalues: λ₁ and λ₂
  - Show explained variance ratio for each component
  
- **Learning Goals**:
  - Students see how PCs align with data spread
  - Understand that PC1 is always the direction of maximum spread
  - Observe that PC1 and PC2 are always perpendicular
  - See how eigenvalues relate to arrow lengths (variance)
  
- **Key Interactions to Demonstrate**:
  - Adjust covariance to 0 → see PCs align with X/Y axes
  - High positive covariance → PC1 along diagonal
  - Equal variances with high correlation → dramatic PC1, tiny PC2

---

## Demo 2: Component Selection - Scree Plot Explorer
**Purpose**: Show how to choose optimal number of components

**Description**:
- **Setup**: Use digits dataset (64 features) or allow upload of custom dataset
- **Interactive Controls**:
  - Slider: Number of components to keep (range: 1 to 64)
  - Radio buttons: Choose visualization type:
    - "Individual Variance" (bar chart)
    - "Cumulative Variance" (line plot)
    - "Both"
  - Threshold lines: Toggle 90% and 95% lines on/off
  
- **Visualization**:
  - **Top panel**: Bar chart or line plot (based on selection)
    - Highlight selected number of components in different color
    - Show threshold lines (90%, 95%) if toggled on
    - Annotate where lines cross thresholds
  
  - **Bottom panel**: Show key statistics
    - "Components selected: k = X"
    - "Variance explained: XX.X%"
    - "Dimensionality reduction: 64 → X (YY% reduction)"
  
- **Learning Goals**:
  - Students understand variance explained vs. cumulative variance
  - See the trade-off: fewer components vs. information loss
  - Practice finding the "elbow" in scree plot
  - Understand that 90-95% variance is often sufficient
  
- **Key Interactions to Demonstrate**:
  - Move slider to show how cumulative variance increases
  - Show that first ~20 components capture 90% variance
  - Demonstrate elbow around k=10-15

---

## Demo 3: Reconstruction Quality Visualization
**Purpose**: Show information loss/preservation with different k values

**Description**:
- **Setup**: Use digits dataset (8×8 images)
- **Interactive Controls**:
  - Slider: Number of PCA components (range: 1 to 64)
  - Dropdown: Select which digit to visualize (0-9)
  - Button: "Random sample" (pick random digit image)
  - Checkbox: "Show difference map"
  
- **Visualization**:
  - **Layout**: Three images side-by-side
    - Left: Original digit image (8×8 grayscale)
    - Middle: Reconstructed from k components
    - Right: Difference map (if checkbox enabled) - shows what was lost
  
  - **Below images**: Display metrics
    - "Components used: X / 64"
    - "Variance retained: XX.X%"
    - "Reconstruction error (MSE): X.XXX"
  
- **Learning Goals**:
  - Students see visual impact of dimension reduction
  - Understand that some information is always lost
  - Observe that loss is often just noise (can improve quality!)
  - See diminishing returns: 1→10 components big improvement, 50→64 minimal
  
- **Key Interactions to Demonstrate**:
  - Start with k=1: very blurry, barely recognizable
  - k=5: Basic shape visible
  - k=10-15: Quite good reconstruction
  - k=30+: Almost identical to original
  - Show difference map at k=10 to see what's lost (usually just noise/edges)

---

## Demo 4: 2D/3D Projection Playground
**Purpose**: Interactive exploration of PCA for visualization

**Description**:
- **Setup**: Digits dataset (or allow multiple dataset options via dropdown)
- **Interactive Controls**:
  - Radio buttons: "2D Projection" or "3D Projection"
  - Checkbox: "Color by true labels" (on/off)
  - Dropdown: Select dataset
    - Digits (10 classes)
    - Iris (3 classes) - optional
    - Custom upload - optional
  - Slider (for 3D only): Rotate view angle
  
- **Visualization**:
  - **2D mode**: Scatter plot in PC1-PC2 space
    - Points colored by class (if checkbox enabled)
    - Axis labels show variance explained
    - Colorbar/legend for classes
  
  - **3D mode**: Interactive 3D scatter in PC1-PC2-PC3 space
    - Rotatable view
    - Points colored by class
    - Grid for depth perception
  
  - **Info box**: Display:
    - "Total variance shown: XX.X%"
    - "PC1: XX.X%, PC2: XX.X%, PC3: XX.X% (if 3D)"
  
- **Learning Goals**:
  - Students see that PCA can reveal structure without labels
  - Observe clustering/separation in low-D space
  - Understand that 2-3 components often show meaningful patterns
  - See that not all variance is needed for visualization
  
- **Key Interactions to Demonstrate**:
  - Start without colors: "Do you see any structure?"
  - Add colors: "Look! Different digits cluster together!"
  - Switch to 3D: "A bit more separation with PC3"
  - Try Iris dataset: "Even more clear separation (3 distinct clusters)"

---

## Demo 5: Standardization Impact Demonstration
**Purpose**: Show why standardization matters

**Description**:
- **Setup**: Create synthetic dataset with features on different scales
  - Feature 1: Small variance (range: 0-1)
  - Feature 2: Large variance (range: 0-1000)
  - Both features equally informative (similar signal-to-noise)
  
- **Interactive Controls**:
  - Checkbox: "Apply standardization" (on/off)
  - Slider: Scale factor for Feature 2 (range: 1 to 1000)
  
- **Visualization**:
  - **Layout**: Two panels side-by-side
    - Left: Raw data projection (no standardization)
    - Right: Standardized data projection
  
  - Each panel shows:
    - 2D scatter plot with PC1 and PC2 overlaid as arrows
    - Bar chart of explained variance ratio
  
  - **Warning banner** (when standardization off and scale high):
    - "⚠️ PC1 dominated by large-scale feature!"
  
- **Learning Goals**:
  - Students see dramatic difference with/without standardization
  - Understand that PCA is sensitive to feature scales
  - Learn when standardization is critical (different units/scales)
  - See that standardization gives each feature fair weight
  
- **Key Interactions to Demonstrate**:
  - Start without standardization, high scale factor
    - Show that PC1 ≈ 100% variance (all from Feature 2)
    - PC1 arrow nearly aligned with Feature 2 axis
  - Enable standardization
    - Show balanced variance between PCs
    - PC1 captures true underlying structure
  - Adjust scale factor to show effect magnitude

---

## General Implementation Notes for Code Team:

**Technical Requirements**:
- Use Jupyter widgets (ipywidgets) for sliders, buttons, checkboxes
- Use matplotlib for 2D plots, plotly for 3D (if interactive rotation desired)
- All demos should run in Google Colab without additional installation
- Include clear title and instructions at top of each demo
- Use colorblind-friendly color palettes (tab10, viridis)
- Add text annotations explaining what to observe

**Pedagogical Notes**:
- Each demo should have a "Reset to defaults" button
- Include a brief text explanation above each demo
- Add "💡 Try this!" prompts suggesting what to adjust
- Keep UI clean and uncluttered
- Responsive updates (immediate, no "run" button needed)

**Code Structure**:
- Modular functions (easy to modify parameters)
- Clear comments for teaching assistants
- Use sklearn for PCA (consistent with lecture)
- Set random seeds for reproducibility when demoing

---

**End of Detailed Outline**