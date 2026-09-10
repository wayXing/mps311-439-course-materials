# **Lesson 5: Linear Discriminant Analysis - Detailed Outline**

---

## **Title:** Linear Discriminant Analysis: A Generative Approach to Classification

---

### **1. Introduction: Looking Back and Moving Forward** (~5 min)

#### 1.1 Where We Left Off
- Lesson 4 recap: Logistic regression as a discriminative classifier
  - Models P(Y|X) directly: "Given features, what's the probability of class Y?"
  - Decision boundary learned by minimizing cross-entropy loss
  - Works well but makes no assumptions about feature distributions

#### 1.2 A Different Question
- What if we flip the question?
  - Instead of P(Y|X), what if we model P(X|Y)?
  - "What do features look like for each class?"
  - Then use Bayes' theorem to classify
- Motivating example: Recognizing handwriting
  - Discriminative: "Does this look more like a 3 or an 8?"
  - Generative: "If I were writing a 3, what would it look like? What about an 8?"

---

### **2. Motivation: Why Model Classes?** (~10 min)

#### 2.1 The Iris Dataset Story
- Brief introduction: 3 species of iris flowers, 4 measurements
- **[Figure: fig1_iris_species.png (800x400)]** - Scatter plot showing two iris species in 2D (sepal length vs sepal width), naturally separated

#### 2.2 Two Ways to Think About Classification
- **Discriminative approach** (what we know):
  - Draw a line between classes
  - Focus on the boundary
- **Generative approach** (new perspective):
  - Model each class as a "cloud" of points
  - Each cloud has a center and spread
  - Classify based on which cloud a new point is closer to

#### 2.3 When Generative Models Shine
- Small datasets: Better statistical stability
- Unbalanced classes: Can model rare classes explicitly
- Need class probabilities: Direct access to P(X|Y)
- Multiple classes: Naturally extends beyond binary

---

### **3. LDA Fundamentals: The Core Idea** (~20 min)

#### 3.1 The Generative Modeling Framework
- **Bayes' Theorem** refresher:
  ```
  P(Y=k|X) = P(X|Y=k) × P(Y=k) / P(X)
  ```
- Three components:
  1. **P(X|Y=k)**: Class-conditional distribution (likelihood)
  2. **P(Y=k)**: Prior probability (class frequency)
  3. **P(X)**: Evidence (same for all classes, can ignore for classification)

#### 3.2 LDA's Key Assumptions
1. **Assumption 1**: Each class follows a Gaussian (normal) distribution
   - Features X given class Y=k ~ N(μₖ, Σ)
   - μₖ: mean vector for class k
   - Σ: covariance matrix
2. **Assumption 2**: All classes share the same covariance matrix Σ
   - This is the "Linear" in LDA
   - Classes have same shape, different centers
   - **[Figure: fig2_gaussian_classes.png (800x500)]** - Two 2D Gaussian distributions with different means but equal covariance ellipses

#### 3.3 From Assumptions to Decision Boundaries
- Mathematical sketch (intuitive level):
  - Compute P(Y=k|X) for each class k
  - Choose class with highest posterior probability
  - Decision boundary: where P(Y=1|X) = P(Y=2|X)
- Key insight: With equal covariances, the decision boundary is **linear**
  - Quadratic terms cancel out in the log-likelihood ratio
  - Result: Hyperplane separating classes

#### 3.4 Geometric Intuition: Projection View
- **[Figure: fig3_lda_projection.png (800x400)]** - 2D data projected onto 1D line showing class separation
- LDA finds direction that:
  - Maximizes separation between class means
  - Minimizes spread within each class
- Connection to Fisher's Linear Discriminant (mentioned, not derived)

---

### **4. Mathematical Foundation** (~25 min) ⭐ **[Advanced Section for MPS439]**

#### 4.1 Setting Up the Problem
- Notation:
  - K classes, N samples, D features
  - Training data: {(x₁, y₁), ..., (xₙ, yₙ)}
  - πₖ = P(Y=k): prior probability of class k
  - μₖ: mean of class k
  - Σ: shared covariance matrix

#### 4.2 Estimating Parameters from Data
- **Prior probabilities**: πₖ = Nₖ/N (class frequency)
- **Class means**: μₖ = (1/Nₖ) Σᵢ:yᵢ=k xᵢ
- **Pooled covariance**: Σ = (1/(N-K)) Σₖ Σᵢ:yᵢ=k (xᵢ - μₖ)(xᵢ - μₖ)ᵀ

#### 4.3 Deriving the Decision Boundary
- Start with discriminant function:
  ```
  δₖ(x) = log P(X=x|Y=k) + log P(Y=k)
  ```
- For Gaussian: 
  ```
  δₖ(x) = -½(x - μₖ)ᵀ Σ⁻¹ (x - μₖ) + log πₖ + constant
  ```
- Expand and simplify:
  ```
  δₖ(x) = xᵀ Σ⁻¹ μₖ - ½ μₖᵀ Σ⁻¹ μₖ + log πₖ
  ```
- This is **linear in x**! Form: δₖ(x) = wₖᵀx + bₖ
- Decision boundary between classes k and l: δₖ(x) = δₗ(x)

#### 4.4 Why Equal Covariance Matters
- With equal covariance: Quadratic terms xᵀΣ⁻¹x cancel
- Without equal covariance: Decision boundary becomes quadratic (leads to QDA)
- Mathematical consequence: Linear vs curved boundaries

#### 4.5 Connection to Fisher's Linear Discriminant
- Fisher's criterion (for 2 classes):
  ```
  Maximize: (μ₁ - μ₂)² / (s₁² + s₂²)
  ```
  - Numerator: between-class variance
  - Denominator: within-class variance
- LDA and Fisher's discriminant give same projection direction
- Brief note: Fisher's is more general, works for dimensionality reduction

---

### **5. Implementation with sklearn** (~15 min)

#### 5.1 Loading and Exploring Data
- Use Iris dataset (2 features for visualization)
- Code snippet:
  ```python
  from sklearn.datasets import load_iris
  from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
  import numpy as np
  import matplotlib.pyplot as plt
  
  # Load data
  iris = load_iris()
  X = iris.data[:, :2]  # First 2 features
  y = iris.target
  ```

#### 5.2 Training LDA
- Minimal code:
  ```python
  # Create and train LDA
  lda = LinearDiscriminantAnalysis()
  lda.fit(X, y)
  
  # Make predictions
  y_pred = lda.predict(X)
  
  # Get probabilities
  probs = lda.predict_proba(X)
  print(probs[0])  # Example output
  ```

#### 5.3 Understanding the Results
- **[Figure: fig4_lda_decision_boundary.png (800x600)]** - Iris data with LDA decision boundaries (colored regions) and training points
- Interpreting coefficients:
  ```python
  print("Class means:", lda.means_)
  print("Covariance:", lda.covariance_)
  ```
- Training accuracy:
  ```python
  from sklearn.metrics import accuracy_score
  print("Accuracy:", accuracy_score(y, y_pred))
  ```

#### 5.4 What LDA Learned
- Discuss the learned parameters:
  - Class means: centers of each Gaussian
  - Shared covariance: spread pattern
  - Prior probabilities: class frequencies
- How it differs from logistic regression's learned weights

---

### **6. Quadratic Discriminant Analysis (QDA)** (~10 min)

#### 6.1 Relaxing the Equal Covariance Assumption
- What if classes have different spreads?
- QDA allows each class k to have its own Σₖ
- Trade-off: More flexibility vs more parameters to estimate

#### 6.2 Mathematical Consequence
- Discriminant function becomes quadratic:
  ```
  δₖ(x) = -½ log|Σₖ| - ½(x - μₖ)ᵀ Σₖ⁻¹ (x - μₖ) + log πₖ
  ```
- Quadratic term xᵀΣₖ⁻¹x does NOT cancel
- Decision boundaries are **curves** (parabolas, ellipses, etc.)

#### 6.3 QDA in sklearn
- Code:
  ```python
  from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
  
  qda = QuadraticDiscriminantAnalysis()
  qda.fit(X, y)
  y_pred_qda = qda.predict(X)
  ```
- **[Figure: fig5_lda_vs_qda_boundaries.png (1000x500)]** - Side-by-side comparison: LDA (linear boundaries) vs QDA (curved boundaries) on same dataset

#### 6.4 When to Use LDA vs QDA
- **Use LDA when**:
  - Classes have similar spreads
  - Small sample size (fewer parameters)
  - Want simpler, more stable model
- **Use QDA when**:
  - Classes clearly have different covariances
  - Large enough sample size
  - Willing to risk overfitting for flexibility

---

### **7. LDA vs Logistic Regression** (~10 min)

#### 7.1 Philosophical Differences
- **Generative (LDA)** vs **Discriminative (Logistic Regression)**
  - LDA: Models P(X|Y), then uses Bayes' rule
  - Logistic: Models P(Y|X) directly
- Analogy: Learning to distinguish cats from dogs
  - Generative: Learn what cats look like, learn what dogs look like
  - Discriminative: Learn features that separate cats from dogs

#### 7.2 Mathematical Connection
- Surprising fact: If LDA assumptions hold, logistic regression emerges
- Logistic regression: log(P(Y=1|X)/P(Y=0|X)) = β₀ + β₁x₁ + ...
- LDA with 2 Gaussian classes: Same form!
- Difference: How β coefficients are estimated

#### 7.3 Performance Comparison
- **[Figure: fig6_lda_vs_logreg.png (800x500)]** - Performance comparison (accuracy bar chart and decision boundaries) on a synthetic dataset
- Code for comparison:
  ```python
  from sklearn.linear_model import LogisticRegression
  
  # Train both models
  lda = LinearDiscriminantAnalysis()
  logreg = LogisticRegression()
  
  lda.fit(X_train, y_train)
  logreg.fit(X_train, y_train)
  
  # Compare on test set
  print("LDA accuracy:", lda.score(X_test, y_test))
  print("Logistic accuracy:", logreg.score(X_test, y_test))
  ```

#### 7.4 When to Choose Which
- **Prefer LDA when**:
  - Features are approximately Gaussian
  - Classes are well-separated
  - Small training set
  - Need class-conditional densities
- **Prefer Logistic Regression when**:
  - Gaussian assumption clearly violated
  - More robust to outliers
  - Large training set
  - Only care about decision boundary

---

### **8. Practical Considerations** (~5 min)

#### 8.1 Assumptions and Reality
- Gaussian assumption: Rarely perfect in practice
- Equal covariance: Often violated
- **But**: LDA often works well even when assumptions don't perfectly hold
- Robustness: Similar to linear regression

#### 8.2 Common Pitfalls
- **Small sample size**: Covariance estimation becomes unstable
  - Rule of thumb: Need more samples than features
- **High dimensions**: Covariance matrix becomes singular
  - Solutions: Regularization, dimensionality reduction
- **Outliers**: Can strongly affect mean and covariance estimates

#### 8.3 Practical Tips
- Always check assumptions with exploratory plots
- Compare LDA with logistic regression and QDA
- Use cross-validation to choose between models
- Consider regularized variants (available in sklearn)

---

### **9. Summary & Key Takeaways** (~5 min)

#### 9.1 What We Learned
- **Generative approach**: Model P(X|Y) instead of P(Y|X)
- **LDA**: Assumes Gaussian classes with equal covariance
  - Leads to linear decision boundaries
  - Estimates class means and shared covariance
- **QDA**: Relaxes equal covariance assumption
  - Leads to quadratic decision boundaries
  - More flexible but requires more data
- **Comparison**: LDA vs Logistic Regression
  - Different philosophies, similar boundaries
  - LDA better for small data, Logistic more robust

#### 9.2 Core Learning Outcomes ✓
- ✅ Use sklearn's LinearDiscriminantAnalysis and QuadraticDiscriminantAnalysis
- ✅ Explain why LDA assumes equal covariances
- ✅ Describe when LDA vs QDA is appropriate
- ✅ Compare LDA vs logistic regression performance

#### 9.3 Advanced Learning Outcomes ✓ *[MPS439]*
- ✅ Derive LDA decision boundaries mathematically
- ✅ Implement LDA in Python (from scratch - optional in appendix)

#### 9.4 Looking Ahead
- Next week: Decision Trees
  - A completely different approach: No parametric assumptions!
  - Handle non-linear relationships naturally
  - Interpretable and intuitive

---

### **10. Further Reading & Resources**
- **sklearn documentation**: 
  - [LinearDiscriminantAnalysis](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html)
  - [QuadraticDiscriminantAnalysis](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis.html)
- **Classical reference**: Fisher, R. A. (1936). "The Use of Multiple Measurements in Taxonomic Problems"
- **Modern treatment**: Elements of Statistical Learning (Hastie, Tibshirani, Friedman), Chapter 4

---

## **APPENDIX: Figure Specifications for Generation**

All figures should use a clean, educational style with:
- Clear labels and legends
- Readable font sizes (12pt minimum)
- Colorblind-friendly color schemes where applicable
- Professional appearance suitable for lecture notes

---

### **Figure 1: fig1_iris_species.png (800x400)**

**Purpose**: Motivate LDA with real data showing naturally separated classes

**Detailed Description**:
- Scatter plot using Iris dataset
- X-axis: Sepal length (cm)
- Y-axis: Sepal width (cm)
- Plot only 2 classes: Setosa (blue circles) and Versicolor (orange triangles)
- Add alpha=0.6 for slight transparency
- Include grid for readability
- Title: "Two Iris Species in Feature Space"
- Legend showing class names
- Clear axis labels

**Python generation guidance**:
```python
from sklearn.datasets import load_iris
iris = load_iris()
# Use only first 2 features, only classes 0 and 1
# Create scatter plot with different markers and colors
```

---

### **Figure 2: fig2_gaussian_classes.png (800x500)**

**Purpose**: Illustrate LDA's core assumption of Gaussian classes with equal covariance

**Detailed Description**:
- 2D plot showing two bivariate Gaussian distributions
- Class 1 (blue): Mean at (2, 2), covariance [[1, 0.5], [0.5, 1]]
- Class 2 (red): Mean at (5, 5), same covariance
- Draw contour lines for both distributions (3-4 contour levels)
- Draw covariance ellipses (at 1 and 2 standard deviations) showing equal shape
- Annotate means with markers (μ₁ and μ₂)
- Add a dashed line showing the LDA decision boundary
- Title: "LDA Assumption: Equal Covariance, Different Means"
- X and Y axes from 0 to 7

**Python generation guidance**:
```python
from scipy.stats import multivariate_normal
import numpy as np
# Create meshgrid, compute PDF for both Gaussians
# Use plt.contour() for distributions
# Use confidence ellipses or scatter samples
```

---

### **Figure 3: fig3_lda_projection.png (800x400)**

**Purpose**: Show geometric intuition of LDA as projection onto a discriminant direction

**Detailed Description**:
- Top panel: 2D scatter plot with two classes (150 points each)
  - Class 1 (blue): Normally distributed around (1, 3)
  - Class 2 (orange): Normally distributed around (4, 1)
  - Draw an arrow showing projection direction (LDA direction)
- Bottom panel: 1D histogram showing projections
  - Blue histogram for Class 1 projections
  - Orange histogram for Class 2 projections
  - Show clear separation between classes after projection
  - Vertical dashed line showing decision threshold
- Title: "LDA Projects Data to Maximize Class Separation"
- Connect the two panels visually (arrows or alignment)

**Python generation guidance**:
```python
# Generate 2D data with sklearn.datasets.make_classification
# Compute LDA projection direction
# Project points onto this direction
# Create subplot: top for 2D scatter, bottom for 1D histograms
```

---

### **Figure 4: fig4_lda_decision_boundary.png (800x600)**

**Purpose**: Show LDA decision boundaries on Iris dataset with sklearn implementation

**Detailed Description**:
- Background: Filled contour plot showing class prediction regions
  - Use 3 distinct colors for 3 Iris classes (light blue, light orange, light green)
  - Create smooth color regions using np.meshgrid
- Overlay: Scatter plot of actual training data
  - Class 0 (Setosa): Blue circles
  - Class 1 (Versicolor): Orange squares  
  - Class 2 (Virginica): Green triangles
  - Edge color for points to make them stand out
- Draw decision boundaries as black lines where regions meet
- Title: "LDA Decision Boundaries on Iris Dataset"
- X-axis: Sepal length (cm)
- Y-axis: Sepal width (cm)
- Legend showing all three classes

**Python generation guidance**:
```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# Train LDA on iris data (first 2 features)
# Create meshgrid and predict on grid points
# Use plt.contourf() for regions
# Overlay scatter plot of training data
```

---

### **Figure 5: fig5_lda_vs_qda_boundaries.png (1000x500)**

**Purpose**: Compare linear (LDA) vs quadratic (QDA) decision boundaries side-by-side

**Detailed Description**:
- Two subplots side by side
- Same synthetic dataset used in both (binary classification)
  - Generate data where classes have different covariances
  - Class 1: Wide spread in X direction, narrow in Y
  - Class 2: More circular spread
  - About 200 points total
- Left subplot: LDA
  - Colored regions (light blue and light orange)
  - Linear decision boundary (straight line)
  - Scatter points overlaid
  - Title: "LDA: Linear Boundary"
- Right subplot: QDA
  - Colored regions (light blue and light orange)
  - Curved decision boundary
  - Same scatter points
  - Title: "QDA: Quadratic Boundary"
- Main title: "LDA vs QDA Decision Boundaries"
- Show that QDA captures the curved separation better

**Python generation guidance**:
```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
# Generate synthetic data with different covariances per class
# Train both LDA and QDA
# Create 2 subplots with filled contours and scatter
```

---

### **Figure 6: fig6_lda_vs_logreg.png (800x500)**

**Purpose**: Compare performance of LDA vs Logistic Regression

**Detailed Description**:
- Two-panel figure:
  
**Left panel (accuracy comparison)**:
- Grouped bar chart
- X-axis: Two groups ("Training Set", "Test Set")
- Y-axis: Accuracy (0 to 1)
- Two bars per group: LDA (blue) and Logistic Regression (orange)
- Add value labels on top of each bar
- Show that performance is similar

**Right panel (decision boundaries)**:
- Scatter plot of binary classification data (synthetic)
- Show both LDA boundary (blue line) and Logistic boundary (orange line)
- They should be very similar but not identical
- Training points as scatter

- Main title: "LDA vs Logistic Regression Comparison"
- Legend for both panels

**Python generation guidance**:
```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
# Generate or use real dataset
# Train both models, compute accuracies
# Left: bar chart with plt.bar()
# Right: scatter + decision boundaries from both models
```

---

**End of Detailed Outline**

---

**Notes for Generation**:
- Total figures: 6 (within the 8 figure limit)
- All figures use matplotlib as the primary package
- Additional packages: sklearn, numpy, scipy as needed
- Color scheme should be consistent and colorblind-friendly
- All figures should be high quality (DPI=100 minimum)
- Save figures with transparent background where appropriate