# **Lesson 5 Lecture Notes - Detailed Outline**

## **Linear Discriminant Analysis: Finding the Best Projection for Classification**

---

### **1. Introduction: Where We Are in Our Journey**

#### 1.1 Recap: Logistic Regression (Lesson 4)
- Last week: Classification using logistic regression
- Key idea: Model P(Y|X) directly - the discriminative approach
- Decision boundary: A line (or hyperplane) that separates classes
- Works well, but is there another way?

#### 1.2 Today's New Perspective
- **Central question**: What if we first project high-dimensional data onto a lower dimension where classes are well-separated?
- Preview: Linear Discriminant Analysis (LDA) finds the "best" projection line
- Learning roadmap for today

#### 1.3 Learning Outcomes
- Core outcomes (all students)
- Advanced outcomes (MPS439 students)

---

### **2. Motivation: Why Do We Need Another Classification Method?**

#### 2.1 The Challenge: High-Dimensional Data
- Real-world data often has many features
- Hard to visualize and understand separation
- **Example scenario**: Classifying iris flowers using petal and sepal measurements

#### 2.2 The Power of Projection
- **Key insight**: Sometimes separating classes in the original space is hard, but becomes easy after projection
- **Figure 1** (`fig1_projection_motivation.png`, 800×400px): Shows 2D scatter plot with two classes (e.g., red and blue points) that overlap in both x and y directions, but when projected onto a diagonal line, they separate clearly
- Informal demonstration: "Imagine looking at a 3D object from different angles"

#### 2.3 The Goal of LDA
- Find the projection direction that:
  1. Maximizes separation between different classes
  2. Minimizes scatter within the same class
- Trade-off between these two goals
- Visual preview of what "good separation" looks like

---

### **3. Fisher's Linear Discriminant: The Core Idea**

#### 3.1 Setting Up the Problem
- We have training data: {(x₁, y₁), (x₂, y₂), ..., (xₙ, yₙ)}
- Start with binary classification: y ∈ {0, 1} (two classes)
- Each xᵢ is a d-dimensional vector
- Goal: Find a direction vector **w** to project data onto

#### 3.2 What Makes a Projection "Good"?
- **Figure 2** (`fig2_good_bad_projections.png`, 800×400px): Side-by-side comparison showing two different projection directions for the same 2D data - one direction shows overlapping projected points (bad), another shows separated projected points (good)

#### 3.3 Quantifying "Good Separation": Between-Class Variance
- After projection onto direction **w**, each point becomes a scalar: z = **w**ᵀ**x**
- Class means in the projected space:
  - μ̃₀ = mean of projected class 0 points
  - μ̃₁ = mean of projected class 1 points
- **Between-class variance**: Distance between projected means squared: (μ̃₁ - μ̃₀)²
- We want this to be LARGE

#### 3.4 Quantifying "Compactness": Within-Class Variance
- Variance of projected points within each class:
  - s̃₀² = variance of projected class 0 points
  - s̃₁² = variance of projected class 1 points
- **Within-class variance**: s̃₀² + s̃₁²
- We want this to be SMALL
- **Figure 3** (`fig3_variance_components.png`, 800×500px): Illustrative diagram showing between-class variance (distance between two projected Gaussian-like distributions) and within-class variance (spread of each distribution)

#### 3.5 Fisher's Criterion: The Optimal Balance
- Combine both goals into a single objective:
  ```
  J(w) = (between-class variance) / (within-class variance)
       = (μ̃₁ - μ̃₀)² / (s̃₀² + s̃₁²)
  ```
- Maximize J(w) to find the best projection direction **w**
- This is called **Fisher's Linear Discriminant**

#### 3.6 The Solution (Without Full Derivation)
- Mathematics shows the optimal **w** is proportional to:
  ```
  w ∝ S_W^(-1) (μ₁ - μ₀)
  ```
  Where:
  - μ₀, μ₁ = mean vectors of class 0 and 1 in original space
  - S_W = within-class scatter matrix (measures spread within classes)
  
- **Intuition**: The direction is roughly "from one class mean to the other", adjusted for the data's spread
- Don't worry about computing S_W by hand - sklearn does this for us!

#### 3.7 Geometric Intuition
- The projection direction **w** points from one class center toward the other
- But it's adjusted to account for the shape/spread of the data
- Visual walkthrough using **Figure 1** again

---

### **4. From Projection to Classification Decision**

#### 4.1 Making Predictions
- Once we have the projection direction **w**, classification is simple:
  1. Project new point **x** onto **w**: z = **w**ᵀ**x**
  2. Compare z to a threshold value c
  3. If z > c, classify as class 1; otherwise class 0

#### 4.2 Finding the Threshold
- Threshold c is typically chosen as the midpoint between projected class means
- Or weighted by class proportions (prior probabilities)
- sklearn handles this automatically

#### 4.3 Decision Boundary in Original Space
- The decision boundary in the original d-dimensional space is a hyperplane
- In 2D: a straight line
- **Figure 4** (`fig4_lda_decision_boundary.png`, 700×600px): 2D scatter plot with two classes and the LDA decision boundary as a straight line, with arrows showing the projection direction perpendicular to the boundary

#### 4.4 Extension to Multiple Classes
- LDA naturally extends to K > 2 classes
- Find K-1 projection directions (instead of just 1)
- Each new point is assigned to the nearest class mean in the projected space
- Brief mention only - focus stays on binary case

---

### **5. Implementation with Python and sklearn**

#### 5.1 The Easy Way: Using sklearn
- Import statement:
  ```python
  from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
  ```
- Basic usage is identical to other sklearn classifiers

#### 5.2 Practical Example: Iris Dataset
- Load the iris dataset (use only 2 classes for simplicity)
- Quick data exploration
  ```python
  from sklearn.datasets import load_iris
  import numpy as np
  import matplotlib.pyplot as plt
  ```

#### 5.3 Training an LDA Model
- Code example (minimal):
  ```python
  lda = LinearDiscriminantAnalysis()
  lda.fit(X_train, y_train)
  ```
- Making predictions:
  ```python
  y_pred = lda.predict(X_test)
  ```

#### 5.4 Visualizing Results
- Plot decision boundary (for 2D data)
- **Figure 5** (`fig5_sklearn_lda_example.png`, 700×600px): Real data example showing training points from two classes, LDA decision boundary, and some test points with predictions
- Code snippet for creating the plot (keep it simple)

#### 5.5 Interpreting the Model
- Accessing the projection direction: `lda.coef_`
- Scaling direction: `lda.scalings_`
- Model accuracy:
  ```python
  accuracy = lda.score(X_test, y_test)
  ```
- Print simple output without fancy formatting

#### 5.6 What LDA Gives Us
- Class predictions
- Decision function values (projected coordinates)
- Can also get probability estimates (transformed decision values)

---

### **6. Quadratic Discriminant Analysis (QDA)**

#### 6.1 The Limitation of LDA
- LDA assumes all classes have the same spread (covariance structure)
- **Reality check**: Often classes have different variances!
- Example: One class is tightly clustered, another is spread out

#### 6.2 QDA: Relaxing the Assumption
- Allow each class to have its own covariance matrix
- Decision boundaries become quadratic curves (not straight lines)
- More flexible, but needs more data to estimate

#### 6.3 Visual Comparison
- **Figure 6** (`fig6_lda_vs_qda_boundaries.png`, 800×400px): Side-by-side comparison showing the same 2D data with two classes having different spreads (one circular, one elliptical), with LDA linear boundary on left and QDA curved boundary on right

#### 6.4 When to Use QDA vs LDA
- **Use LDA when**:
  - Classes have similar spread
  - Limited training data
  - Want simpler, more interpretable model
  
- **Use QDA when**:
  - Classes clearly have different spreads
  - Plenty of training data (need ~p(p+1)/2 more parameters per class)
  - Can tolerate more complex model

#### 6.5 QDA Implementation
- Almost identical to LDA:
  ```python
  from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
  qda = QuadraticDiscriminantAnalysis()
  qda.fit(X_train, y_train)
  ```
- Same prediction interface

#### 6.6 Quick Comparison Example
- Fit both LDA and QDA on same dataset
- Compare accuracy
- Simple code snippet

---

### **7. LDA vs Logistic Regression: When to Use Which?**

#### 7.1 Philosophical Differences
- **Logistic Regression**: Directly models decision boundary (discriminative)
- **LDA**: Models each class, then derives boundary (generative perspective - see appendix)
- Both produce linear decision boundaries!

#### 7.2 Performance Comparison
- Run both on the same dataset
- **Figure 7** (`fig7_lda_vs_logreg.png`, 800×400px): Side-by-side scatter plots showing LDA and Logistic Regression decision boundaries on the same data - boundaries should be very similar but not identical
- Compare accuracy metrics

#### 7.3 Practical Guidelines
- **LDA advantages**:
  - Can be more stable with small sample sizes
  - Provides natural dimensionality reduction
  - Works well when classes are approximately Gaussian
  
- **Logistic Regression advantages**:
  - More robust to non-Gaussian data
  - Easier to regularize (Ridge/Lasso)
  - Better when classes have very different sizes

#### 7.4 Rule of Thumb
- Start with Logistic Regression (more robust)
- Try LDA if you have:
  - Small datasets
  - Reason to believe classes are roughly Gaussian
  - Need dimensionality reduction

---

### **8. Practical Considerations and Summary**

#### 8.1 When LDA Works Well
- Classes are roughly Gaussian distributed
- Classes have similar covariance (for LDA) or different covariance (for QDA)
- Sufficient training data (especially for QDA)
- Need interpretable projection directions

#### 8.2 When LDA Struggles
- Highly non-Gaussian data (e.g., multimodal classes)
- Outliers can distort the mean estimates
- Very small sample sizes relative to dimensionality

#### 8.3 Key Takeaways
- LDA finds optimal projection direction to separate classes
- Balance between maximizing between-class distance and minimizing within-class spread
- Linear decision boundaries (LDA) vs quadratic (QDA)
- sklearn makes it easy: `LinearDiscriminantAnalysis` and `QuadraticDiscriminantAnalysis`
- Compare with logistic regression on your data to see which works better

#### 8.4 Looking Ahead
- Next week: Decision Trees - a completely different approach to classification
- Decision trees don't assume linear boundaries!

---

### **Optional Advanced Section: The Generative Model Perspective** 
*(Clearly marked with colored box/sidebar)*

#### A.1 Another View of LDA
- LDA can also be derived from a probabilistic approach
- Assume each class follows a Gaussian (normal) distribution
- P(X | Y=k) ~ N(μₖ, Σ)
- Key assumption: All classes share the same covariance matrix Σ

#### A.2 Bayes' Theorem Connection
- Use Bayes' theorem: P(Y|X) ∝ P(X|Y) P(Y)
- Model each P(X|Y=k) as Gaussian
- When you work through the math, you get the same decision boundary as Fisher's approach!
- Both perspectives lead to the same LDA algorithm

#### A.3 Why Equal Covariance Matters
- Equal covariance → linear decision boundary
- Different covariance → quadratic boundary (QDA)
- This connects the projection view to the probabilistic view

#### A.4 For Curious Students
- This generative approach is powerful for understanding
- Opens door to more advanced methods (e.g., Gaussian Mixture Models)
- Not required for using LDA effectively!
- Recommended reading for MPS439 students

---

### **Summary of Learning Outcomes**

**Core Outcomes - All Students Should Be Able To:**
- [ ] Use `LinearDiscriminantAnalysis` and `QuadraticDiscriminantAnalysis` from sklearn
- [ ] Explain why LDA assumes equal covariances (same spread for all classes)
- [ ] Describe when LDA vs QDA is appropriate (similar vs different spreads)
- [ ] Compare LDA vs logistic regression performance on a dataset
- [ ] Visualize decision boundaries for 2D classification problems

**Advanced Outcomes - MPS439 Students:**
- [ ] Derive LDA decision boundaries mathematically (from generative perspective)
- [ ] Explain Fisher's criterion and the between/within variance trade-off
- [ ] Implement basic LDA from scratch using numpy
- [ ] Understand the connection between projection and generative perspectives

---

## **Appendix: Figure Specifications for Generation**

### **Figure 1: Projection Motivation** (`fig1_projection_motivation.png`)
- **Size**: 800×400 pixels
- **Description**: Create a figure with two side-by-side subplots. Left subplot: 2D scatter plot showing two classes (use red and blue colors) with approximately 100 points each. Class 0 (blue) centered around (2, 3) with some spread, Class 1 (red) centered around (5, 6) with some spread. The classes should overlap somewhat when viewed in the original 2D space. Right subplot: Show a 1D histogram or strip plot of the same points projected onto a specific direction (the optimal LDA direction, approximately 45-degree angle). After projection, the two classes should show clear separation with minimal overlap. Add a diagonal line on the left plot showing the projection direction. Use clear axis labels and a legend.

### **Figure 2: Good vs Bad Projections** (`fig2_good_bad_projections.png`)
- **Size**: 800×400 pixels
- **Description**: Create a figure with two side-by-side subplots showing the same 2D data (two classes, approximately 80 points each). Left subplot titled "Bad Projection": Show a 2D scatter plot with two classes (blue and red) and a projection line that is nearly horizontal, causing the classes to overlap heavily when projected onto this line. Below the scatter plot, show a 1D representation of the projected points with significant overlap. Right subplot titled "Good Projection": Same 2D data but with a diagonal projection line that maximizes class separation. Below it, show the 1D projection with clear separation between blue and red points. Use arrows to indicate projection directions. Add annotations like "Much overlap" and "Clear separation".

### **Figure 3: Between-Class and Within-Class Variance** (`fig3_variance_components.png`)
- **Size**: 800×500 pixels
- **Description**: Create an illustrative diagram showing 1D projected data (horizontal axis). Draw two overlapping bell curves (Gaussian-like distributions) representing the two classes after projection - one in blue (left), one in red (right). Clearly mark with vertical lines and labels: (1) μ̃₀ (mean of blue class), (2) μ̃₁ (mean of red class), (3) horizontal double-headed arrow showing "Between-class variance" spanning the distance between the two means, (4) show the spread/width of each distribution with annotations indicating "Within-class variance" for each. Use shaded regions to indicate variance within each class. Make it visually clear that we want large between-class distance and small within-class spreads. Add title: "Components of Fisher's Criterion".

### **Figure 4: LDA Decision Boundary and Projection Direction** (`fig4_lda_decision_boundary.png`)
- **Size**: 700×600 pixels
- **Description**: Create a 2D scatter plot showing two classes (approximately 100 points each, blue and red). Fit an LDA classifier and plot the linear decision boundary as a solid black line cutting through the data. Draw a thick arrow perpendicular to the decision boundary indicating the projection direction **w** (this is the normal to the decision boundary). Optionally, show a few dotted lines from some points to the projection direction to illustrate the projection concept. Mark the two class means (μ₀ and μ₁) with larger symbols (e.g., crosses or stars). Use a legend and label axes as "Feature 1" and "Feature 2". Title: "LDA Decision Boundary and Projection Direction".

### **Figure 5: sklearn LDA Example** (`fig5_sklearn_lda_example.png`)
- **Size**: 700×600 pixels
- **Description**: Use a real-world looking dataset (e.g., first two features of Iris dataset with two classes only - setosa vs versicolor). Create a scatter plot showing: (1) training points for two classes (use different markers like circles and triangles, in blue and red), (2) test points (different opacity or outline), (3) LDA decision boundary as a solid line, (4) background shading or contour showing the classification regions (light blue and light red). Add a legend indicating training vs test points and class labels. Mark a few misclassified points (if any) with a special marker (e.g., 'X'). Title: "LDA Classification on Iris Dataset". Include a text annotation with accuracy score on the plot.

### **Figure 6: LDA vs QDA Decision Boundaries** (`fig6_lda_vs_qda_boundaries.png`)
- **Size**: 800×400 pixels
- **Description**: Create two side-by-side subplots. Generate synthetic 2D data with two classes that have different covariance structures - Class 0 (blue) should be roughly circular, Class 1 (red) should be elliptical/stretched with different orientation. Left subplot: Fit LDA and show the linear decision boundary (straight line) with background shading for classification regions. Title: "LDA (Linear Boundary)". Right subplot: Same data but fit QDA and show the curved/quadratic decision boundary with background shading. Title: "QDA (Quadratic Boundary)". Use approximately 150 points per class. Make it visually obvious that the QDA boundary better captures the elliptical shape of one class. Both subplots should use the same axis limits for easy comparison.

### **Figure 7: LDA vs Logistic Regression Comparison** (`fig7_lda_vs_logreg.png`)
- **Size**: 800×400 pixels
- **Description**: Create two side-by-side subplots with the same 2D dataset (approximately 100 points per class, two classes in blue and red). Left subplot: Fit LDA and plot the decision boundary with scatter points. Title: "LDA (Accuracy: XX.X%)". Right subplot: Fit Logistic Regression and plot its decision boundary with the same scatter points. Title: "Logistic Regression (Accuracy: XX.X%)". The decision boundaries should be similar but not identical - both linear but with slightly different angles. Use background shading to show classification regions. Add a text note below the figure: "Both produce linear boundaries, but from different approaches". This should visually demonstrate that both methods are reasonable for this data.

---

**End of Detailed Outline**