# Lesson 6 Decision Trees - Detailed Slide Outline

## Slide 1: Title Slide
- **Title**: Lesson 6: Decision Trees
- **Subtitle**: From Linear Boundaries to Hierarchical Decisions
- **Course info**: MPS311/439 Machine Learning
- **Instructor**: Dr. Wei Xing
- **Date**: November 2025

---

## Slide 2: Where We Left Off
- **Header**: Last Week - Linear Discriminant Analysis
- **Content**:
  - Recap: LDA/QDA assume Gaussian distributions
  - LDA: Equal covariances → linear boundaries
  - QDA: Different covariances → quadratic boundaries
  - Both work well when assumptions hold
- **Transition box**: "But what if our data doesn't fit these neat assumptions?"
- **Visual suggestion**: Simple icons or shapes representing Gaussian distributions

---

## Slide 3: The Challenge - When Linear Methods Fail
- **Header**: The XOR Problem
- **Main visual**: Import `./figures/fig_xor_linear_fail.png` (shows XOR pattern with no linear separator)
- **Key points**:
  - Four clusters in a diagonal pattern
  - No straight line can separate blue from red points
  - Even QDA struggles with this pattern
  - Simple pattern, but defeats linear/quadratic methods
- **Callout box**: "We need a fundamentally different approach!"

---

## Slide 4: How Humans Actually Make Decisions
- **Header**: Thinking Like Humans - Hierarchical Questions
- **Main content**:
  - Humans don't calculate weighted sums
  - We ask sequences of simple yes/no questions
- **Example in visual tree format**:
  - "Should I go running today?"
    - Is it raining? → Yes: Stay home / No: Continue
    - Is temp > 5°C? → No: Too cold / Yes: Continue
    - Is temp < 30°C? → No: Too hot / Yes: Go running!
- **Other examples** (brief list):
  - Medical diagnosis: "Fever?" → "Throat swollen?" → ...
  - Troubleshooting: "Plugged in?" → "Power on?" → ...
- **Bottom callout**: "Can we teach machines to think this way? YES!"

---

## Slide 5: What is a Decision Tree?
- **Header**: Decision Tree - The Core Idea
- **Definition box**: "A flowchart-like structure that makes predictions through hierarchical yes/no questions"
- **Three key components** (with icons):
  1. **Internal nodes**: Questions/tests about features
  2. **Branches**: Outcomes of tests (yes/no)
  3. **Leaf nodes**: Final predictions (class labels)
- **Analogy**: "Like playing '20 Questions' to classify data"
- **Simple diagram**: Generic tree structure showing these components

---

## Slide 6: Visual Example - Iris Classification
- **Header**: A Real Decision Tree
- **Main visual**: Import `./figures/fig_simple_tree_example.png` (Iris tree, depth=3)
- **Walk-through example box**:
  - Sample: petal width = 1.5 cm, petal length = 4.5 cm
  - Path: "petal width ≤ 0.8?" → No (go right)
  - → "petal width ≤ 1.75?" → Yes (go left)
  - → "petal length ≤ 4.95?" → Yes (go left)
  - → Predict: **Versicolor**!
- **Caption**: "This tree achieves ~96% accuracy with just 3 levels"

---

## Slide 7: How to Read a Tree Node
- **Header**: Anatomy of a Decision Node
- **Visual**: Enlarged single node from a tree showing all components
- **Each component explained**:
  - **Top line**: Decision rule (e.g., "petal width ≤ 0.8")
  - **gini**: Measure of impurity (how mixed the classes are)
  - **samples**: Number of training samples reaching this node
  - **value**: Array [class_0_count, class_1_count, ...]
  - **class**: Majority class (what this node would predict)
  - **Color intensity**: Darker = more pure (more samples of majority class)
- **Practice tip**: "You can trace any prediction by following the path from root to leaf"

---

## Slide 8: Building Trees - The Key Question
- **Header**: How Do We Build a Tree?
- **The core challenge box**: "At each node: Which feature and threshold should we split on?"
- **Two principles** (side by side):
  - **Maximize distance**: Between class means after split
  - **Minimize spread**: Within each class after split
- **Goal statement**: "Find splits that create PURE children nodes"
- **Definition of purity**:
  - Pure node = all samples belong to one class
  - Impure node = mixed classes
- **Visual suggestion**: Simple diagram showing good split (separated colors) vs bad split (mixed colors)

---

## Slide 9: Measuring Purity - Gini Impurity
- **Header**: Gini Impurity - Measuring How Mixed a Node Is
- **Formula box**: 
  - For binary classification: Gini = 1 - p₁² - p₂² = 2p₁(1-p₁)
  - Where p₁ = proportion of class 1
- **Interpretation**:
  - Range: [0, 0.5] for binary classification
  - Gini = 0: Perfect purity (all one class) ✓
  - Gini = 0.5: Maximum impurity (50-50 split) ✗
- **Example calculation**:
  - Node with 60 Class A, 40 Class B samples
  - p₁ = 0.6, p₂ = 0.4
  - Gini = 1 - (0.6² + 0.4²) = 1 - (0.36 + 0.16) = 0.48
- **Visual**: Simple bar chart showing Gini values for different class proportions

---

## Slide 10: The Greedy Algorithm
- **Header**: Building a Tree - Step by Step
- **Algorithm flowchart**:
  1. Start with all data at root
  2. For each node:
     - Try all features and all thresholds
     - Calculate weighted Gini for each split
     - Choose split that minimizes weighted Gini
  3. Recursively apply to children
  4. Stop when: pure node OR stopping criterion met
- **Key term highlight**: "GREEDY = Make locally optimal choice at each step"
- **Stopping criteria** (brief list):
  - Max depth reached
  - Node too small to split
  - Perfect purity achieved
- **Note box**: "This is the CART algorithm (Classification and Regression Trees)"

---

## Slide 11: Implementation in Python
- **Header**: Decision Trees in sklearn - Incredibly Simple!
- **Code block** (~10 lines):
```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# 1. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# 2. Create and train
clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(X_train, y_train)

# 3. Evaluate
accuracy = clf.score(X_test, y_test)
print(f"Accuracy: {accuracy:.3f}")
```
- **Highlight box**: "Just 3 steps: Split data → Train tree → Evaluate"
- **Bottom note**: "We'll see visualizations and feature importance next!"

---

## Slide 12: Feature Importance
- **Header**: Which Features Matter Most?
- **Concept explanation**:
  - Trees automatically rank feature importance
  - Importance = how much each feature reduces impurity across all splits
  - Values sum to 1.0
- **Example output** (with bar chart visual):
  - petal_width: 0.65
  - petal_length: 0.35
  - sepal_width: 0.00
  - sepal_length: 0.00
- **Code snippet**:
```python
importances = clf.feature_importances_
# Returns array of importance values
```
- **Interpretation box**: "Features with 0.0 importance weren't used in any split"
- **Why useful**: Understand your data, guide feature engineering, simplify model

---

## Slide 13: The Overfitting Problem
- **Header**: The Perfect Training Accuracy Trap
- **Side-by-side comparison**:
  - **Left - Shallow Tree** (depth=3):
    - Training: 85%
    - Test: 84%
    - Simple boundary
  - **Right - Deep Tree** (depth=15):
    - Training: 100% ✗
    - Test: 78% ✗
    - Complex, jagged boundary
- **Main visual**: Import `./figures/fig_overfitting_comparison.png` (top row showing boundaries)
- **Key insight box**: "Higher training accuracy doesn't mean better! The gap between train and test is the problem."
- **Definition**: "Overfitting = memorizing training data instead of learning patterns"

---

## Slide 14: Overfitting Visualization
- **Header**: How Accuracy Changes with Tree Depth
- **Main visual**: Import `./figures/fig_overfitting_comparison.png` (bottom-left plot: accuracy vs depth)
- **Key observations**:
  - Training accuracy (blue): monotonically increases → 100%
  - Test accuracy (red): peaks around depth 5-6, then declines
  - The gap between them grows with depth
- **The sweet spot**: Highlight the peak test accuracy region
- **Callout boxes**:
  - "Too shallow" → underfitting (both accuracies low)
  - "Too deep" → overfitting (large train-test gap)
  - "Just right" → best generalization
- **Analogy**: "Like memorizing exam answers vs understanding concepts"

---

## Slide 15: Controlling Overfitting - Hyperparameters
- **Header**: Three Key Hyperparameters to Control Complexity
- **Three hyperparameters explained** (in columns):

  1. **max_depth**
     - Limits maximum tree depth
     - Smaller → simpler tree
     - Typical values: 3-10
     - Example: `max_depth=5`

  2. **min_samples_split**
     - Minimum samples to split a node
     - Larger → fewer splits
     - Typical values: 2-50
     - Example: `min_samples_split=20`

  3. **min_samples_leaf**
     - Minimum samples in each leaf
     - Larger → smoother boundaries
     - Typical values: 1-20
     - Example: `min_samples_leaf=5`

- **Bottom recommendation box**: "Start with max_depth=5, then tune based on results"

---

## Slide 16: Practical Guidelines
- **Header**: How to Train Trees in Practice
- **Starting configuration** (code box):
```python
clf = DecisionTreeClassifier(
    max_depth=5,           # Moderate depth
    min_samples_split=10,  # Don't split tiny nodes
    min_samples_leaf=5,    # No tiny leaves
    random_state=42
)
```
- **Tuning decision tree**:
  - **If underfitting** (both train & test accuracy low):
    - Increase `max_depth`
    - Decrease `min_samples_split` and `min_samples_leaf`
  - **If overfitting** (high train, low test):
    - Decrease `max_depth`
    - Increase `min_samples_split` and `min_samples_leaf`
- **Golden rules**:
  - Always use cross-validation (not just single train/test split)
  - Monitor BOTH training and test performance
  - Start simple, add complexity only if needed

---

## Slide 17: When to Use Decision Trees
- **Header**: Decision Trees vs Linear Methods
- **Comparison table**:

| Aspect | Linear Methods (LR, LDA) | Decision Trees |
|--------|-------------------------|----------------|
| Decision boundary | Linear/Quadratic | Complex, axis-aligned |
| Assumptions | Gaussian/Linear | None |
| Interpretability | High (coefficients) | High (tree structure) |
| Feature scaling | Required | Not required |
| Overfitting risk | Low | **High** |
| Handles non-linearity | Manual | Automatic |

- **When to use trees** (checkmarks):
  - ✓ Need interpretability
  - ✓ Data clearly non-linear
  - ✓ Mixed feature types (numerical & categorical)
  - ✓ No time for feature scaling

- **When to avoid** (X marks):
  - ✗ Very small dataset (< 100 samples)
  - ✗ Need stable predictions
  - ✗ Linear patterns work well

---

## Slide 18: Looking Ahead
- **Header**: What's Next?
- **Next week box** (prominent):
  - **Lesson 7: Principal Component Analysis (PCA)**
  - Big shift: Supervised → **Unsupervised learning**
  - No labels! Find structure in X alone
  - Dimensionality reduction: 100D → 2D
  - Understand what's "important" in high-dimensional data
- **For MPS439 students** (optional section):
  - Ensemble methods in lecture notes:
    - Random Forests: Average many trees → reduce overfitting
    - XGBoost: State-of-the-art gradient boosting
    - Trade-off: More accuracy, less interpretability
  - Read Section 9 of lecture notes for details
- **Connection**: "Trees: feature importance for prediction. PCA: principal components for variance"

---

## Slide 19: Key Takeaways
- **Header**: What We Learned Today
- **Main points** (6 items with icons):
  1. **Hierarchical decisions**: Trees ask yes/no questions sequentially
  2. **Gini impurity**: Measures how mixed a node is (want low values)
  3. **Greedy algorithm**: Builds tree by choosing locally optimal splits
  4. **Overfitting**: Deep trees memorize → control with hyperparameters
  5. **Easy implementation**: Just 3 lines in sklearn
  6. **Interpretability**: Can explain any prediction step-by-step
- **Bottom emphasis box**: "Start simple (depth=3-5), tune with cross-validation, visualize your trees!"

---

## Slide 20: Thank You & Next Steps
- **Header**: Thank You!
- **Questions?** (centered)
- **Contact info**:
  - Office Hours: Check course website
  - Email: w.xing@sheffield.ac.uk
- **Next session**: 
  - Lab on Friday: Hands-on with decision trees on real datasets
  - Lesson 7 Lecture: PCA and unsupervised learning
- **Encouragement box**: "Experiment with trees on different datasets! Try adjusting max_depth and observe the effects on train/test accuracy. Understanding overfitting is key to becoming a good ML practitioner."
- **Resources**: Lecture notes Section 1-7 (core), Section 8-9 (MPS439 optional)

---

# Appendix: Interactive Demonstrations for Google Colab

These demonstrations should be prepared as separate code cells in Google Colab and executed live during the lecture. Each demonstration should be self-contained with clear visualizations.

## Demo 1: XOR Problem - Visualizing the Challenge (Slide 3 context)
**Purpose**: Show students a concrete example where linear methods fail completely

**What to demonstrate**:
- Generate synthetic XOR dataset: 4 clusters arranged diagonally (100 points per cluster)
  - Cluster 1 (class 0): center at (0, 0)
  - Cluster 2 (class 0): center at (1, 1)
  - Cluster 3 (class 1): center at (0, 1)
  - Cluster 4 (class 1): center at (1, 0)
  - Add small Gaussian noise (std=0.1) to each cluster
- Plot the XOR pattern as a scatter plot (blue vs red)
- Train Logistic Regression on this data
- Visualize the linear decision boundary (a straight line)
- Show classification accuracy: ~50% (no better than random!)
- Optional: Try QDA and show it also struggles

**Key message**: "See? No straight line or simple curve can separate these classes. We need something fundamentally different."

**Visualizations needed**:
- Scatter plot showing the XOR pattern
- Decision boundary overlay (straight line from logistic regression)
- Accuracy score displayed prominently

---

## Demo 2: Simple Decision Tree on Iris (Slide 6 context)
**Purpose**: Show students their first real decision tree and how to interpret it

**What to demonstrate**:
- Load Iris dataset from sklearn
- Use only 2 features for simplicity: petal width and petal length
- Train DecisionTreeClassifier with max_depth=3
- Visualize the tree using plot_tree:
  - Set filled=True for colors
  - Set feature_names and class_names
  - Make figure large (figsize=(15, 10))
- Walk through one prediction manually:
  - Pick a specific sample from test set
  - Show its feature values
  - Trace through tree nodes from root to leaf
  - Show the final prediction
- Display training and test accuracy

**Key message**: "This tree structure is completely interpretable - we can explain exactly why it made each prediction."

**Visualizations needed**:
- Tree visualization with all node information visible
- Sample data point highlighted
- Accuracy metrics

---

## Demo 3: Decision Boundaries in 2D (Slide 6-7 context)
**Purpose**: Show how trees create axis-aligned rectangular regions

**What to demonstrate**:
- Use 2D subset of Iris (or generate synthetic 2-class data)
- Train decision tree with max_depth=3
- Create meshgrid covering the feature space
- Predict class for every point in meshgrid
- Visualize with contourf to show colored regions
- Overlay training data as scatter plot
- Show how boundary is made of horizontal/vertical lines only

**Key message**: "Trees can only split along axes - they create rectangular decision regions. But by combining many splits, they can approximate any boundary."

**Visualizations needed**:
- 2D contour plot showing colored decision regions
- Training data overlaid as scatter points
- Clear axis labels

---

## Demo 4: Effect of Tree Depth on Overfitting (Slide 13-14 context)
**Purpose**: Demonstrate the critical concept of overfitting with varying tree depth

**What to demonstrate**:
- Use Iris or a synthetic dataset (e.g., make_moons with noise)
- Split into train/test (70/30)
- Train trees with depths from 1 to 15
- For each depth:
  - Calculate training accuracy
  - Calculate test accuracy
  - Store both values
- Create two visualizations:
  
  **Plot 1**: Decision boundaries for depth=2, 5, 10, 15 (2x2 subplot)
  - Show how boundary becomes increasingly complex
  - Shallow tree: smooth, simple regions
  - Deep tree: jagged, overly complex regions
  
  **Plot 2**: Line plot of accuracy vs depth
  - Training accuracy (blue solid line): increases monotonically
  - Test accuracy (red dashed line): peaks then declines
  - Mark the "sweet spot" (best test accuracy)
  - Shade the overfitting region

**Key message**: "Watch how test accuracy peaks around depth 5-6, then drops even as training accuracy reaches 100%. This gap is overfitting!"

**Visualizations needed**:
- 2x2 subplot of decision boundaries at different depths
- Dual-line plot showing train/test accuracy vs depth
- Annotations marking underfitting, optimal, and overfitting regions

---

## Demo 5: Feature Importance Analysis (Slide 12 context)
**Purpose**: Show how to interpret which features drive the tree's decisions

**What to demonstrate**:
- Train a tree on Iris (all 4 features)
- Extract feature importances: `clf.feature_importances_`
- Create horizontal bar chart sorted by importance
- Show that some features may have zero importance
- Relate back to tree visualization: verify that important features appear in top splits
- Optional: Compare importance across different depths

**Key message**: "The tree tells us petal measurements are crucial, while sepal features barely matter for this classification task."

**Visualizations needed**:
- Horizontal bar chart of feature importances
- Features sorted from most to least important
- Values displayed on bars

---

## Demo 6: Hyperparameter Tuning Example (Slide 15-16 context)
**Purpose**: Show practical workflow for finding good hyperparameters

**What to demonstrate**:
- Use Wine or Breast Cancer dataset from sklearn
- Define parameter grid:
  - max_depth: [3, 5, 7, 10]
  - min_samples_split: [2, 10, 20]
  - min_samples_leaf: [1, 5, 10]
- Use GridSearchCV with 5-fold cross-validation
- Fit the grid search (this may take 10-15 seconds - explain what's happening)
- Display best parameters found
- Show best cross-validation score
- Compare to baseline (default parameters)
- Visualize: heatmap showing accuracy for different max_depth values

**Key message**: "Grid search tries all combinations systematically. Here we tested 36 different configurations to find the best one!"

**Visualizations needed**:
- Text output showing best parameters
- Bar chart comparing baseline vs tuned model accuracy
- Optional: heatmap of accuracy vs max_depth and min_samples_split

---

## Demo 7: Trees vs Logistic Regression Comparison (Slide 17 context)
**Purpose**: Direct comparison to solidify understanding of when to use which method

**What to demonstrate**:
- Use two datasets:
  
  **Dataset 1**: Linearly separable (e.g., Iris setosa vs others)
  - Train both logistic regression and decision tree
  - Show similar accuracy (both ~95%+)
  - Visualize boundaries: both are essentially straight lines
  
  **Dataset 2**: Non-linear (e.g., make_moons with noise)
  - Train both methods
  - Logistic regression: ~75% accuracy (straight boundary fails)
  - Decision tree: ~90% accuracy (captures non-linearity)
  - Visualize boundaries side-by-side

**Key message**: "For linear patterns, both work fine. For non-linear patterns, trees have the advantage. But trees need careful regularization!"

**Visualizations needed**:
- 2x2 subplot grid:
  - Row 1: Linear data (LR boundary, Tree boundary)
  - Row 2: Non-linear data (LR boundary, Tree boundary)
- Accuracy scores annotated on each plot

---

## Technical Notes for All Demos:

**General setup for all demonstrations**:
- Import statements at the top of notebook
- Use consistent random_state=42 for reproducibility
- Use clear, large fonts for all plots (font size 12-14)
- Color scheme: use colorblind-friendly palettes
- Add titles and axis labels to all plots
- Keep code concise and well-commented
- Print key metrics prominently

**Timing**:
- Each demo should take 2-3 minutes maximum
- Have pre-run cells ready to avoid waiting
- Focus on showing outputs and explaining, not typing code

**Pedagogical approach**:
- Start each demo with: "Let me show you what this looks like..."
- Pause after showing visualization to let students absorb
- Ask questions: "What do you notice here?" or "Why do you think...?"
- Connect back to theory: "Remember we said Gini measures purity? Look at these node colors..."