# MPS311/439: Lesson 5 Lab - Linear & Quadratic Discriminant Analysis

**Duration**: 50 minutes  
**Objective**: Learn to use LDA and QDA for classification, understand when to use each method, and compare with logistic regression

---

## Setup: Import Libraries & Load Data (5 minutes)

First, let's import everything we need and prepare the Iris dataset. **Copy and run this code block:**

```python
# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load Iris dataset
iris = load_iris()
X_full = iris.data
y_full = iris.target

# For visualization, we'll use only 2 features: sepal length and sepal width
# This makes it easy to plot decision boundaries
X = X_full[:, [0, 1]]  # Columns 0 and 1
feature_names = [iris.feature_names[0], iris.feature_names[1]]

# Split into train and test (70/30 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")
print(f"Number of classes: {len(iris.target_names)}")
print(f"Class names: {iris.target_names}")
print(f"Features used: {feature_names}")
print("\nSetup complete! ✓")
```

**What just happened?**
- We loaded the famous Iris dataset (3 species of flowers)
- Selected 2 features for easy visualization
- Split into 70% training, 30% test
- We have 3 classes: Setosa, Versicolor, Virginica

---

## Part 1: Linear Discriminant Analysis (LDA) (10 minutes)

### Background
LDA is a **generative classifier** that models each class as a Gaussian distribution. The key assumption: **all classes share the same covariance matrix** (same shape, different centers). This leads to **linear** decision boundaries.

### Task 1.1: Fit LDA Model

Let's create and train an LDA model.

**YOUR TASK**: Complete the code below by filling in the blanks.

```python
# Create LDA model
# n_components controls dimensionality reduction (we'll use 2 to keep it simple)
lda_model = LinearDiscriminantAnalysis(n_components=____)  # FILL IN: use 2

# Fit the model on training data
lda_model.fit(____, ____)  # FILL IN: what data and labels to use for training?

# Make predictions on test data
y_pred_lda = lda_model.predict(____)  # FILL IN: what data to predict on?

# Calculate accuracy
accuracy_lda = accuracy_score(y_test, y_pred_lda)
print(f"LDA Test Accuracy: {accuracy_lda:.3f}")
```

**Hint**: Use `X_train` and `y_train` for fitting, and `X_test` for prediction.

**AI Help**: Ask ChatGPT: "How do I use sklearn's LinearDiscriminantAnalysis to fit and predict?"

**Record your results**:
- LDA Test Accuracy: _________

---

### Task 1.2: Access Learned Parameters

LDA learns the mean (center) of each class and a shared covariance matrix. Let's look at these!

**Copy and run this code**:

```python
# Access the learned class means (mu_k for each class k)
class_means = lda_model.means_
print("Class Means (centers of each class):")
print(class_means)
print()

# Access the shared covariance matrix (Sigma)
shared_cov = lda_model.covariance_
print("Shared Covariance Matrix (same for all classes):")
print(shared_cov)
print()

# Access class priors (how common each class is)
priors = lda_model.priors_
print("Class Priors:")
for i, class_name in enumerate(iris.target_names):
    print(f"  {class_name}: {priors[i]:.3f}")
```

**Questions**:
1. How many class means are there? _________
2. The covariance matrix is shared across how many classes? _________
3. Which class is most common in the training data? _________

---

### Task 1.3: Visualize LDA Decision Boundaries

Let's see what the decision boundaries look like. **Copy and run this code**:

```python
# Create a mesh to plot decision boundaries
h = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Predict class for each point in the mesh
Z = lda_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolors='k', cmap='viridis')
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])
plt.title('LDA Decision Boundaries')
plt.show()
```

**Question**: Are the decision boundaries straight lines or curves? _________

---

## Part 2: Quadratic Discriminant Analysis (QDA) (10 minutes)

### Background
QDA is similar to LDA, but it allows **each class to have its own covariance matrix**. This means classes can have different shapes and orientations, leading to **curved** (quadratic) decision boundaries.

### Task 2.1: Fit QDA Model

**YOUR TASK**: Complete the code below (following the same pattern as LDA).

```python
# Create QDA model
qda_model = QuadraticDiscriminantAnalysis()  # QDA doesn't have n_components

# Fit the model on training data
qda_model.fit(____, ____)  # FILL IN: training data and labels

# Make predictions on test data
y_pred_qda = qda_model.predict(____)  # FILL IN: test data

# Calculate accuracy
accuracy_qda = accuracy_score(____, ____)  # FILL IN: true labels and predictions
print(f"QDA Test Accuracy: {accuracy_qda:.3f}")
```

**Hint**: Use the same data as LDA - `X_train`, `y_train`, `X_test`, and `y_test`.

**Record your results**:
- QDA Test Accuracy: _________

---

### Task 2.2: Compare Covariance Matrices

Unlike LDA, QDA learns a **separate covariance matrix for each class**.

**Copy and run this code**:

```python
# Access covariance matrices (one per class)
qda_covs = qda_model.covariance_
print("Number of covariance matrices in QDA:", len(qda_covs))
print()

print("QDA Covariance Matrix for Class 0 (Setosa):")
print(qda_covs[0])
print()

print("QDA Covariance Matrix for Class 1 (Versicolor):")
print(qda_covs[1])
print()

print("QDA Covariance Matrix for Class 2 (Virginica):")
print(qda_covs[2])
```

**Questions**:
1. How many covariance matrices does QDA learn? _________
2. Are the covariance matrices identical across classes? _________
3. Why is this different from LDA? _________

---

### Task 2.3: Visualize QDA Decision Boundaries

Let's see how QDA's boundaries differ from LDA's. **Copy and run this code**:

```python
# Predict class for each point in the mesh using QDA
Z_qda = qda_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z_qda = Z_qda.reshape(xx.shape)

# Plot
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z_qda, alpha=0.3, cmap='viridis')
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolors='k', cmap='viridis')
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])
plt.title('QDA Decision Boundaries')
plt.show()
```

**Questions**:
1. Are QDA's decision boundaries straight or curved? _________
2. Which model (LDA or QDA) has more flexible boundaries? _________

---

## Part 3: LDA vs Logistic Regression (10 minutes)

### Background
Both LDA and Logistic Regression produce **linear** decision boundaries, but they estimate parameters differently:
- **LDA**: Generative (models each class distribution)
- **Logistic Regression**: Discriminative (models the boundary directly)

### Task 3.1: Fit Logistic Regression

**YOUR TASK**: Complete the code below.

```python
# Create Logistic Regression model
# We'll use multi_class='multinomial' for 3-class classification
logreg_model = LogisticRegression(multi_class='multinomial', max_iter=200)

# Fit the model
logreg_model.fit(____, ____)  # FILL IN: training data and labels

# Make predictions
y_pred_logreg = logreg_model.predict(____)  # FILL IN: test data

# Calculate accuracy
accuracy_logreg = accuracy_score(y_test, y_pred_logreg)
print(f"Logistic Regression Test Accuracy: {accuracy_logreg:.3f}")
```

**Hint**: Use the same training and test data as before.

**AI Help**: Ask ChatGPT: "How do I use sklearn's LogisticRegression for multiclass classification?"

**Record your results**:
- Logistic Regression Test Accuracy: _________

---

### Task 3.2: Compare All Three Methods

Let's create a summary table comparing all three classifiers.

**YOUR TASK**: Fill in the blanks to complete the comparison.

```python
print("\n" + "="*60)
print("MODEL COMPARISON SUMMARY")
print("="*60)
print(f"{'Model':<25} {'Test Accuracy':>15} {'Boundary Type':>18}")
print("-"*60)
print(f"{'LDA':<25} {____:>15.3f} {'Linear':>18}")  # FILL IN: LDA accuracy
print(f"{'QDA':<25} {____:>15.3f} {'Quadratic':>18}")  # FILL IN: QDA accuracy
print(f"{'Logistic Regression':<25} {accuracy_logreg:>15.3f} {'Linear':>18}")
print("="*60)
```

**Hint**: Use the accuracy variables you calculated: `accuracy_lda` and `accuracy_qda`.

**Questions**:
1. Which model performed best on the test set? _________
2. Did QDA's flexibility help or hurt performance here? _________

---

### Task 3.3: Visualize All Three Side-by-Side

**Copy and run this code** to see all decision boundaries together:

```python
# Create figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# LDA
Z_lda = lda_model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
axes[0].contourf(xx, yy, Z_lda, alpha=0.3, cmap='viridis')
axes[0].scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolors='k', cmap='viridis')
axes[0].set_title(f'LDA (Acc: {accuracy_lda:.3f})')
axes[0].set_xlabel(feature_names[0])
axes[0].set_ylabel(feature_names[1])

# QDA
Z_qda = qda_model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
axes[1].contourf(xx, yy, Z_qda, alpha=0.3, cmap='viridis')
axes[1].scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolors='k', cmap='viridis')
axes[1].set_title(f'QDA (Acc: {accuracy_qda:.3f})')
axes[1].set_xlabel(feature_names[0])

# Logistic Regression
Z_log = logreg_model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
axes[2].contourf(xx, yy, Z_log, alpha=0.3, cmap='viridis')
axes[2].scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolors='k', cmap='viridis')
axes[2].set_title(f'Logistic Reg (Acc: {accuracy_logreg:.3f})')
axes[2].set_xlabel(feature_names[0])

plt.tight_layout()
plt.show()
```

**Question**: Which boundaries look most similar to each other? _________

---

## Part 4: When to Use Which? (10 minutes)

### Background
Different models work better in different situations. Let's experiment!

### Task 4.1: Small Training Dataset Experiment

What happens when we have very little training data?

**YOUR TASK**: Complete the code to train all three models on a **smaller** dataset.

```python
# Create a smaller training set (only 30 samples)
X_train_small = X_train[:30]
y_train_small = y_train[:30]

print(f"Small training set size: {len(X_train_small)}")

# Train LDA on small dataset
lda_small = LinearDiscriminantAnalysis(n_components=2)
lda_small.fit(X_train_small, y_train_small)
acc_lda_small = lda_small.score(X_test, y_test)

# Train QDA on small dataset
qda_small = QuadraticDiscriminantAnalysis()
qda_small.fit(____, ____)  # FILL IN: use small training data
acc_qda_small = qda_small.score(____, ____)  # FILL IN: test on X_test and y_test

# Train Logistic Regression on small dataset
logreg_small = LogisticRegression(multi_class='multinomial', max_iter=200)
logreg_small.fit(X_train_small, y_train_small)
acc_logreg_small = logreg_small.score(X_test, y_test)

print("\nTest Accuracy with Small Training Set:")
print(f"  LDA: {acc_lda_small:.3f}")
print(f"  QDA: {acc_qda_small:.3f}")
print(f"  Logistic Regression: {acc_logreg_small:.3f}")
```

**Hint**: Use `X_train_small` and `y_train_small` for fitting.

**Questions**:
1. Which model performed worst with small data? _________
2. Why might QDA struggle with limited data? _________

---

### Task 4.2: Add Noise to Test Robustness

Let's add some random noise to the features and see which model is most robust.

**Copy and run this code**:

```python
# Add Gaussian noise to test data
np.random.seed(42)
noise = np.random.normal(0, 0.5, X_test.shape)
X_test_noisy = X_test + noise

# Evaluate all models on noisy data
acc_lda_noisy = lda_model.score(X_test_noisy, y_test)
acc_qda_noisy = qda_model.score(X_test_noisy, y_test)
acc_logreg_noisy = logreg_model.score(X_test_noisy, y_test)

print("\nTest Accuracy on Noisy Data:")
print(f"  LDA: {acc_lda_noisy:.3f}")
print(f"  QDA: {acc_qda_noisy:.3f}")
print(f"  Logistic Regression: {acc_logreg_noisy:.3f}")

print("\nAccuracy Drop:")
print(f"  LDA: {accuracy_lda - acc_lda_noisy:.3f}")
print(f"  QDA: {accuracy_qda - acc_qda_noisy:.3f}")
print(f"  Logistic Regression: {accuracy_logreg - acc_logreg_noisy:.3f}")
```

**Questions**:
1. Which model had the smallest accuracy drop? _________
2. Is Logistic Regression more robust to noise than LDA? _________

---

## Part 5: Reflection Questions (5 minutes)

Based on everything you've learned today, answer these questions:

### Question 1: Equal Covariance Assumption
**Why does LDA assume all classes have equal covariance? What happens to the decision boundary if they have equal covariance vs different covariances?**

Your answer: 

_________________________________________________________________

_________________________________________________________________

**Hint**: Think about what "equal covariance" means geometrically (same shape), and how that affects the math.

---

### Question 2: When to Use LDA vs QDA
**When would you choose LDA over QDA, and vice versa? Consider dataset size and class characteristics.**

Your answer:

_________________________________________________________________

_________________________________________________________________

**Hint**: Think about what you observed in Task 4.1 with the small dataset.

---

### Question 3: LDA vs Logistic Regression
**Both LDA and Logistic Regression produce linear boundaries. When would you prefer LDA? When would you prefer Logistic Regression?**

Your answer:

_________________________________________________________________

_________________________________________________________________

**Hint**: Consider the assumptions each method makes and the robustness experiments.

---

### Question 4: Practical Application
**Imagine you're working with medical data to diagnose diseases. You have 50 patients, 3 disease types, and 5 blood test measurements. Which classifier would you choose and why?**

Your answer:

_________________________________________________________________

_________________________________________________________________

---

## Summary: Core Learning Outcomes ✓

**Congratulations! You've learned to:**

- ✅ **Use sklearn's LinearDiscriminantAnalysis and QuadraticDiscriminantAnalysis** - You can now import, fit, predict, and evaluate both models
- ✅ **Explain why LDA assumes equal covariances** - Equal covariance makes decision boundaries linear; different covariances create quadratic boundaries
- ✅ **Describe when LDA vs QDA is appropriate** - LDA for small data or similar class spreads; QDA for large data with clearly different class shapes
- ✅ **Compare LDA vs logistic regression performance** - Both produce linear boundaries but estimate parameters differently; LDA better for small Gaussian data, Logistic Regression more robust

**Key Takeaways:**
1. LDA models each class as a Gaussian with the same covariance → linear boundaries
2. QDA allows different covariances per class → curved boundaries (more flexible but needs more data)
3. With limited data, LDA often outperforms QDA
4. Logistic Regression is often more robust but makes fewer distributional assumptions

---

## Part 6: For MPS439 Students Only (30 minutes)

### Advanced Task 6.1: Implement LDA from Scratch

Now let's implement LDA ourselves to understand what's happening under the hood!

**Background**: LDA involves three steps:
1. Calculate the mean of each class
2. Calculate the pooled (shared) covariance matrix
3. Compute discriminant functions and classify

**YOUR TASK**: Complete the implementation below.

```python
def lda_from_scratch(X_train, y_train, X_test):
    """
    Implement LDA from scratch.
    
    Parameters:
    - X_train: Training features (n_samples, n_features)
    - y_train: Training labels
    - X_test: Test features (n_samples_test, n_features)
    
    Returns:
    - y_pred: Predicted labels for X_test
    """
    # Step 1: Calculate class means
    classes = np.unique(y_train)
    n_classes = len(classes)
    n_features = X_train.shape[1]
    
    class_means = np.zeros((n_classes, n_features))
    for i, c in enumerate(classes):
        class_means[i] = X_train[y_train == c].____()  # FILL IN: what operation computes the mean?
    
    # Step 2: Calculate pooled covariance matrix
    # This is the weighted average of individual class covariances
    pooled_cov = np.zeros((n_features, n_features))
    n_total = len(y_train)
    
    for i, c in enumerate(classes):
        X_c = X_train[y_train == c]
        n_c = len(X_c)
        # Center the data (subtract class mean)
        X_centered = X_c - class_means[i]
        # Add to pooled covariance
        pooled_cov += (X_centered.T @ ____) / (n_total - n_classes)  # FILL IN: what should we multiply with?
    
    # Step 3: Calculate class priors (how common each class is)
    priors = np.zeros(n_classes)
    for i, c in enumerate(classes):
        priors[i] = np.sum(y_train == c) / len(y_train)
    
    # Step 4: Compute discriminant functions for each test sample
    # For each test point, calculate delta_k(x) for all classes k
    # Then pick the class with highest delta_k(x)
    
    # Inverse of covariance matrix (needed for discriminant function)
    cov_inv = np.linalg.inv(pooled_cov)
    
    # Compute discriminant scores
    n_test = X_test.shape[0]
    scores = np.zeros((n_test, n_classes))
    
    for k in range(n_classes):
        # Discriminant function: delta_k(x) = x^T * Sigma^-1 * mu_k - 0.5 * mu_k^T * Sigma^-1 * mu_k + log(prior_k)
        # Let's compute this step by step
        
        # Linear term: x^T * Sigma^-1 * mu_k
        linear_term = X_test @ cov_inv @ ____  # FILL IN: multiply with what?
        
        # Quadratic term: -0.5 * mu_k^T * Sigma^-1 * mu_k (constant for each class)
        quadratic_term = -0.5 * class_means[k] @ cov_inv @ class_means[k]
        
        # Prior term: log(prior_k)
        prior_term = np.log(priors[k])
        
        scores[:, k] = linear_term + quadratic_term + prior_term
    
    # Predict: choose class with highest score
    y_pred = np.argmax(____, axis=1)  # FILL IN: what to take argmax of?
    
    return y_pred, class_means, pooled_cov
```

**Hints**:
- For mean: use `.mean(axis=0)`
- For covariance: X_centered.T @ X_centered
- For linear term: multiply with `class_means[k]`
- For prediction: take argmax of `scores`

**AI Help**: Ask ChatGPT: "How do I compute a covariance matrix manually in numpy?"

---

### Advanced Task 6.2: Test Your Implementation

**YOUR TASK**: Use your implementation and compare with sklearn.

```python
# Test your implementation
y_pred_custom, means_custom, cov_custom = lda_from_scratch(X_train, y_train, X_test)

# Calculate accuracy
acc_custom = accuracy_score(____, ____)  # FILL IN: true labels and predictions

# Compare with sklearn's LDA
print(f"Custom LDA Test Accuracy: {acc_custom:.3f}")
print(f"sklearn LDA Test Accuracy: {accuracy_lda:.3f}")
print(f"Difference: {abs(acc_custom - accuracy_lda):.3f}")

# Compare learned parameters
print("\nClass Means Comparison (first class only):")
print(f"Custom:  {means_custom[0]}")
print(f"sklearn: {lda_model.means_[0]}")
```

**Hint**: Use `y_test` and `y_pred_custom` to calculate accuracy.

**Question**: How close is your implementation to sklearn's? _________

---

### Advanced Task 6.3: Derive Decision Boundary (2-Class Case)

For the 2-class case (e.g., Setosa vs Versicolor), the decision boundary is where the two discriminant functions are equal.

**Mathematical Derivation**:

The decision boundary between class 0 and class 1 occurs where:
$$\delta_0(x) = \delta_1(x)$$

Expanding the discriminant functions:
$$x^T\Sigma^{-1}\mu_0 - \frac{1}{2}\mu_0^T\Sigma^{-1}\mu_0 + \log(\pi_0) = x^T\Sigma^{-1}\mu_1 - \frac{1}{2}\mu_1^T\Sigma^{-1}\mu_1 + \log(\pi_1)$$

Rearranging:
$$x^T\Sigma^{-1}(\mu_0 - \mu_1) = \frac{1}{2}(\mu_0^T\Sigma^{-1}\mu_0 - \mu_1^T\Sigma^{-1}\mu_1) + \log\left(\frac{\pi_1}{\pi_0}\right)$$

This is a **linear equation** in $x$, which means the decision boundary is a **straight line** (or hyperplane in higher dimensions).

**YOUR TASK**: Let's verify this mathematically for Iris classes 0 and 1.

```python
# Extract data for only classes 0 and 1
mask = (y_train == 0) | (y_train == 1)
X_train_binary = X_train[mask]
y_train_binary = y_train[mask]

mask_test = (y_test == 0) | (y_test == 1)
X_test_binary = X_test[mask_test]
y_test_binary = y_test[mask_test]

# Fit LDA on binary problem
lda_binary = LinearDiscriminantAnalysis()
lda_binary.fit(X_train_binary, y_train_binary)

# Extract parameters
mu_0 = lda_binary.means_[0]
mu_1 = lda_binary.means_[1]
sigma_inv = np.linalg.inv(lda_binary.covariance_)
pi_0 = lda_binary.priors_[0]
pi_1 = lda_binary.priors_[1]

# Calculate decision boundary coefficients
# w = Sigma^-1 * (mu_0 - mu_1)
w = sigma_inv @ (mu_0 - mu_1)

# b = 0.5 * (mu_0^T * Sigma^-1 * mu_0 - mu_1^T * Sigma^-1 * mu_1) + log(pi_1/pi_0)
b = 0.5 * (mu_0 @ sigma_inv @ mu_0 - mu_1 @ sigma_inv @ mu_1) + np.log(pi_1 / pi_0)

print("Decision Boundary Equation: w^T * x = b")
print(f"w = {w}")
print(f"b = {b:.4f}")

# The decision boundary line is: w[0]*x + w[1]*y = b
# Rearranging: y = (b - w[0]*x) / w[1]
print(f"\nDecision line: {w[1]:.4f}*y = {b:.4f} - {w[0]:.4f}*x")
```

---

### Advanced Task 6.4: Visualize the Mathematical Boundary

**Copy and run this code** to plot the decision boundary you derived:

```python
# Plot the data
plt.figure(figsize=(8, 6))
plt.scatter(X_train_binary[y_train_binary==0, 0], 
            X_train_binary[y_train_binary==0, 1], 
            label='Class 0', alpha=0.6)
plt.scatter(X_train_binary[y_train_binary==1, 0], 
            X_train_binary[y_train_binary==1, 1], 
            label='Class 1', alpha=0.6)

# Plot the decision boundary
x_boundary = np.linspace(X_train_binary[:, 0].min(), X_train_binary[:, 0].max(), 100)
y_boundary = (b - w[0] * x_boundary) / w[1]
plt.plot(x_boundary, y_boundary, 'r--', linewidth=2, label='Decision Boundary')

plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])
plt.title('LDA Decision Boundary (Derived from Math)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

**Question**: Does this line match sklearn's decision boundary? _________

---

## Advanced Summary: What MPS439 Students Learned ✓

**Beyond the core content, you now understand:**

- ✅ **Derived LDA decision boundaries mathematically** - You've seen the full derivation from discriminant functions to linear boundaries
- ✅ **Implemented LDA in Python** - Not just using sklearn as a black box, but understanding the actual computations
- ✅ **Computed pooled covariance** - You understand how LDA combines information across classes
- ✅ **Verified mathematical theory** - Your implementation matches sklearn's results

**Key insights:**
1. LDA's "linear" comes from equal covariance causing quadratic terms to cancel
2. The discriminant function is just a weighted inner product plus a constant
3. Everything boils down to: calculate means, calculate shared covariance, compute scores, pick best class
4. The decision boundary equation $w^Tx = b$ is a straight line

---

## Reflection for MPS439 Students

Based on your implementation:

1. **What is the computational complexity of LDA?**
   
   Answer: _________________________________________________

2. **Why is the covariance matrix inverted in the discriminant function?**
   
   Answer: _________________________________________________

3. **How would you modify your code to implement QDA?**
   
   Answer: _________________________________________________

---

**Lab Complete!** 🎉

You now have both practical and theoretical understanding of LDA and QDA. Next week: Decision Trees - a completely different approach!
