---
pagetitle: "Lab 5: Linear and Quadratic Discriminant Analysis"
---

# Lab 5: Linear and Quadratic Discriminant Analysis
## MPS311/439 - Machine Learning
**Lesson 5 Lab Session | Duration: 50 minutes**

---

## Introduction

Welcome to Lab 5! Today you'll explore **Linear Discriminant Analysis (LDA)** and **Quadratic Discriminant Analysis (QDA)** - classification methods that work by finding optimal projections of your data.

**What you'll learn today:**
- How to use sklearn's LDA and QDA classifiers
- When LDA works better than logistic regression
- How to visualize projections to understand what LDA is doing
- When to choose QDA over LDA (hint: it's about covariances!)

**Remember:** This lab uses a fill-in-the-blanks approach. Don't write code from scratch - just fill in the blanks marked with `____`. Focus on understanding the concepts!

---

## Setup: Import Libraries and Load Data

We'll use the famous **Iris dataset** - it has 3 classes of flowers with 4 measurements per flower. It's perfect for understanding LDA because the classes are naturally well-separated.

**Copy and run this code:**

```python
# Import packages
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

# Load iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")
print(f"Number of features: {X_train.shape[1]}")
print(f"Number of classes: {len(np.unique(y))}")
```

**What's happening here?**
- We have 4 features (sepal length, sepal width, petal length, petal width)
- We have 3 classes (setosa, versicolor, virginica)
- We split data 70/30 for training and testing

---

## Part 1: Your First LDA Model (10 minutes)

**Background:** LDA finds a projection that maximizes separation between classes. For 3 classes, it can project to at most 2 dimensions (number of classes - 1). Let's train an LDA classifier!

### Task 1.1: Create and train an LDA model

**Fill in the blanks below:**

```python
# Create an LDA model that projects to 2 dimensions
lda = LinearDiscriminantAnalysis(n_components=____)

# Train the model
lda.____(X_train, y_train)

# Make predictions on test set
y_pred = lda.____(X_test)

# Calculate accuracy
accuracy = lda.____(X_test, y_test)
print(f"LDA Test Accuracy: {accuracy:.3f}")
```

**Hints:**
- `n_components=2` means we project to 2D (the maximum for 3 classes)
- Method to train a model: `.fit(X, y)`
- Method to make predictions: `.predict(X)`
- Method to calculate accuracy: `.score(X, y)`

**AI Help:** Ask ChatGPT: "How do I use LinearDiscriminantAnalysis in sklearn? Show me fit, predict, and score methods."

---

## Part 2: LDA vs Logistic Regression (10 minutes)

**Background:** In the lecture, you learned that LDA takes a different approach than logistic regression. Let's compare them! Logistic regression directly models the decision boundary, while LDA first projects the data.

### Task 2.1: Train a Logistic Regression model

**Fill in the blanks:**

```python
# Create a logistic regression model
logreg = ____(max_iter=200, random_state=42)

# Train it
logreg.fit(____, ____)

# Calculate accuracy
logreg_accuracy = logreg.score(____, ____)

print(f"LDA Accuracy:              {accuracy:.3f}")
print(f"Logistic Regression Accuracy: {logreg_accuracy:.3f}")
```

**Hints:**
- Import: `from sklearn.linear_model import LogisticRegression`
- Model class name: `LogisticRegression`
- Training data: `X_train, y_train`
- Test data: `X_test, y_test`

**AI Help:** Ask ChatGPT: "How do I use LogisticRegression from sklearn?"

### Task 2.2: Interpret the results

Look at the two accuracy scores. Are they similar? Which one is higher?

Write your observation here:
```
Your answer: _______________________________________________
```

**Think about:** For the Iris dataset, both methods work well. But LDA has an advantage - it can also project data for visualization! Let's see that next.

---

## Part 3: Visualizing LDA Projections (10 minutes)

**Background:** LDA's superpower is that it finds a lower-dimensional projection where classes are well-separated. We can visualize this! The `.transform()` method projects our 4D data to 2D.

### Task 3.1: Project data using LDA

**Fill in the blanks:**

```python
# Project training and test data to 2D
X_train_lda = lda.____(X_train)
X_test_lda = lda.transform(____)

print(f"Original data shape: {X_train.shape}")
print(f"Projected data shape: {X_train_lda.shape}")
```

**Hint:** 
- Method to project data: `.transform(X)`
- We project both training and test data

**AI Help:** Ask ChatGPT: "What does the transform method do in LDA?"

### Task 3.2: Visualize the projection

**Copy and run this plotting code:**

```python
# Plot the 2D projection
plt.figure(figsize=(10, 4))

# Training data
plt.subplot(1, 2, 1)
plt.scatter(X_train_lda[:, 0], X_train_lda[:, 1], c=y_train, cmap='viridis', alpha=0.6)
plt.xlabel('First LDA Component')
plt.ylabel('Second LDA Component')
plt.title('Training Data (LDA Projection)')
plt.colorbar(label='Class')

# Test data
plt.subplot(1, 2, 2)
plt.scatter(X_test_lda[:, 0], X_test_lda[:, 1], c=y_test, cmap='viridis', alpha=0.6)
plt.xlabel('First LDA Component')
plt.ylabel('Second LDA Component')
plt.title('Test Data (LDA Projection)')
plt.colorbar(label='Class')

plt.tight_layout()
plt.show()
```

**What do you see?** The three classes should be clearly separated in this 2D space! This is what LDA does - it finds the best 2D projection for separating the 3 classes.

---

## Part 4: When LDA Fails - Enter QDA (10 minutes)

**Background:** LDA assumes all classes have the **same covariance** (same spread/shape). But what if classes have different spreads? That's when QDA (Quadratic Discriminant Analysis) shines! QDA allows each class to have its own covariance.

Let's create synthetic data where classes have different spreads to see this in action.

### Task 4.1: Create data with different covariances

**Copy and run this code to create synthetic data:**

```python
# Generate synthetic 2D data with different spreads
np.random.seed(42)

# Class 0: small spread
X_class0 = np.random.randn(100, 2) * 0.5 + np.array([0, 0])

# Class 1: large spread
X_class1 = np.random.randn(100, 2) * 2.0 + np.array([3, 3])

# Combine
X_synth = np.vstack([X_class0, X_class1])
y_synth = np.hstack([np.zeros(100), np.ones(100)])

# Split
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_synth, y_synth, test_size=0.3, random_state=42)

# Plot the data
plt.figure(figsize=(6, 6))
plt.scatter(X_synth[y_synth==0, 0], X_synth[y_synth==0, 1], alpha=0.5, label='Class 0 (small spread)')
plt.scatter(X_synth[y_synth==1, 0], X_synth[y_synth==1, 1], alpha=0.5, label='Class 1 (large spread)')
plt.legend()
plt.title('Synthetic Data with Different Covariances')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
```

**Notice:** Class 0 is tightly clustered (small spread), while Class 1 is spread out (large spread). They have **different covariances**!

### Task 4.2: Compare LDA vs QDA

**Fill in the blanks:**

```python
# Train LDA
lda_synth = LinearDiscriminantAnalysis()
lda_synth.fit(X_train_s, y_train_s)
lda_acc = lda_synth.____(X_test_s, y_test_s)

# Train QDA
qda_synth = ____(____=______)
qda_synth.fit(____, ____)
qda_acc = qda_synth.score(____, ____)

print(f"LDA Accuracy: {lda_acc:.3f}")
print(f"QDA Accuracy: {qda_acc:.3f}")
```

**Hints:**
- QDA class: `QuadraticDiscriminantAnalysis()`
- Training data: `X_train_s, y_train_s`
- Test data: `X_test_s, y_test_s`
- QDA doesn't need `n_components` parameter

**AI Help:** Ask ChatGPT: "How do I use QuadraticDiscriminantAnalysis in sklearn?"

### Task 4.3: Interpret the results

Which model performed better? Why?

```
Your answer: _______________________________________________
```

**Key insight:** When classes have different covariances (different spreads), QDA should outperform LDA because it doesn't assume equal covariances!

---

## Part 5: Reflection Questions (5 minutes)

Answer these questions based on what you've learned today:

**Question 1:** What does LDA's `.transform()` method do? Why is this useful?

```
Your answer: _______________________________________________
___________________________________________________________
```

**Question 2:** LDA assumes all classes have equal covariance. What does this mean in simple terms?

```
Your answer: _______________________________________________
___________________________________________________________
```

**Question 3:** When should you use QDA instead of LDA? Give a concrete example.

```
Your answer: _______________________________________________
___________________________________________________________
```

**Question 4:** In Part 2, you compared LDA and Logistic Regression. What's the main conceptual difference between these two approaches?

```
Your answer: _______________________________________________
___________________________________________________________
```

---

## Summary

**What you learned today:**
- ✅ How to use `LinearDiscriminantAnalysis` and `QuadraticDiscriminantAnalysis`
- ✅ How to visualize LDA projections with `.transform()`
- ✅ When LDA works well (similar covariances) and when QDA is better (different covariances)
- ✅ How LDA compares to logistic regression

**Key takeaway:** LDA finds projections that separate classes. It's powerful for both classification AND dimensionality reduction. When the equal covariance assumption is violated, use QDA!

---

## Part 6: Advanced Challenge - For MPS439 Students Only (30 minutes)

**Background:** Now you'll implement LDA from scratch for 2-class, 2D data. This will help you understand what sklearn is doing under the hood!

### Task 6.1: Implement LDA from scratch

You'll implement Fisher's Linear Discriminant for binary classification. Recall from lecture:

1. Calculate class means: $\boldsymbol{\mu}_0, \boldsymbol{\mu}_1$
2. Calculate within-class scatter matrix: $\mathbf{S}_W$
3. Find optimal direction: $\mathbf{w} = \mathbf{S}_W^{-1}(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0)$
4. Project data and classify based on threshold

**Fill in the implementation:**

```python
def lda_from_scratch(X_train, y_train, X_test):
    """
    Implement LDA for binary classification (2 classes, any dimensions)
    
    Parameters:
    - X_train: training features (n_samples, n_features)
    - y_train: training labels (n_samples,) - should be 0 or 1
    - X_test: test features (m_samples, n_features)
    
    Returns:
    - y_pred: predicted labels for X_test
    - w: projection direction
    """
    
    # Step 1: Separate data by class
    X_class0 = X_train[y_train == ____]
    X_class1 = X_train[y_train == ____]
    
    # Step 2: Calculate class means
    mu_0 = np.mean(____, axis=0)  # Mean of class 0
    mu_1 = np.____(X_class1, axis=____)  # Mean of class 1
    
    print(f"Class 0 mean: {mu_0}")
    print(f"Class 1 mean: {mu_1}")
    
    # Step 3: Calculate within-class scatter matrices
    # S_0 = sum of (x - mu_0)(x - mu_0)^T for all x in class 0
    S_0 = np.zeros((X_train.shape[1], X_train.shape[1]))
    for x in X_class0:
        diff = (x - mu_0).reshape(-1, 1)
        S_0 += diff @ ____  # outer product
    
    S_1 = np.zeros((X_train.shape[1], X_train.shape[1]))
    for x in ____:
        diff = (x - ____).reshape(-1, 1)
        S_1 += ____ @ diff.T
    
    # Total within-class scatter
    S_W = ____ + ____
    
    print(f"Within-class scatter matrix shape: {S_W.shape}")
    
    # Step 4: Calculate optimal projection direction
    # w = S_W^(-1) * (mu_1 - mu_0)
    S_W_inv = np.linalg.inv(____)
    mu_diff = ____ - ____
    w = S_W_inv @ ____
    
    print(f"Projection direction w: {w}")
    
    # Step 5: Project training data and find threshold
    # Project data onto w
    z_train_0 = X_class0 @ ____
    z_train_1 = X_class1 @ w
    
    # Threshold is midpoint between projected means
    threshold = (np.mean(____) + np.mean(____)) / 2
    
    print(f"Classification threshold: {threshold:.3f}")
    
    # Step 6: Project test data and classify
    z_test = ____ @ ____
    y_pred = (z_test > ____).astype(int)
    
    return y_pred, w

# Test your implementation on synthetic 2-class data
X_2class = X_synth[y_synth <= 1]  # Just use first 2 classes
y_2class = y_synth[y_synth <= 1]
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_2class, y_2class, test_size=0.3, random_state=42)

# Run your implementation
y_pred_scratch, w_scratch = lda_from_scratch(X_train_2, y_train_2, X_test_2)

# Calculate accuracy
accuracy_scratch = np.mean(y_pred_scratch == y_test_2)
print(f"\nYour LDA from scratch accuracy: {accuracy_scratch:.3f}")

# Compare with sklearn
lda_sklearn = LinearDiscriminantAnalysis()
lda_sklearn.fit(X_train_2, y_train_2)
accuracy_sklearn = lda_sklearn.score(X_test_2, y_test_2)
print(f"Sklearn LDA accuracy: {accuracy_sklearn:.3f}")
```

**Hints:**
- Class 0 has label 0, class 1 has label 1
- `axis=0` means compute mean along rows (one mean per column/feature)
- `.reshape(-1, 1)` converts 1D array to column vector
- `diff.T` is the transpose of diff
- Outer product: `diff @ diff.T`

**AI Help:** Ask ChatGPT: "Explain the mathematical steps in Fisher's Linear Discriminant for binary classification."

### Task 6.2: Visualize the projection direction

**Fill in the blanks to plot the decision boundary:**

```python
# Plot the data and projection direction
plt.figure(figsize=(8, 6))
plt.scatter(X_train_2[y_train_2==0, 0], X_train_2[y_train_2==0, 1], alpha=0.5, label='Class 0')
plt.scatter(X_train_2[y_train_2==1, 0], X_train_2[y_train_2==1, 1], alpha=0.5, label='Class 1')

# Plot projection direction as an arrow
origin = np.mean(X_train_2, axis=0)
plt.arrow(origin[0], origin[1], w_scratch[____]*2, w_scratch[____]*2, 
          head_width=0.3, head_length=0.4, fc='red', ec='red', linewidth=2, label='Projection direction w')

plt.legend()
plt.title('LDA Projection Direction (from scratch)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.axis('equal')
plt.grid(True, alpha=0.3)
plt.show()
```

**Hints:**
- The arrow shows the direction of w
- `w_scratch[0]` is the x-component, `w_scratch[1]` is the y-component

### Task 6.3: Understanding the math

Answer these questions to check your understanding:

**Question 1:** Why do we use $\mathbf{S}_W^{-1}$ (the inverse of within-class scatter)? What does this achieve?

```
Your answer: _______________________________________________
___________________________________________________________
```

**Question 2:** The threshold is the midpoint between projected class means. What happens if one class has many more samples than the other? Should we adjust the threshold?

```
Your answer: _______________________________________________
___________________________________________________________
```

**Question 3:** What would change in the code if we wanted to handle unequal class priors (different proportions of each class)?

```
Your answer: _______________________________________________
___________________________________________________________
```

---

**Congratulations!** You've completed the advanced section and implemented LDA from scratch. This is exactly what sklearn does internally (with more optimizations)!

---

**End of Lab Worksheet**

<!-- COURSE_FEEDBACK_QR:START -->
---

## 30-second feedback

What would help you learn better next time?

Scan to share anonymous feedback on today's lab.

![Feedback QR code for Lesson 05 lab](./feedback-qr.png){fig-align="center" width="180px"}
<!-- COURSE_FEEDBACK_QR:END -->
