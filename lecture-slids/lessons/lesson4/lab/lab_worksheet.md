---
pagetitle: "Lab 4: Logistic Regression & Classification Metrics"
---

# Lab 4: Logistic Regression & Classification Metrics
## MPS311/439 - Machine Learning
**Lesson 4 Lab Session | Duration: 50 minutes**

**Objective**: Learn to apply logistic regression, interpret classification metrics, and understand decision thresholds

---

## Setup: Import Libraries & Load Data (5 minutes)

First, let's import everything we need and prepare our dataset. **Copy and run this code block:**

```python
# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score

# Load breast cancer dataset (binary classification: malignant vs benign)
data = load_breast_cancer()
X = data.data
y = data.target

# Print basic information
print(f"Dataset shape: {X.shape}")
print(f"Number of samples: {X.shape[0]}")
print(f"Number of features: {X.shape[1]}")
print(f"Target classes: {data.target_names}")
print(f"Class distribution: {np.bincount(y)}")
print(f"  - Malignant (0): {np.sum(y == 0)}")
print(f"  - Benign (1): {np.sum(y == 1)}")

# Split into train and test (70/30 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"\nTraining set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")
print("\nSetup complete! ✓")
```

**What just happened?**
- We loaded the breast cancer dataset (569 tumors, 30 features each)
- Target: 0 = malignant (cancer), 1 = benign (not cancer)
- Split into 70% training, 30% test
- This is a **binary classification** problem!

---

## Part 1: Basic Logistic Regression & Probabilities (15 minutes)

### Background
Remember from the lecture: logistic regression uses the **sigmoid function** to convert a linear combination of features into a probability between 0 and 1. This probability tells us how confident the model is that a tumor is benign (class 1).

### Task 1.1: Fit a Logistic Regression Model

Let's train our first classifier!

**YOUR TASK**: Complete the code below by filling in the blanks.

```python
# Create a logistic regression model
model = LogisticRegression(max_iter=10000, random_state=42)

# Fit the model on training data
model.fit(____, ____)  # Fill in: what data should we train on?

print("Model trained! ✓")
print(f"Number of coefficients: {len(model.coef_[0])}")
```

**Hint**: Use `X_train` and `y_train` for training.

**AI Help**: Ask ChatGPT: "How do I fit a LogisticRegression model in sklearn?"

---

### Task 1.2: Make Predictions (Class Labels)

Now let's use our model to predict classes on the test set.

**YOUR TASK**: Complete the code below.

```python
# Predict class labels (0 or 1) for test set
y_pred = model.____(____)  # Fill in: method name and data to predict on

print(f"Predictions shape: {y_pred.shape}")
print(f"First 10 predictions: {y_pred[:10]}")
print(f"First 10 actual values: {y_test[:10]}")
```

**Hint**: Use the `.predict()` method with `X_test`.

**AI Help**: Ask ChatGPT: "What's the difference between .predict() and .predict_proba() in sklearn?"

---

### Task 1.3: Get Probability Predictions

Instead of just 0 or 1, we can get the **probability** that each tumor is benign.

**YOUR TASK**: Complete the code below.

```python
# Get probability predictions
y_pred_proba = model.predict_proba(____)  # Fill in: what data?

# predict_proba returns probabilities for BOTH classes [P(class 0), P(class 1)]
# We only need P(class 1) - probability of being benign
y_pred_proba_class1 = y_pred_proba[:, ____]  # Fill in: which column? (0 or 1)

print(f"Probability predictions shape: {y_pred_proba.shape}")
print(f"\nFirst 5 samples:")
print(f"P(malignant) | P(benign) | Predicted class | True class")
print("-" * 60)
for i in range(5):
    print(f"{y_pred_proba[i, 0]:.3f}        | {y_pred_proba[i, 1]:.3f}     | {y_pred[i]}               | {y_test[i]}")
```

**Hint**: Column 0 is P(class 0), column 1 is P(class 1). We want P(benign) which is class 1.

**Key Insight**: Notice that the two probabilities always sum to 1.0! And the predicted class is the one with probability > 0.5.

---

### Task 1.4: Visualize Probability Distribution

Let's see how confident our model is in its predictions.

**Copy and run this code**:

```python
# Separate probabilities by true class
proba_benign_actual = y_pred_proba_class1[y_test == 1]  # True benign cases
proba_malignant_actual = y_pred_proba_class1[y_test == 0]  # True malignant cases

# Plot histogram
plt.figure(figsize=(10, 5))

plt.hist(proba_benign_actual, bins=20, alpha=0.6, label='Actually Benign', color='green')
plt.hist(proba_malignant_actual, bins=20, alpha=0.6, label='Actually Malignant', color='red')

plt.axvline(0.5, color='black', linestyle='--', linewidth=2, label='Decision Threshold (0.5)')
plt.xlabel('Predicted Probability of Benign (Class 1)')
plt.ylabel('Count')
plt.title('Distribution of Predicted Probabilities')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"Benign tumors: {len(proba_benign_actual)} samples")
print(f"Malignant tumors: {len(proba_malignant_actual)} samples")
```

**Questions**:
1. Do most benign tumors have predicted probability > 0.5? _________
2. Do most malignant tumors have predicted probability < 0.5? _________
3. Are there any cases where the model is very uncertain (probability near 0.5)? _________

---

## Part 2: Confusion Matrix & Classification Metrics (15 minutes)

### Background
From the lecture: A **confusion matrix** shows how many predictions were correct or incorrect, broken down by class. It's the foundation for calculating precision, recall, and F1-score.

### Task 2.1: Create a Confusion Matrix

**YOUR TASK**: Complete the code below.

```python
# Calculate confusion matrix
cm = confusion_matrix(____, ____)  # Fill in: true labels and predicted labels

print("Confusion Matrix:")
print(cm)
print("\nInterpretation:")
print(f"True Negatives (TN):  {cm[0, 0]}  - Correctly predicted malignant")
print(f"False Positives (FP): {cm[0, 1]}  - Incorrectly predicted benign (missed cancer!)")
print(f"False Negatives (FN): {cm[1, 0]}  - Incorrectly predicted malignant (false alarm)")
print(f"True Positives (TP):  {cm[1, 1]}  - Correctly predicted benign")
```

**Hint**: The first argument is `y_test` (true labels), second is `y_pred` (predicted labels).

**AI Help**: Ask ChatGPT: "How do I interpret a confusion matrix for binary classification?"

**Visualize the confusion matrix** (copy and run):

```python
# Plot confusion matrix
plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap='Blues', interpolation='nearest')
plt.title('Confusion Matrix')
plt.colorbar()

# Add labels
classes = ['Malignant (0)', 'Benign (1)']
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes)
plt.yticks(tick_marks, classes)

# Add text annotations
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha='center', va='center', 
                color='white' if cm[i, j] > cm.max()/2 else 'black', 
                fontsize=20, fontweight='bold')

plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()
```

---

### Task 2.2: Calculate Accuracy

Accuracy is the simplest metric: what fraction of predictions were correct?

**YOUR TASK**: Complete the code.

```python
# Calculate accuracy
accuracy = accuracy_score(____, ____)  # Fill in: true and predicted labels

print(f"Accuracy: {accuracy:.3f}")
print(f"This means {accuracy*100:.1f}% of predictions were correct!")
```

**Hint**: Same arguments as confusion_matrix.

**Question**: Is 95% accuracy good enough for cancer detection? Why or why not? 

_________________________________________________________________________

---

### Task 2.3: Calculate Precision, Recall, and F1-Score

These metrics give us more detailed information than accuracy alone.

**YOUR TASK**: Complete the code below.

```python
# Calculate precision, recall, F1-score
precision = precision_score(y_test, y_pred)
recall = recall_score(____, ____)  # Fill in: same pattern
f1 = f1_score(____, ____)  # Fill in: same pattern

print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1-Score:  {f1:.3f}")

print("\nInterpretation:")
print(f"- Precision: Of all tumors we predicted as benign, {precision*100:.1f}% were actually benign")
print(f"- Recall:    Of all actual benign tumors, we correctly identified {recall*100:.1f}%")
print(f"- F1-Score:  Harmonic mean of precision and recall: {f1:.3f}")
```

**Hint**: Use `y_test` and `y_pred` for all three metrics.

**AI Help**: Ask ChatGPT: "Explain precision and recall with an example for cancer detection"

---

### Task 2.4: Understanding the Metrics

**YOUR TASK**: Answer these questions based on your results.

1. **Which is more important for cancer detection: precision or recall?**
   - [ ] Precision (avoid false alarms)
   - [ ] Recall (don't miss any cancer cases)
   
   **Why?** _________________________________________________________

2. **Calculate these metrics manually** (using your confusion matrix values):
   
   From your confusion matrix: TN = ____, FP = ____, FN = ____, TP = ____
   
   - Precision = TP / (TP + FP) = ____ / (____ + ____) = ____
   - Recall = TP / (TP + FN) = ____ / (____ + ____) = ____

3. **Do your manual calculations match sklearn's output?** _________

**Key Insight**: 
- **Precision** answers: "When we predict benign, how often are we right?"
- **Recall** answers: "Of all benign cases, how many did we catch?"
- **F1-score** balances both when we care about precision AND recall equally

---

## Part 3: Decision Threshold & ROC Curve (15 minutes)

### Background
So far, we've used the default threshold of 0.5: if P(benign) > 0.5, predict benign. But we can change this threshold! A **lower threshold** means we predict benign more often (higher recall, lower precision). A **higher threshold** means we're more cautious (lower recall, higher precision).

### Task 3.1: Experiment with Different Thresholds

Let's see what happens when we use threshold = 0.7 (more cautious).

**YOUR TASK**: Complete the code below.

```python
# Try threshold = 0.7 (higher threshold = more cautious)
threshold = 0.7
y_pred_new = (y_pred_proba_class1 >= ____).astype(int)  # Fill in: threshold value

# Calculate metrics with new threshold
cm_new = confusion_matrix(y_test, y_pred_new)
accuracy_new = accuracy_score(y_test, y_pred_new)
precision_new = precision_score(y_test, y_pred_new)
recall_new = recall_score(y_test, y_pred_new)
f1_new = f1_score(y_test, y_pred_new)

print(f"With threshold = {threshold}:")
print(f"Accuracy:  {accuracy_new:.3f} (was {accuracy:.3f})")
print(f"Precision: {precision_new:.3f} (was {precision:.3f})")
print(f"Recall:    {recall_new:.3f} (was {recall:.3f})")
print(f"F1-Score:  {f1_new:.3f} (was {f1:.3f})")

print(f"\nConfusion Matrix:")
print(cm_new)
```

**Hint**: Use the threshold value (0.7) in the comparison.

---

### Task 3.2: Compare Different Thresholds

Let's systematically test several thresholds.

**Copy and run this code**:

```python
# Test different thresholds
thresholds_to_test = [0.3, 0.5, 0.7, 0.9]
results = []

print(f"{'Threshold':<12} {'Accuracy':<10} {'Precision':<12} {'Recall':<10} {'F1':<10}")
print("-" * 60)

for threshold in thresholds_to_test:
    y_pred_thresh = (y_pred_proba_class1 >= threshold).astype(int)
    
    acc = accuracy_score(y_test, y_pred_thresh)
    prec = precision_score(y_test, y_pred_thresh)
    rec = recall_score(y_test, y_pred_thresh)
    f1 = f1_score(y_test, y_pred_thresh)
    
    results.append([threshold, acc, prec, rec, f1])
    print(f"{threshold:<12.1f} {acc:<10.3f} {prec:<12.3f} {rec:<10.3f} {f1:<10.3f}")

# Visualize the trade-off
results = np.array(results)
plt.figure(figsize=(10, 5))
plt.plot(results[:, 0], results[:, 2], 'o-', label='Precision', linewidth=2)
plt.plot(results[:, 0], results[:, 3], 's-', label='Recall', linewidth=2)
plt.plot(results[:, 0], results[:, 4], '^-', label='F1-Score', linewidth=2)
plt.xlabel('Decision Threshold')
plt.ylabel('Score')
plt.title('Precision-Recall Trade-off')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

**Questions**:
1. As threshold increases, what happens to precision? _________
2. As threshold increases, what happens to recall? _________
3. Which threshold gives the best F1-score? _________

**Key Insight**: There's a **trade-off** between precision and recall. We need to choose based on our priorities!

---

### Task 3.3: Plot the ROC Curve

The **ROC curve** (Receiver Operating Characteristic) shows the trade-off between True Positive Rate (recall) and False Positive Rate across ALL possible thresholds.

**YOUR TASK**: Complete the code below.

```python
# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(____, ____)  # Fill in: true labels and predicted probabilities

# Calculate AUC (Area Under Curve)
auc = roc_auc_score(____, ____)  # Fill in: true labels and predicted probabilities

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('ROC Curve')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"AUC Score: {auc:.3f}")
```

**Hint**: Use `y_test` and `y_pred_proba_class1` (the probabilities, not the class labels).

**AI Help**: Ask ChatGPT: "What does AUC score mean in ROC curve analysis?"

---

### Task 3.4: Interpret the ROC Curve

**YOUR TASK**: Answer these questions.

1. **What does AUC = 1.0 mean?**
   - [ ] Perfect classifier (separates classes completely)
   - [ ] Random classifier (no better than guessing)
   - [ ] Poor classifier (worse than random)

2. **What does AUC = 0.5 mean?**
   - [ ] Perfect classifier
   - [ ] Random classifier
   - [ ] Poor classifier

3. **Is your model's AUC good?** (Hint: AUC > 0.9 is excellent)
   
   _________

4. **Why is ROC curve useful?**
   
   _________________________________________________________________________

**Key Insight**: ROC curve and AUC give us a single score that summarizes model performance across ALL possible thresholds. Higher AUC = better model!

---

## Summary: What You Learned Today ✓

Congratulations! You now know how to:

- ✅ **Train a logistic regression classifier** using sklearn
- ✅ **Get both class predictions and probability predictions**
- ✅ **Interpret a confusion matrix** (TP, TN, FP, FN)
- ✅ **Calculate and interpret metrics**: accuracy, precision, recall, F1-score
- ✅ **Adjust decision thresholds** to control precision-recall trade-off
- ✅ **Plot and interpret ROC curves and AUC scores**

**Key Takeaways**:
1. **Accuracy alone can be misleading** - always check precision and recall!
2. **Threshold choice depends on the problem** - for cancer detection, high recall is critical
3. **ROC curve and AUC** provide a comprehensive measure of classifier performance
4. **Probabilities give more information** than just class labels

---

## Reflection Questions

Before you leave, take a moment to reflect:

1. **Why is recall more important than precision for cancer detection?**
   
   _________________________________________________________________________

2. **If you were building a spam filter, would you prioritize precision or recall? Why?**
   
   _________________________________________________________________________

3. **What's the main advantage of using predict_proba() instead of just predict()?**
   
   _________________________________________________________________________

4. **When would you use a threshold other than 0.5?**
   
   _________________________________________________________________________

---

## Part 4: For MPS439 Students Only - Implementing Logistic Regression from Scratch (~30 minutes)

### Background
Now that you understand how to USE logistic regression, let's implement it from scratch to understand what's happening under the hood. You'll implement:
1. The sigmoid function
2. Cross-entropy loss
3. Gradient descent for optimization

This will deepen your understanding and prepare you for neural networks later in the course!

---

### Task 4.1: Implement the Sigmoid Function

The sigmoid function maps any real number to (0, 1):

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

**YOUR TASK**: Complete the implementation.

```python
def sigmoid(z):
    """
    Compute the sigmoid function.
    
    Parameters:
    z : array-like, shape (n_samples,)
        Input values
    
    Returns:
    array-like, shape (n_samples,)
        Sigmoid of input
    """
    return 1 / (1 + np.exp(____))  # Fill in: what goes in the exponent?

# Test the function
z_test = np.array([-5, -2, 0, 2, 5])
sigmoid_values = sigmoid(z_test)

print("Testing sigmoid function:")
print(f"z values:       {z_test}")
print(f"sigmoid(z):     {sigmoid_values}")
print(f"Expected near:  [0.007, 0.119, 0.500, 0.881, 0.993]")

# Visualize sigmoid
z_range = np.linspace(-10, 10, 100)
plt.figure(figsize=(8, 5))
plt.plot(z_range, sigmoid(z_range), linewidth=2)
plt.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='y=0.5')
plt.axvline(0, color='red', linestyle='--', alpha=0.5, label='z=0')
plt.xlabel('z')
plt.ylabel('σ(z)')
plt.title('Sigmoid Function')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()
```

**Hint**: Remember that $e^{-z}$ in Python is `np.exp(-z)`.

**Question**: What is sigmoid(0)? Does your implementation give the correct answer? _________

---

### Task 4.2: Implement Cross-Entropy Loss

Cross-entropy loss measures how well our predictions match the true labels:

$$L = -\frac{1}{n}\sum_{i=1}^{n} [y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i)]$$

**YOUR TASK**: Complete the implementation.

```python
def cross_entropy_loss(y_true, y_pred):
    """
    Compute cross-entropy loss.
    
    Parameters:
    y_true : array-like, shape (n_samples,)
        True labels (0 or 1)
    y_pred : array-like, shape (n_samples,)
        Predicted probabilities (between 0 and 1)
    
    Returns:
    float
        Cross-entropy loss value
    """
    # Clip predictions to avoid log(0)
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    # Calculate cross-entropy
    n = len(y_true)
    loss = -np.mean(y_true * np.____(y_pred) + (1 - y_true) * np.log(____))
    # Fill in: what function for log? what goes in the second log?
    
    return loss

# Test the function
y_true_test = np.array([1, 0, 1, 1, 0])
y_pred_test = np.array([0.9, 0.1, 0.8, 0.6, 0.3])

loss = cross_entropy_loss(y_true_test, y_pred_test)
print(f"Test loss: {loss:.4f}")
print("Expected: ~0.20 (good predictions should have low loss)")

# Compare with different predictions
y_pred_perfect = np.array([1.0, 0.0, 1.0, 1.0, 0.0])
y_pred_terrible = np.array([0.0, 1.0, 0.0, 0.0, 1.0])

loss_perfect = cross_entropy_loss(y_true_test, y_pred_perfect)
loss_terrible = cross_entropy_loss(y_true_test, y_pred_terrible)

print(f"\nPerfect predictions loss: {loss_perfect:.4f}")
print(f"Terrible predictions loss: {loss_terrible:.4f}")
```

**Hint**: Use `np.log()` for natural logarithm. For the second log, we need log(1 - y_pred).

**Question**: Which predictions have lower loss - perfect or terrible? _________

---

### Task 4.3: Implement Gradient Descent

Now let's implement the full training algorithm!

**Background**: The gradient of cross-entropy loss with respect to weights is:

$$\frac{\partial L}{\partial w} = \frac{1}{n} X^T (\hat{y} - y)$$

Where $\hat{y} = \sigma(Xw)$

**Copy and complete this code**:

```python
def logistic_regression_gradient_descent(X, y, learning_rate=0.01, n_iterations=1000):
    """
    Train logistic regression using gradient descent.
    
    Parameters:
    X : array-like, shape (n_samples, n_features)
        Training data
    y : array-like, shape (n_samples,)
        Target labels (0 or 1)
    learning_rate : float
        Step size for gradient descent
    n_iterations : int
        Number of iterations to train
    
    Returns:
    weights : array-like, shape (n_features,)
        Learned weights
    losses : list
        Loss at each iteration
    """
    n_samples, n_features = X.shape
    
    # Initialize weights to zero
    weights = np.zeros(n_features)
    losses = []
    
    for iteration in range(n_iterations):
        # Forward pass: compute predictions
        z = X @ weights  # Linear combination
        y_pred = sigmoid(____)  # Fill in: what do we apply sigmoid to?
        
        # Compute loss
        loss = cross_entropy_loss(y, y_pred)
        losses.append(loss)
        
        # Backward pass: compute gradient
        gradient = (1 / n_samples) * (X.T @ (y_pred - ____))  # Fill in: y_pred minus what?
        
        # Update weights
        weights = weights - ____ * gradient  # Fill in: what multiplies the gradient?
        
        # Print progress every 100 iterations
        if (iteration + 1) % 100 == 0:
            print(f"Iteration {iteration + 1}/{n_iterations}, Loss: {loss:.4f}")
    
    return weights, losses

# Prepare data (use only 2 features for simplicity)
X_simple = X_train[:, :2]  # Use first 2 features
X_test_simple = X_test[:, :2]

# Train our implementation
print("Training logistic regression from scratch...")
my_weights, my_losses = logistic_regression_gradient_descent(
    X_simple, y_train, learning_rate=0.01, n_iterations=1000
)

print(f"\nFinal weights: {my_weights}")
print(f"Final loss: {my_losses[-1]:.4f}")
```

**Hint**: 
- First blank: apply sigmoid to `z`
- Second blank: subtract `y` 
- Third blank: use `learning_rate`

---

### Task 4.4: Visualize Training Progress

Let's see how the loss decreased during training.

**Copy and run**:

```python
# Plot loss curve
plt.figure(figsize=(10, 5))
plt.plot(my_losses, linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('Cross-Entropy Loss')
plt.title('Training Loss Over Time')
plt.grid(True, alpha=0.3)
plt.show()

print(f"Initial loss: {my_losses[0]:.4f}")
print(f"Final loss: {my_losses[-1]:.4f}")
print(f"Loss reduction: {my_losses[0] - my_losses[-1]:.4f}")
```

**Questions**:
1. Does the loss decrease over time? _________
2. Does the loss curve flatten out (converge)? _________
3. Around which iteration did the model mostly converge? _________

---

### Task 4.5: Compare with sklearn

Let's see how our implementation compares to sklearn's optimized version!

**YOUR TASK**: Complete the comparison.

```python
# Train sklearn's logistic regression on same data
sklearn_model = LogisticRegression(max_iter=1000)
sklearn_model.fit(____, ____)  # Fill in: training data

# Make predictions with our implementation
z_our = X_test_simple @ my_weights
y_pred_our = sigmoid(z_our)
y_pred_classes_our = (y_pred_our >= 0.5).astype(int)

# Make predictions with sklearn
y_pred_classes_sklearn = sklearn_model.predict(X_test_simple)

# Compare accuracy
accuracy_our = accuracy_score(y_test, y_pred_classes_our)
accuracy_sklearn = accuracy_score(y_test, y_pred_classes_sklearn)

print("Comparison:")
print(f"Our implementation accuracy:    {accuracy_our:.4f}")
print(f"sklearn implementation accuracy: {accuracy_sklearn:.4f}")
print(f"Difference:                      {abs(accuracy_our - accuracy_sklearn):.4f}")

# Compare weights
print("\nWeight comparison (first 2 features only):")
print(f"Our weights:    {my_weights}")
print(f"sklearn weights: {sklearn_model.coef_[0]}")
```

**Hint**: Use `X_simple` and `y_train` for sklearn's fit.

**Questions**:
1. How close are the accuracies? _________
2. Why might there be small differences? _________
3. Which one trained faster? _________

**Key Insights**:
- Your implementation works! 🎉
- sklearn is faster because it uses optimized C code and better algorithms
- Understanding the implementation helps you know what's happening inside the "black box"
- This is exactly how neural networks work - just with more layers!

---

### Task 4.6: Experiment with Learning Rate

Let's see how learning rate affects convergence.

**Copy and run**:

```python
# Test different learning rates
learning_rates = [0.001, 0.01, 0.1]

plt.figure(figsize=(12, 4))

for i, lr in enumerate(learning_rates):
    weights, losses = logistic_regression_gradient_descent(
        X_simple, y_train, learning_rate=lr, n_iterations=500
    )
    
    plt.subplot(1, 3, i+1)
    plt.plot(losses, linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title(f'Learning Rate = {lr}')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

**Questions**:
1. Which learning rate converges fastest? _________
2. Which learning rate is too slow? _________
3. What might happen with a learning rate that's too large (try 1.0 if you have time)? _________

---

## Summary: What MPS439 Students Learned ✓

**Beyond the core content, you now understand:**

- ✅ **The sigmoid function** - how it maps any number to (0, 1)
- ✅ **Cross-entropy loss** - why it's the right loss function for classification
- ✅ **Gradient descent algorithm** - how we optimize the weights iteratively
- ✅ **Implementation from scratch** - what sklearn is doing under the hood
- ✅ **Learning rate tuning** - how step size affects convergence

**This prepares you for:**
- **Deep learning (Weeks 10-11)** - neural networks are just stacked logistic regression!
- **Understanding optimization** - all modern ML uses variants of gradient descent
- **Debugging models** - knowing the internals helps you troubleshoot
- **Advanced implementations** - you can now modify and extend algorithms

**Key insights:**
1. Logistic regression is a **single-layer neural network**
2. Sigmoid + cross-entropy creates a **convex optimization problem**
3. Gradient descent **guarantees finding the global minimum** (for logistic regression)
4. sklearn uses **more sophisticated optimizers** (L-BFGS, Newton-CG) that converge faster

---

## Reflection for MPS439 Students

Based on what you implemented today:

1. **Why does the sigmoid function ensure outputs are always between 0 and 1?**
   
   _________________________________________________________________________

2. **How does the gradient point in the direction of steepest ascent, and why do we subtract it?**
   
   _________________________________________________________________________

3. **What's the connection between logistic regression and neural networks?**
   
   _________________________________________________________________________

4. **Why is cross-entropy loss better than squared error for classification?**
   
   _________________________________________________________________________

---

**Lab Complete!** 🎉

You now have both practical skills (using sklearn) and theoretical understanding (implementing from scratch) of logistic regression. This dual understanding will serve you well in more advanced ML topics!

**Next week**: Linear Discriminant Analysis (LDA) - another approach to classification that models class distributions directly.

<!-- COURSE_FEEDBACK_QR:START -->
---

## 30-second feedback

Scan the code to share anonymous feedback or post a question for this lab. It opens the correct **Lesson 04 lab** record automatically.

![Feedback QR code for Lesson 04 lab](./feedback-qr.png){fig-align="center" width="180px"}
<!-- COURSE_FEEDBACK_QR:END -->
