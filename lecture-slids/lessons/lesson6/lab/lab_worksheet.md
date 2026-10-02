---
pagetitle: "Lab 6: Decision Trees"
---

# Lab 6: Decision Trees
## MPS311/439 - Machine Learning
**Lesson 6 Lab Session | Duration: 50 minutes**

---

## Introduction

Welcome to Lab 6! Today you'll explore **Decision Trees** - a powerful and intuitive classification method that makes decisions through hierarchical yes/no questions.

**What you'll learn today:**
- How to build decision trees using sklearn
- How to visualize and interpret tree structure
- Which features are most important for predictions
- When trees overfit and how to prevent it

**Remember:** This lab uses a fill-in-the-blanks approach. Don't write code from scratch - just fill in the blanks marked with `____`. Focus on understanding the concepts!

---

## Setup: Import Libraries and Load Data

We'll use the **Iris dataset** again - it has 3 classes of flowers with 4 measurements per flower. Decision trees work particularly well on this dataset because the classes can be separated with simple rules.

**Copy and run this code:**

```python
# Import packages
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Load iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")
print(f"Features: {iris.feature_names}")
print(f"Classes: {iris.target_names}")
```

**What's happening here?**
- We have 4 features: sepal length, sepal width, petal length, petal width
- We have 3 classes: setosa, versicolor, virginica
- We split 70/30 for training and testing

---

## Part 1: Building Your First Decision Tree (8 minutes)

**Background:** Decision trees split data at each node by asking questions like "Is petal width ≤ 0.8?" The tree grows by recursively splitting until stopping criteria are met. Let's build our first tree!

### Task 1.1: Create and train a simple tree

**Fill in the blanks below:**

```python
# Create a decision tree with maximum depth of 3
clf = DecisionTreeClassifier(max_depth=____, random_state=42)

# Train the tree
clf.____(X_train, y_train)

# Make predictions
y_pred = clf.____(X_test)

# Calculate accuracy
train_acc = clf.score(____, ____)
test_acc = clf.score(____, ____)

print(f"Training accuracy: {train_acc:.3f}")
print(f"Test accuracy: {test_acc:.3f}")
```

**Hints:**
- Start with `max_depth=3` for a shallow tree
- Method to train: `.fit(X, y)`
- Method to predict: `.predict(X)`
- Training data: `X_train, y_train`
- Test data: `X_test, y_test`

**AI Help:** Ask ChatGPT: "How do I use DecisionTreeClassifier in sklearn? Show me fit, predict, and score methods."

### Task 1.2: Interpret the results

Look at your training and test accuracy. Are they similar or different?

```
Your observation: _______________________________________________
```

---

## Part 2: Visualizing the Tree Structure (8 minutes)

**Background:** One of the best features of decision trees is that we can **see** exactly how they make decisions! The `plot_tree()` function creates a visual representation of the tree structure.

### Task 2.1: Visualize your tree

**Fill in the blanks:**

```python
# Create a large figure
plt.figure(figsize=(20, 10))

# Plot the tree
plot_tree(____, 
          filled=____,
          rounded=True,
          feature_names=iris.feature_names,
          class_names=iris.target_names)

plt.title("Decision Tree for Iris Classification")
plt.show()
```

**Hints:**
- First argument is your trained model: `clf`
- `filled=True` colors the nodes by majority class

**AI Help:** Ask ChatGPT: "How do I visualize a decision tree with plot_tree in sklearn?"

### Task 2.2: Understand the tree nodes

Look at your tree visualization. Each box (node) shows several pieces of information:

- **Top line**: The decision rule (e.g., "petal width ≤ 0.8")
- **gini**: How mixed the classes are (0 = pure, 0.5 = maximum mixture for 2 classes)
- **samples**: Number of training samples at this node
- **value**: Array showing count of each class [setosa, versicolor, virginica]
- **class**: The majority class at this node

**Answer these questions by looking at your tree:**

1. What is the first split (at the root node)?
```
Feature: ________________  Threshold: ________________
```

2. Look at a pure leaf node (gini=0). How many samples and which class?
```
Samples: ______  Class: ________________
```

3. Why are the leaf nodes different colors?
```
Your answer: _______________________________________________
```

---

## Part 3: Understanding Feature Importance (8 minutes)

**Background:** Decision trees automatically tell us which features are most important! Feature importance measures how much each feature contributes to reducing impurity across all splits.

### Task 3.1: Extract and display feature importance

**Fill in the blanks:**

```python
# Get feature importances
importances = clf.____

# Print them
print("Feature Importances:")
for name, importance in zip(iris.feature_names, importances):
    print(f"  {name}: {importance:.3f}")
```

**Hint:** 
- Attribute name: `.feature_importances_`

**AI Help:** Ask ChatGPT: "How do I get feature importance from a DecisionTreeClassifier?"

### Task 3.2: Visualize feature importance

**Fill in the blanks:**

```python
# Create bar plot
plt.figure(figsize=(8, 5))
plt.bar(iris.feature_names, ____)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance in Decision Tree')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

**Hint:**
- Y-axis data: `importances`

### Task 3.3: Interpret the importance

Which feature is most important? Which features have zero importance (not used)?

```
Most important: _______________________________________________
Zero importance: ______________________________________________
```

**Think about:** Why might some features have zero importance? Does that mean they're useless for classification?

---

## Part 4: The Overfitting Problem (12 minutes)

**Background:** Decision trees can easily overfit by growing too deep and memorizing training data. Let's see this in action by comparing a shallow tree vs. a very deep tree.

### Task 4.1: Train a shallow tree (baseline)

We already have this from Part 1! Your shallow tree with `max_depth=3`.

```python
# Remind ourselves of the shallow tree performance
print(f"Shallow tree (depth=3):")
print(f"  Training accuracy: {train_acc:.3f}")
print(f"  Test accuracy: {test_acc:.3f}")
```

### Task 4.2: Train a very deep tree

**Fill in the blanks:**

```python
# Create a tree with no depth limit
clf_deep = DecisionTreeClassifier(max_depth=____, random_state=42)

# Train it
clf_deep.fit(____, ____)

# Evaluate
deep_train_acc = clf_deep.score(X_train, y_train)
deep_test_acc = clf_deep.score(____, ____)

print(f"\nDeep tree (no limit):")
print(f"  Training accuracy: {deep_train_acc:.3f}")
print(f"  Test accuracy: {deep_test_acc:.3f}")
```

**Hints:**
- Use `max_depth=None` for no limit
- Training data: `X_train, y_train`
- Test data: `X_test, y_test`

**AI Help:** Ask ChatGPT: "What does max_depth=None mean in DecisionTreeClassifier?"

### Task 4.3: Compare and diagnose overfitting

Let's visualize the comparison:

```python
# Compare the two trees
models = ['Shallow (depth=3)', 'Deep (no limit)']
train_scores = [train_acc, deep_train_acc]
test_scores = [test_acc, deep_test_acc]

x = np.arange(len(models))
width = 0.35

plt.figure(figsize=(8, 5))
plt.bar(x - width/2, train_scores, width, label='Train')
plt.bar(x + width/2, test_scores, width, label='Test')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Shallow vs Deep Tree Performance')
plt.xticks(x, models)
plt.legend()
plt.ylim([0.8, 1.0])
plt.show()
```

**Answer these questions:**

1. Which tree has higher training accuracy?
```
Answer: _______________________________________________
```

2. Which tree has higher test accuracy?
```
Answer: _______________________________________________
```

3. Look at the gap between training and test accuracy for the deep tree. What does this tell you?
```
Answer: _______________________________________________
```

4. Which tree would you choose to use in practice? Why?
```
Answer: _______________________________________________
___________________________________________________________
```

**Key insight:** Perfect training accuracy (100%) is often a **bad sign** - it means the model memorized the training data rather than learning general patterns!

---

## Part 5: Preventing Overfitting with Hyperparameters (8 minutes)

**Background:** We can control tree complexity using hyperparameters. The main ones are:
- `max_depth`: Limits how deep the tree can grow
- `min_samples_split`: Minimum samples needed to split a node
- `min_samples_leaf`: Minimum samples required in each leaf

Let's experiment!

### Task 5.1: Experiment with min_samples_split

**Fill in the blanks:**

```python
# Try different min_samples_split values
for min_split in [2, 10, 20]:
    clf_temp = DecisionTreeClassifier(
        max_depth=None,
        min_samples_split=____,
        random_state=42
    )
    clf_temp.fit(X_train, y_train)
    
    train = clf_temp.score(X_train, y_train)
    test = clf_temp.score(X_test, y_test)
    
    print(f"min_samples_split={min_split:2d}: Train={train:.3f}, Test={test:.3f}")
```

**Hint:**
- Use the loop variable: `min_split`

**AI Help:** Ask ChatGPT: "What does min_samples_split control in decision trees?"

### Task 5.2: Experiment with min_samples_leaf

**Fill in the blanks:**

```python
# Try different min_samples_leaf values
for min_leaf in [1, 5, 10]:
    clf_temp = DecisionTreeClassifier(
        max_depth=None,
        min_samples_leaf=____,
        random_state=42
    )
    clf_temp.____(X_train, y_train)
    
    train = clf_temp.score(____, ____)
    test = clf_temp.score(X_test, y_test)
    
    print(f"min_samples_leaf={min_leaf:2d}: Train={train:.3f}, Test={test:.3f}")
```

**Hints:**
- Use the loop variable: `min_leaf`
- Training method: `.fit(X_train, y_train)`
- Score on training data: `X_train, y_train`

### Task 5.3: Interpret the results

1. What happens to training accuracy as you increase `min_samples_split` or `min_samples_leaf`?
```
Answer: _______________________________________________
```

2. What happens to test accuracy?
```
Answer: _______________________________________________
```

3. Which hyperparameter setting gives the best test accuracy?
```
Answer: _______________________________________________
```

**Key insight:** Larger values of `min_samples_split` and `min_samples_leaf` create simpler trees that generalize better!

---

## Reflection Questions (3 minutes)

Answer these questions based on what you've learned today:

**Question 1:** What does a decision tree's `.fit()` method actually do? Describe the process in your own words.

```
Your answer: _______________________________________________
___________________________________________________________
```

**Question 2:** You have a tree with training accuracy of 100% and test accuracy of 75%. Is this a good model? Why or why not?

```
Your answer: _______________________________________________
___________________________________________________________
```

**Question 3:** A feature has importance of 0.0. Does this mean it's useless for classification? Explain.

```
Your answer: _______________________________________________
___________________________________________________________
```

**Question 4:** When would you choose a decision tree over logistic regression?

```
Your answer: _______________________________________________
___________________________________________________________
```

---

## Summary

**What you learned today:**
- ✅ How to build decision trees with `DecisionTreeClassifier`
- ✅ How to visualize trees with `plot_tree()` and interpret nodes
- ✅ How to extract and interpret feature importance
- ✅ How to recognize overfitting (perfect training accuracy, poor test accuracy)
- ✅ How to prevent overfitting using `max_depth`, `min_samples_split`, `min_samples_leaf`

**Key takeaway:** Decision trees are powerful and interpretable, but they easily overfit! Always control complexity with hyperparameters and monitor both training and test performance.

---

## Part 6: Advanced Challenge - For MPS439 Students Only (30 minutes)

**Background:** Now you'll calculate Gini impurity by hand and explore state-of-the-art ensemble methods like XGBoost!

### Task 6.1: Calculate Gini Impurity by Hand

Given a node with the following class distribution:
- Class A: 40 samples
- Class B: 30 samples  
- Class C: 30 samples

**Calculate the Gini impurity step by step:**

Recall the formula: Gini = 1 - Σ(p_i²)

```python
# Step 1: Calculate total samples
total = ____

# Step 2: Calculate proportions
p_A = ____ / total
p_B = 30 / ____
p_C = ____ / total

print(f"Proportions: p_A={p_A:.3f}, p_B={p_B:.3f}, p_C={p_C:.3f}")

# Step 3: Calculate Gini impurity
gini = 1 - (____**2 + p_B**2 + ____**2)

print(f"Gini impurity: {gini:.3f}")
```

**Verify your calculation:**
- Total = 100
- p_A = 0.4, p_B = 0.3, p_C = 0.3
- Gini = 1 - (0.16 + 0.09 + 0.09) = 0.66

**Now calculate for a split:**

A split divides the above node into two children:
- **Left child**: 50 samples [35 A, 10 B, 5 C]
- **Right child**: 50 samples [5 A, 20 B, 25 C]

**Fill in the blanks:**

```python
# Left child Gini
p_A_left = 35 / ____
p_B_left = ____ / 50
p_C_left = 5 / ____

gini_left = 1 - (p_A_left**2 + ____**2 + p_C_left**2)

# Right child Gini
p_A_right = ____ / 50
p_B_right = 20 / ____
p_C_right = ____ / 50

gini_right = 1 - (____**2 + p_B_right**2 + p_C_right**2)

# Weighted Gini for the split
weighted_gini = (50/100) * ____ + (50/100) * gini_right

# Gini reduction
gini_reduction = gini - ____

print(f"Left Gini: {gini_left:.3f}")
print(f"Right Gini: {gini_right:.3f}")
print(f"Weighted Gini: {weighted_gini:.3f}")
print(f"Gini reduction: {gini_reduction:.3f}")
```

**Question:** Is this a good split? Why or why not?
```
Your answer: _______________________________________________
```

### Task 6.2: Explore XGBoost

**Background:** XGBoost is a state-of-the-art gradient boosting library that often wins machine learning competitions. Let's compare it to a single decision tree!

First, install XGBoost (if not already installed):
```python
# Run this in a separate cell if needed
# !pip install xgboost
```

**Fill in the blanks to use XGBoost:**

```python
import xgboost as xgb

# Create XGBoost classifier
xgb_clf = xgb.XGBClassifier(
    n_estimators=____,  # Try 100 trees
    max_depth=3,
    learning_rate=0.1,
    random_state=42
)

# Train
xgb_clf.____(X_train, y_train)

# Evaluate
xgb_train = xgb_clf.____(X_train, y_train)
xgb_test = xgb_clf.score(____, ____)

print(f"XGBoost Training accuracy: {xgb_train:.3f}")
print(f"XGBoost Test accuracy: {xgb_test:.3f}")
```

**Hints:**
- Use 100 estimators
- Methods are same as sklearn: `.fit()`, `.score()`
- Test data: `X_test, y_test`

**AI Help:** Ask ChatGPT: "How do I use XGBoost for classification in Python?"

### Task 6.3: Compare all methods

**Fill in the blanks to create a comparison:**

```python
# Compare all methods
methods = ['Single Tree\n(depth=3)', 'Single Tree\n(no limit)', 'XGBoost\n(100 trees)']
train_scores = [train_acc, deep_train_acc, ____]
test_scores = [test_acc, deep_test_acc, ____]

x = np.arange(len(methods))
width = 0.35

plt.figure(figsize=(10, 6))
plt.bar(x - width/2, ____, width, label='Train')
plt.bar(x + width/2, test_scores, width, label='Test')
plt.xlabel('Method')
plt.ylabel('Accuracy')
plt.title('Comparison of Tree-based Methods')
plt.xticks(x, methods)
plt.legend()
plt.ylim([0.85, 1.0])
plt.tight_layout()
plt.show()
```

**Hints:**
- XGBoost scores: `xgb_train`, `xgb_test`
- Training scores list: `train_scores`

### Task 6.4: Advanced reflection

**Question 1:** How does XGBoost achieve better performance than a single tree?
```
Your answer: _______________________________________________
___________________________________________________________
```

**Question 2:** XGBoost has high training accuracy but doesn't overfit as much as the deep tree. Why?
```
Your answer: _______________________________________________
___________________________________________________________
```

**Question 3:** When would you use a single interpretable tree instead of XGBoost in a real-world application?
```
Your answer: _______________________________________________
___________________________________________________________
```

### Task 6.5: Explore another classifier (Optional)

Try using Support Vector Machine (SVM) from sklearn and compare it with the tree methods:

```python
from sklearn.svm import SVC

# Your code here - create, train, and evaluate an SVM
# Compare its performance with trees and XGBoost
```

**Challenge:** Can you beat XGBoost's performance on this dataset with SVM?

---

**Congratulations!** You've completed the advanced section. You now understand decision trees deeply - both how to use them and how they work mathematically!

---

**End of Lab Worksheet**

<!-- COURSE_FEEDBACK_QR:START -->
---

## 30-second feedback

What would help you learn better next time?

Scan to share anonymous feedback on today's lab.

![Feedback QR code for Lesson 06 lab](./feedback-qr.png){fig-align="center" width="180px"}
<!-- COURSE_FEEDBACK_QR:END -->
