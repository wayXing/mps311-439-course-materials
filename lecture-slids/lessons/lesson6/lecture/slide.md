---
theme: default
background: https://cover.sli.dev
class: text-center
highlighter: shiki
lineNumbers: false
info: |
  ## Lesson 6: Decision Trees
  MPS311/439 Machine Learning - Dr. Wei Xing
drawings:
  persist: false
transition: slide-left
title: 'Lesson 6: Decision Trees'
routerMode: hash
mdc: true
---

# Lesson 6: Decision Trees

## From Linear Boundaries to Hierarchical Decisions

<div class="pt-12">
  <span class="text-xl">
    MPS311/439 Machine Learning<br>
    Dr. Wei Xing<br>
    2026–27
  </span>
</div>

---

# Where We Left Off

## Lesson 5: Linear Discriminant Analysis

<div class="mt-8">

**What we learned:**

- LDA/QDA assume **Gaussian distributions**
- LDA: Equal covariances → **linear boundaries**
- QDA: Different covariances → **quadratic boundaries**
- Both work well when assumptions hold

</div>

<div class="mt-10 p-6 bg-yellow-50 rounded-lg border-l-4 border-yellow-400">

**Today's Challenge**: What if our data doesn't fit these neat distributional assumptions?

What if the patterns are **fundamentally non-linear**?

</div>

---

# The Challenge: When Linear Methods Fail

<div class="flex justify-center mt-4">
  <img src="./figures/fig_xor_linear_fail.png" class="h-76" />
</div>

<div class="mt-2 p-1 bg-red-50 rounded-lg border-2 border-red-400">

**The XOR Problem**: Four clusters in a diagonal pattern

No straight line (or even quadratic curve) can separate blue from red!

</div>

<div class="mt-1 text-center text-xl font-bold text-blue-600">

We need a fundamentally different approach! 🤔

</div>

---

# How Humans Actually Make Decisions

<div class="mt-6">

## Humans don't calculate weighted sums...

<div class="mt-4 text-lg">

We ask **sequences of simple yes/no questions**!

</div>

</div>

<div class="mt-8 p-6 bg-blue-50 rounded-lg">

**Example: Should I go running today?**

<div class="mt-4 grid grid-cols-3 gap-4">

<div class="p-4 bg-white rounded shadow">

**Question 1**

Is it raining?

→ Yes: Stay home ❌

→ No: Continue ✓

</div>

<div class="p-4 bg-white rounded shadow">

**Question 2**

Is temp > 5°C?

→ No: Too cold ❌

→ Yes: Continue ✓

</div>

<div class="p-4 bg-white rounded shadow">

**Question 3**

Is temp < 30°C?

→ No: Too hot ❌

→ Yes: Go running! ✅

</div>

</div>

</div>

<div class="mt-8 p-4 bg-green-50 border-l-4 border-green-400 rounded text-lg">

Can we teach machines to think this way? **YES!** That's what decision trees do.

</div>

---

# What is a Decision Tree?

<div class="mt-6">

<div class="p-6 bg-blue-50 rounded-lg border-2 border-blue-400">

**Definition**: A flowchart-like structure that makes predictions through hierarchical yes/no questions

</div>

</div>

<div class="mt-8 grid grid-cols-3 gap-6">

<div class="p-6 bg-purple-50 rounded-lg text-center">

**Internal Nodes**

Questions/tests about features

"Is petal width ≤ 0.8?"

</div>

<div class="p-6 bg-green-50 rounded-lg text-center">

**Branches**

Outcomes of tests

Yes (left) / No (right)

</div>

<div class="p-6 bg-orange-50 rounded-lg text-center">

**Leaf Nodes**

Final predictions

"Predict: Setosa"

</div>

</div>

<div class="mt-8 text-center text-xl text-gray-600">

Think of it as playing **"20 Questions"** to classify data 🎯

</div>

---

# Visual Example: Iris Classification

<div class="mt-0 p-0 bg-green-50 rounded-lg">

**Walk through an example**: Flower with petal width = 1.5 cm, petal length = 4.5 cm

<div class="mt-1 text-lg">

1. Root: "petal width ≤ 0.8?" → **No** (1.5 > 0.8), go right
2. "petal width ≤ 1.75?" → **Yes** (1.5 ≤ 1.75), go left
3. "petal length ≤ 4.95?" → **Yes** (4.5 ≤ 4.95), go left
4. **Prediction: Versicolor!** 🌸

</div>
</div>

<!-- <div class="flex justify-end mt-0">
  <img src="./figures/fig_simple_tree_example.png" class="h-70" />
</div> -->
<!-- 图片容器：使用绝对定位，z-index 为负值，并调整位置 -->
<div class="absolute z-[1]    <!-- 核心：绝对定位并置于后面 -->
            top-50 right-0   <!-- 将元素顶部和左侧边缘定位到父容器的中心 -->
            "> <!-- 让图片尽量覆盖整个页面，但保持宽高比 -->
  <img src="./figures/fig_simple_tree_example.png" class="w-full h-90 object-cover opacity-80" alt="Simple Decision Tree Example" />
</div>


---

# How to Read a Tree Node

<div class="mt-2">

## Each box (node) contains important information:


<div class="mt-2 grid grid-cols-2 gap-2">

<div class="space-y-2">

<div class="p-0 bg-blue-50 rounded">

**Decision rule**

e.g., "petal width ≤ 0.8"

The question being asked

</div>

<div class="p-0 bg-purple-50 rounded">

**gini**

Measure of impurity

How mixed the classes are

</div>

<div class="p-0 bg-green-50 rounded">

**samples**

Number of training samples

at this node

</div>

</div>

<div class="space-y-4">

<div class="p-0 bg-orange-50 rounded">

**value**

Array: [class_0, class_1, . ..]

Count for each class

</div>

<div class="p-0 bg-pink-50 rounded-lg">

**class**

Majority class

What this node predicts

</div>

<div class="p-0 bg-yellow-50 rounded">

**Color intensity**

Darker = more pure

More samples of majority class

</div>

</div>

</div>

</div>

<div class="mt-0 text-center text-lg text-gray-600">

You can trace any prediction by following the path from root to leaf! 🔍

</div>

---

# Building Trees: The Key Question

<div class="mt-6">

<div class="p-6 bg-red-50 rounded-lg border-2 border-red-400 text-center text-2xl">

At each node: **Which feature and threshold should we split on?**

</div>

</div>

<div class="mt-8 grid grid-cols-2 gap-8">

<div class="p-6 bg-blue-50 rounded-lg">

### Principle 1: Maximize Distance

**Between class means after split**

<div class="mt-4 text-center text-4xl">

← →

</div>

<div class="mt-4">

Class centers should be **far apart**

</div>

</div>

<div class="p-6 bg-purple-50 rounded-lg">

### Principle 2: Minimize Spread

**Within each class after split**

<div class="mt-4 text-center text-4xl">

● ● ●

</div>

<div class="mt-4">

Points in same class should be **tightly clustered**

</div>

</div>

</div>

<div class="mt-8 p-6 bg-green-50 border-l-4 border-green-400 rounded text-xl">

**Goal**: Find splits that create **PURE** children nodes!

</div>

---

# Measuring Purity: Gini Impurity

<div class="mt-4">

<div class="p-6 bg-blue-50 rounded-lg">

**Gini Impurity** measures how "mixed" a node is

<div class="mt-4 text-center text-2xl p-4 bg-white rounded">

For binary classification: Gini = 1 - p₁² - p₂²

</div>

<div class="mt-4">

where p₁ = proportion of class 1, p₂ = proportion of class 2

</div>

</div>

</div>

<div class="mt-6 grid grid-cols-3 gap-4">

<div class="p-4 bg-green-100 rounded text-center">

**Gini = 0**

Perfect purity ✓

All one class

</div>

<div class="p-4 bg-yellow-100 rounded text-center">

**Gini = 0.48**

Moderate impurity

60-40 split

</div>

<div class="p-4 bg-red-100 rounded text-center">

**Gini = 0.5**

Maximum impurity ✗

50-50 split

</div>

</div>

<div class="mt-6 p-4 bg-purple-50 rounded">

**Example**: Node with 60 Class A, 40 Class B

- p₁ = 0.6, p₂ = 0.4
- Gini = 1 - (0.6² + 0.4²) = 1 - (0.36 + 0.16) = **0.48**

</div>

---

# The Greedy Algorithm

<div class="mt-1">

## How trees are built: Step by step

<div class="mt-2 grid grid-cols-4 gap-2">

<div class="p-6 bg-blue-50 rounded-lg">

### Step 1: Start

Begin with all training data at root node

</div>

<div class="p-6 bg-purple-50 rounded-lg">

### Step 2: Find Best Split

For current node:

- Try all features
- Try all thresholds
- Calculate weighted Gini for each
- Choose split that **minimizes** Gini

</div>

<div class="p-6 bg-green-50 rounded-lg">

### Step 3: Recurse

Apply same process to each child node

Repeat steps 2-3

</div>

<div class="p-6 bg-orange-50 rounded-lg">

### Step 4: Stop

When stopping criterion met:

- Perfect purity (Gini = 0)
- Max depth reached
- Node too small to split

</div>

</div>

</div>

<div class="mt-6 p-4 bg-yellow-50 border-l-4 border-yellow-400 rounded text-lg">

**"GREEDY"** = Make locally optimal choice at each step (not globally optimal)

</div>

---

# Implementation in Python

<div class="mt-4">

## Decision Trees in sklearn - Incredibly Simple!

<div class="mt-6">

```python {all|1-2|4-6|8-10|12-14|all}
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# 1. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# 2. Create and train tree
clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(X_train, y_train)

# 3. Evaluate
accuracy = clf.score(X_test, y_test)
print(f"Accuracy: {accuracy:.3f}")
```

</div>

</div>

<div class="mt-6 p-6 bg-green-50 border-l-4 border-green-400 rounded-lg text-xl">

**That's it!** Just 3 steps: Split data → Train tree → Evaluate

</div>

<div class="mt-4 text-center text-gray-600">

Next: Visualizations and feature importance! 📊

</div>

---

# Feature Importance

<div class="mt-1">

## Which features matter most for prediction?

<div class="mt-2 p-2 bg-blue-50 rounded-lg">

**Feature importance** = How much each feature reduces impurity across all splits

- Values sum to 1.0
- Higher value = more important
- Features not used in any split have importance = 0.0

</div>

</div>

<div class="mt-2 grid grid-cols-2 gap-2">

<div class="p-2 bg-blue-50 rounded-lg">

**Example output** (Iris dataset):
```python
importances = clf.feature_importances_
```

<div class="mt-2 p-2 bg-gray-50 rounded">

- petal_width: **0.65** ⭐
- petal_length: **0.35** ⭐
- sepal_width: **0.00**
- sepal_length: **0.00**

</div>

</div>

<div class="p-2 bg-purple-50 rounded">

**Why useful?**

✓ Understand your data

✓ Guide feature engineering

✓ Simplify model (remove unimportant features)

✓ Explain predictions to stakeholders

</div>

</div>

---

# The Overfitting Problem

<div class="mt-2">

<div class="p-6 bg-red-50 rounded-lg border-2 border-red-400 text-center text-xl">

⚠️ Decision trees are **highly prone to overfitting** ⚠️

</div>

</div>

<div class="mt-6 grid grid-cols-2 gap-8">

<div class="p-6 bg-green-50 rounded-lg border-2 border-green-400">

### Shallow Tree (depth=3)

**Training**: 85%

**Test**: 84%

<div class="mt-4 text-lg">

✓ Simple boundary

✓ Small gap (good!)

✓ Generalizes well

</div>

</div>

<div class="p-6 bg-red-50 rounded-lg border-2 border-red-400">

### Deep Tree (depth=15)

**Training**: 100% 😱

**Test**: 78%

<div class="mt-4 text-lg">

✗ Complex, jagged boundary

✗ **Large gap** (bad!)

✗ Memorized training data

</div>

</div>

</div>

<div class="mt-6 p-6 bg-yellow-50 border-l-4 border-yellow-400 rounded text-xl">

Higher training accuracy ≠ Better model! Watch the train-test **gap**

</div>

---

# Overfitting Visualization

<div class="flex justify-center mt-4">
  <img src="./figures/fig_overfitting_comparison.png" class="h-76" />
</div>

<div class="mt-2 grid grid-cols-3 gap-4">

<div class="p-0 bg-blue-50 rounded text-center border-2 border-blue-400">

**Too Shallow**

Underfitting

Both accuracies low

</div>

<div class="p-0 bg-green-50 rounded text-center border-2 border-green-400">

**Just Right** ✓

Sweet spot!

Best test accuracy

</div>

<div class="p-0 bg-red-50 rounded text-center border-2 border-red-400">

**Too Deep**

Overfitting

Large train-test gap

</div>

</div>

<div class="mt-0 text-center text-lg text-gray-600">

Training accuracy (blue) → 100% | Test accuracy (red) → peaks then declines

</div>

---

# Controlling Overfitting: Hyperparameters

<div class="mt-4">

## Three key hyperparameters to control tree complexity:

<div class="mt-6 grid grid-cols-3 gap-6">

<div class="p-6 bg-blue-50 rounded-lg">

### max_depth

**Limits maximum tree depth**

<div class="mt-4">

- Smaller → simpler tree
- Typical: 3-10
- Start with 5

</div>

<div class="mt-4 p-2 bg-white rounded">

`max_depth=5`

</div>

</div>

<div class="p-6 bg-purple-50 rounded-lg">

### min_samples_split

**Minimum samples to split a node**

<div class="mt-4">

- Larger → fewer splits
- Typical: 2-50
- Try 10-20 for noisy data

</div>

<div class="mt-4 p-2 bg-white rounded">

`min_samples_split=20`

</div>

</div>

<div class="p-6 bg-green-50 rounded-lg">

### min_samples_leaf

**Minimum samples in each leaf**

<div class="mt-4">

- Larger → smoother boundaries
- Typical: 1-20
- Try 5-10

</div>

<div class="mt-4 p-2 bg-white rounded">

`min_samples_leaf=5`

</div>

</div>

</div>

</div>

---

# Practical Guidelines

<div class="mt-6">

## Recommended starting configuration:

<div class="mt-1">
```python
clf = DecisionTreeClassifier(
    max_depth=5,           # Moderate depth
    min_samples_split=10,  # Don't split tiny nodes
    min_samples_leaf=5,    # No tiny leaves
    random_state=42
)
```

</div>

</div>

<div class="mt-2 grid grid-cols-2 gap-1">

<div class="p-1 bg-red-50 rounded-lg">

### If Underfitting

(Both train & test accuracy low)

<div class="mt-4">

→ **Increase** `max_depth`

→ **Decrease** `min_samples_split`

→ **Decrease** `min_samples_leaf`

</div>

</div>

<div class="p-1 bg-blue-50 rounded-lg">

### If Overfitting

(High train, low test)

<div class="mt-4">

→ **Decrease** `max_depth`

→ **Increase** `min_samples_split`

→ **Increase** `min_samples_leaf`

</div>

</div>

</div>

<div class="mt-2 p-1 bg-yellow-50 border-l-4 border-yellow-400 rounded text-lg">

**Golden Rules**: Use cross-validation | Monitor train AND test | Start simple, add complexity only if needed

</div>

---

# When to Use Decision Trees

<div class="mt-4">

<div class="grid grid-cols-2 gap-6">

<div class="p-6 bg-green-50 rounded-lg">

### Advantages ✓

<div class="mt-4 space-y-3">

✓ **Highly interpretable** - explain any prediction

✓ **Handles non-linearity** automatically

✓ **No feature scaling needed**

✓ **Mixed data types** (numerical & categorical)

✓ **Fast** training and prediction

✓ **Automatic feature selection**

</div>

</div>

<div class="p-6 bg-red-50 rounded-lg">

### Disadvantages ✗

<div class="mt-4 space-y-3">

✗ **Prone to overfitting** (needs tuning)

✗ **High variance** (unstable)

✗ **Axis-aligned only** (can't do diagonals efficiently)

✗ **Biased** with imbalanced classes

✗ **Not globally optimal** (greedy)

</div>

</div>

</div>

</div>

<div class="mt-8 p-6 bg-blue-50 border-l-4 border-blue-400 rounded text-xl">

**When to use**: Need interpretability AND data is clearly non-linear

</div>

<div class="mt-4 text-center text-lg text-gray-600">

Rule of thumb: Try linear methods first (baseline), then try trees if needed

</div>

---

# Looking Ahead

<div class="mt-6">

<div class="p-8 bg-gradient-to-r from-purple-50 to-blue-50 rounded-lg border-2 border-blue-400">

## Next Week: Principal Component Analysis (PCA)

<div class="mt-6 text-xl">

**Big shift**: Supervised → **Unsupervised Learning**

</div>

<div class="mt-6 grid grid-cols-3 gap-6">

<div class="p-4 bg-white rounded shadow">

❌ No labels!

Find structure in X alone

</div>

<div class="p-4 bg-white rounded shadow">

📉 Dimensionality reduction

100D → 2D or 3D

</div>

<div class="p-4 bg-white rounded shadow">

🎯 What's "important"?

Find principal directions

</div>

</div>

</div>

</div>

<div class="mt-8 p-6 bg-orange-50 rounded-lg">

**For MPS439 students**: Check lecture notes Section 9 for **Random Forests** and **XGBoost** (ensemble methods that combine many trees → much better accuracy!)

</div>

---

# Key Takeaways

<div class="mt-6">

<div class="grid grid-cols-2 gap-6">

<div class="space-y-4">

<div class="p-4 bg-blue-50 rounded-lg">

**1. Hierarchical Decisions**

Trees ask yes/no questions sequentially

</div>

<div class="p-4 bg-purple-50 rounded-lg">

**2. Gini Impurity**

Measures how mixed a node is (want low values)

</div>

<div class="p-4 bg-green-50 rounded-lg">

**3. Greedy Algorithm**

Builds tree by choosing locally optimal splits

</div>

</div>

<div class="space-y-4">

<div class="p-4 bg-orange-50 rounded-lg">

**4. Overfitting**

Deep trees memorize → control with hyperparameters

</div>

<div class="p-4 bg-pink-50 rounded-lg">

**5. Easy Implementation**

Just 3 lines in sklearn!

</div>

<div class="p-4 bg-yellow-50 rounded-lg">

**6. Interpretability**

Can explain any prediction step-by-step

</div>

</div>

</div>

</div>

<div class="mt-8 p-6 bg-green-50 border-l-4 border-green-400 rounded text-xl text-center">

**Start simple** (depth=3-5), **tune** with cross-validation, **visualize** your trees!

</div>

---
layout: center
class: text-center
---

# Thank You!

<div class="mt-2">

## Questions?

<div class="mt-2 text-lg">

**Office Hours**: Check course website

**Email**: w.xing@sheffield.ac.uk

</div>

</div>

<div class="mt-2 p-1 bg-blue-50 rounded-lg">

**Next Session**: Lab on Friday - Hands-on with decision trees on real datasets!

**Lesson 7 Lecture**: PCA and unsupervised learning

</div>

<div class="mt-2 p-1 bg-green-50 rounded-lg">

**Encouragement**: Experiment with trees on different datasets! Try adjusting max_depth and observe the effects on train/test accuracy.

Understanding overfitting is key to becoming a good ML practitioner. 🚀

</div>

---
layout: center
class: text-center
---

<!-- COURSE_FEEDBACK_QR:START -->
---
layout: center
class: text-center
---

# 30-second feedback

<p class="text-3xl mb-4">What would help you learn better next time?</p>

<p class="text-2xl mb-4">Scan to share anonymous feedback on today's lecture.</p>

<img src="./feedback-qr.svg" alt="Feedback QR code for Lesson 06 lecture" class="w-44 mx-auto rounded-lg shadow" />

<p class="text-sm mt-4 opacity-70">MPS311/439 · 2026-27 · Lesson 06 lecture</p>
<!-- COURSE_FEEDBACK_QR:END -->
