---
theme: default
background: https://cover.sli.dev
class: text-center
highlighter: shiki
lineNumbers: false
info: |
  ## Lesson 4: Logistic Regression
  MPS311/439 Machine Learning - Dr. Wei Xing
drawings:
  persist: false
transition: slide-left
title: 'Lesson 4: Logistic Regression'
routerMode: hash
mdc: true
---

# Lesson 4: Logistic Regression
## Linear Classification and Model Evaluation

<div class="pt-12">
  <span class="text-xl">
    MPS311/439 Machine Learning<br>
    Dr. Wei Xing<br>
    2026–27
  </span>
</div>

---
layout: default
---

# Where We Left Off... 🎓

<div class="grid grid-cols-1 gap-4 mt-8">

<div class="bg-blue-50 p-4 rounded-lg">
<strong class="text-blue-700">Lesson 1:</strong> Introduction to ML, Python basics, data exploration
</div>

<div class="bg-green-50 p-4 rounded-lg">
<strong class="text-green-700">Lesson 2:</strong> Linear Regression - predicting continuous values<br/>
<span class="text-sm">(house prices, temperatures)</span>
</div>

<div class="bg-purple-50 p-4 rounded-lg">
<strong class="text-purple-700">Lesson 3:</strong> Feature Engineering & Regularization<br/>
<span class="text-sm">(polynomial features, Ridge/Lasso to prevent overfitting)</span>
</div>

<div class="bg-yellow-100 p-6 rounded-lg mt-6 border-2 border-yellow-500">
<div class="text-center text-xl font-bold text-yellow-800">
💡 But what if we want to predict <span class="text-red-600">categories</span> instead of numbers?
</div>
</div>

</div>

---
layout: two-cols
---

# Today's Challenge
## Classification Problems

<div class="mt-6">

**Definition:** Predicting <span class="text-red-600 font-bold">discrete categories</span> (classes), not continuous values

**Key characteristic:** Output is categorical
- Yes/No
- Class A/B/C
- 0 or 1

</div>

<div class="bg-blue-50 p-4 rounded-lg mt-6">
<strong class="text-blue-700">Today's Focus:</strong><br/>
<span class="text-2xl font-bold">Binary Classification</span><br/>
<span class="text-sm">(2 classes: 0 and 1)</span>
</div>

::right::

<div class="mt-12 ml-8">

## Real-World Examples

<div class="space-y-4 text-lg">

📧 **Email Filtering**
<div class="text-sm ml-6">Spam or Not Spam?</div>

🏥 **Medical Diagnosis**
<div class="text-sm ml-6">Disease or Healthy?</div>

💳 **Credit Approval**
<div class="text-sm ml-6">Approve or Reject?</div>

🖼️ **Image Recognition**
<div class="text-sm ml-6">Cat or Dog?</div>

</div>

</div>

---
layout: default
---

# Can We Just Use Linear Regression?

<div class="mt-8">

<div class="flex justify-center mb-6">
<img src="./figures/fig_regression_classification_failure.png" class="w-85% rounded-lg shadow-lg" />
</div>

<div class="bg-red-50 p-6 rounded-lg border-2 border-red-500">
<div class="text-red-800 font-bold text-2xl mb-4">❌ Problems with Linear Regression for Classification:</div>
<div class="grid grid-cols-1 md:grid-cols-3 gap-4 text-red-700">
<div class="bg-white p-3 rounded-lg">
<div class="font-bold mb-2">Unbounded Predictions</div>
<div class="text-sm">Predictions can be outside [0, 1] range</div>
</div>
<div class="bg-white p-3 rounded-lg">
<div class="font-bold mb-2">Outlier Sensitivity</div>
<div class="text-sm">Very sensitive to outliers in the data</div>
</div>
<div class="bg-white p-3 rounded-lg">
<div class="font-bold mb-2">No Probability Interpretation</div>
<div class="text-sm">Can't properly interpret as probabilities</div>
</div>
</div>
</div>

</div>

---
layout: default
---

# What We Need for Classification

<div class="mt-8 space-y-6">

<div class="flex items-start gap-4">
<div class="text-4xl">✅</div>
<div>
<strong class="text-xl text-green-700">Outputs between 0 and 1</strong>
<div class="text-gray-600">So we can interpret them as probabilities</div>
</div>
</div>

<div class="flex items-start gap-4">
<div class="text-4xl">✅</div>
<div>
<strong class="text-xl text-green-700">Smooth and differentiable</strong>
<div class="text-gray-600">So we can use gradient descent to train</div>
</div>
</div>

<div class="flex items-start gap-4">
<div class="text-4xl">✅</div>
<div>
<strong class="text-xl text-green-700">Proper probability interpretation</strong>
<div class="text-gray-600 font-mono">P(y=1|x) = probability of class 1 given features</div>
</div>
</div>

<div class="flex items-start gap-4">
<div class="text-4xl">✅</div>
<div>
<strong class="text-xl text-green-700">Appropriate loss function</strong>
<div class="text-gray-600">Designed for classification, not regression</div>
</div>
</div>

</div>

<div class="mt-8 bg-gradient-to-r from-blue-500 to-purple-600 p-6 rounded-lg text-white text-center">
<div class="text-3xl font-bold">🎯 Logistic Regression gives us all of this!</div>
</div>

---
layout: two-cols
---

# The Sigmoid Function
## Our "Squashing" Function

<div class="mt-4">

<img src="./figures/fig_sigmoid_function.png" class="w-full rounded-lg shadow-lg" />

</div>

::right::

<div class="mt-8 ml-6">

## Formula

<div class="my-6 text-center">

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

</div>

## Key Properties

<div class="space-y-3 mt-6">

<div class="bg-blue-50 p-3 rounded">
Maps any real number to (0, 1)
</div>

<div class="bg-green-50 p-3 rounded">
Center point: <code>σ(0) = 0.5</code>
</div>

<div class="bg-purple-50 p-3 rounded">
As <code>z → +∞</code>, <code>σ(z) → 1</code>
</div>

<div class="bg-orange-50 p-3 rounded">
As <code>z → -∞</code>, <code>σ(z) → 0</code>
</div>

<div class="bg-pink-50 p-3 rounded">
Smooth S-curve, differentiable
</div>

</div>

<div class="mt-4 bg-yellow-100 p-3 rounded border-2 border-yellow-500 text-sm">
<strong>Perfect for probabilities!</strong>
</div>

</div>

---
layout: default
---

# Logistic Regression Model
<!-- ## Two Simple Steps -->

<div class="flex flex-col items-center mt-8 space-y-6">

<div class="bg-blue-100 p-4 rounded-lg w-full max-w-3xl border-2 border-blue-500">
<div class="text-center">
<strong class="text-2xl text-blue-800">Step 1: Linear Combination</strong>
<div class="text-sm text-blue-600 mb-4">(Just like before!)</div>

<div class="my-4">

$$
z = w_0 + w_1 x_1 + w_2 x_2 + \cdots + w_p x_p = \mathbf{w}^T \mathbf{x}
$$

</div>

</div>
</div>

<div class="bg-green-100 p-4 rounded-lg w-full max-w-3xl border-2 border-green-500">
<div class="text-center">
<strong class="text-2xl text-green-800">Step 2: Apply Sigmoid</strong>
<div class="text-sm text-green-600 mb-4">(Squash to probability)</div>

<div class="my-4">
$$
P(y=1|\mathbf{x}) = \sigma(z) = \frac{1}{1 + e^{-z}}
$$
</div>

</div>
</div>

</div>

<div class="mt-8 text-center text-xl bg-purple-50 p-4 rounded">
<strong>Logistic Regression</strong> = <span class="text-blue-600">Linear Model</span> + <span class="text-green-600">Sigmoid Squashing</span>
</div>

---
layout: default
---

# Linear Decision Boundaries

<div class="mt-6">

<div class="grid grid-cols-3 gap-6">

<div class="col-span-2">
<img src="./figures/fig_decision_boundary_2d.png" class="w-90% rounded-lg shadow-lg" />
</div>

<div class="col-span-1 space-y-4">

<div class="bg-blue-50 p-4 rounded-lg border-2 border-blue-500">
<div class="font-bold text-lg mb-3 text-blue-800">Decision Rule</div>
<div class="space-y-2 text-sm">
<div>• If <code>P(y=1|x) > 0.5</code></div>
<div class="ml-4">→ Predict <strong>Class 1</strong></div>
<div>• If <code>P(y=1|x) ≤ 0.5</code></div>
<div class="ml-4">→ Predict <strong>Class 0</strong></div>
</div>
</div>

<div class="bg-green-50 p-4 rounded-lg border-2 border-green-500">
<div class="font-bold text-lg mb-3 text-green-800">Key Points</div>
<div class="space-y-2 text-sm">
<div>• <strong>Decision boundary:</strong></div>
<div class="ml-4">where w<sup>T</sup>x = 0</div>
<div>• Still <strong class="text-red-600">linear</strong> (straight line)</div>
<div>• <strong>Far from boundary</strong> = high confidence</div>
<div>• <strong>Near boundary</strong> = uncertain</div>
</div>
</div>

<div class="bg-yellow-50 p-3 rounded-lg border-2 border-yellow-500">
<div class="text-sm">💡 Colored regions show <strong>probability contours</strong></div>
</div>

</div>

</div>

</div>

---
layout: default
---

# Training with Cross-Entropy Loss

<div class="text-center mt-2">
<div class="text-xl mb-2">How do we train logistic regression?</div>
</div>

<div class="bg-purple-50 p-2 rounded-lg border-2 border-purple-500 my-6">
<div class="text-center">
<strong class="text-2xl text-purple-800">Binary Cross-Entropy Loss</strong>
<div class="my-6 text-lg">

$$
L(y, \hat{y}) = -\left[ y \log(\hat{y}) + (1-y) \log(1-\hat{y}) \right]
$$

</div>
<div class="text-sm text-purple-700">

where $\hat{y} = P(y=1|\mathbf{x})$ is our predicted probability

</div>
</div>
</div>

<div class="grid grid-cols-2 gap-6 mt-6">

<div class="bg-blue-50 p-2 rounded-lg">
<strong class="text-blue-800">When actual y = 1:</strong>

$$L = -\log(\hat{y})$$

<div class="text-sm">→ Penalizes low predictions for positive class</div>
</div>

<div class="bg-green-50 p-2 rounded-lg">
<strong class="text-green-800">When actual y = 0:</strong>

$$L = -\log(1-\hat{y})$$

<div class="text-sm">→ Penalizes high predictions for negative class</div>
</div>

</div>

<div class="mt-6 bg-yellow-100 p-4 rounded border-2 border-yellow-500">
<strong>Why not MSE?</strong> Cross-entropy creates a <span class="text-red-600 font-bold">convex</span> optimization problem → easier to train!
</div>

---
layout: default
---

# Why Cross-Entropy Loss? 🤔

<div class="mt-6">

## The Problem with Mean Squared Error (MSE)

<div class="grid grid-cols-2 gap-6 mt-4">

<div class="bg-red-50 p-4 rounded-lg border-2 border-red-500">
<strong class="text-red-800 text-lg">❌ MSE for Classification</strong>
<div class="my-4 text-center">

$$
L = \frac{1}{2}(y - \hat{y})^2
$$

</div>
<div class="space-y-2 text-sm">
<div>• Creates <strong>non-convex</strong> loss surface</div>
<div>• Many local minima</div>
<div>• Gradient can vanish</div>
<div>• Hard to optimize</div>
</div>
</div>

<div class="bg-green-50 p-4 rounded-lg border-2 border-green-500">
<strong class="text-green-800 text-lg">✅ Cross-Entropy</strong>
<div class="my-4 text-center text-sm">

$$
L = -[y \log(\hat{y}) + (1-y) \log(1-\hat{y})]
$$

</div>
<div class="space-y-2 text-sm">
<div>• Creates <strong>convex</strong> loss surface</div>
<div>• Single global minimum</div>
<div>• Strong gradients when wrong</div>
<div>• Reliable training</div>
</div>
</div>

</div>

<div class="mt-6 bg-blue-100 p-6 rounded-lg border-2 border-blue-500">
<strong class="text-xl">📊 Key Insight:</strong>
<div class="mt-3">Cross-entropy naturally handles probabilities and provides the right gradient signal for classification tasks!</div>
</div>

</div>

---
layout: default
---

# Understanding the Gradient

<div class="mt-6">

<div class="bg-purple-50 p-6 rounded-lg border-2 border-purple-500">
<strong class="text-xl text-purple-800">Good News: Simple Gradient! 🎉</strong>

<div class="my-6 text-center">

The gradient of cross-entropy loss w.r.t. weights is surprisingly clean:

$$
\frac{\partial L}{\partial w_j} = (\hat{y} - y) x_j
$$

</div>

<div class="text-center text-gray-700">
(Same form as linear regression, but with sigmoid-transformed predictions!)
</div>
</div>

<div class="grid grid-cols-2 gap-6 mt-6">

<div class="bg-blue-50 p-4 rounded-lg">
<strong class="text-blue-800">Training Algorithm:</strong>
<div class="mt-3 space-y-2 text-sm">
<div>1. Initialize weights randomly</div>
<div>

2. Predict: $\hat{y} = \sigma(\mathbf{w}^T \mathbf{x})$

</div>
<div>3. Compute gradient</div>
<div>

4. Update: $w_j := w_j - \alpha \frac{\partial L}{\partial w_j}$

</div>
<div>5. Repeat until convergence</div>
</div>
</div>

<div class="bg-green-50 p-4 rounded-lg">
<strong class="text-green-800">In Practice:</strong>
<div class="mt-3 space-y-2 text-sm">
<div>• Use mini-batch gradient descent</div>
<div>• Add regularization (L2/L1)</div>
<div>• Use optimizers like Adam</div>
<div>• sklearn does all this for you!</div>
</div>
</div>

</div>

</div>

---
layout: center
class: text-center
---

# Evaluation Metrics 📊
## Beyond Accuracy

<div class="mt-8 space-y-4">

<div class="text-2xl">
Why isn't accuracy enough?
</div>

<div class="bg-red-100 p-6 rounded-lg border-2 border-red-500 inline-block text-left max-w-2xl">
<strong class="text-red-800 text-xl">Example: Rare Disease Detection</strong>
<div class="mt-4 space-y-2">
<div>• Disease prevalence: 1% of population</div>
<div>• A model that <strong>always predicts "healthy"</strong></div>
<div class="ml-6">→ Gets <strong>99% accuracy</strong>! 🤯</div>
<div class="ml-6">→ But catches <strong>0% of sick patients</strong>! 😱</div>
</div>
</div>

<div class="mt-8 text-xl font-bold text-blue-600">
We need better metrics for imbalanced problems!
</div>

</div>

---
layout: default
---

# Confusion Matrix 📋

<div class="grid grid-cols-2 gap-8 mt-6">

<div>

<div class="bg-gray-50 p-6 rounded-lg border-2 border-gray-400">
<table class="w-full text-center">
<thead>
<tr>
<th colspan="2" rowspan="2" class="border-b-2 border-r-2 border-gray-400"></th>
<th colspan="2" class="border-b-2 border-gray-400 pb-2">Predicted</th>
</tr>
<tr>
<th class="border-b-2 border-gray-400 p-2 bg-blue-100">Negative (0)</th>
<th class="border-b-2 border-gray-400 p-2 bg-blue-100">Positive (1)</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="2" class="border-r-2 border-gray-400 font-bold bg-green-100" style="writing-mode: vertical-lr; transform: rotate(180deg);">Actual</td>
<td class="border-r-2 border-b border-gray-400 p-2 bg-green-100 font-bold">Negative (0)</td>
<td class="border-r border-b border-gray-400 p-3 bg-green-50">
<div class="font-bold text-green-700">TN</div>
<div class="text-xs">True Negative</div>
</td>
<td class="border-b border-gray-400 p-3 bg-red-50">
<div class="font-bold text-red-700">FP</div>
<div class="text-xs">False Positive</div>
</td>
</tr>
<tr>
<td class="border-r-2 border-gray-400 p-2 bg-green-100 font-bold">Positive (1)</td>
<td class="border-r border-gray-400 p-3 bg-red-50">
<div class="font-bold text-red-700">FN</div>
<div class="text-xs">False Negative</div>
</td>
<td class="p-3 bg-green-50">
<div class="font-bold text-green-700">TP</div>
<div class="text-xs">True Positive</div>
</td>
</tr>
</tbody>
</table>
</div>

</div>

<div>

<div class="space-y-4">

<div class="bg-green-50 p-4 rounded-lg border-2 border-green-500">
<strong class="text-green-800">True Positive (TP):</strong>
<div class="text-sm mt-1">Correctly predicted positive</div>
</div>

<div class="bg-green-50 p-4 rounded-lg border-2 border-green-500">
<strong class="text-green-800">True Negative (TN):</strong>
<div class="text-sm mt-1">Correctly predicted negative</div>
</div>

<div class="bg-red-50 p-4 rounded-lg border-2 border-red-500">
<strong class="text-red-800">False Positive (FP):</strong>
<div class="text-sm mt-1">Predicted positive, actually negative<br/>(Type I Error)</div>
</div>

<div class="bg-red-50 p-4 rounded-lg border-2 border-red-500">
<strong class="text-red-800">False Negative (FN):</strong>
<div class="text-sm mt-1">Predicted negative, actually positive<br/>(Type II Error)</div>
</div>

</div>

</div>

</div>

---
layout: default
---

# Key Metrics from Confusion Matrix

<div class="mt-2 space-y-2">

<div class="bg-blue-50 p-2 rounded-lg border-2 border-blue-500">
<strong class="text-xl text-blue-800">Accuracy</strong>
<div class="my-3">

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

</div>
<div class="text-sm">Overall correctness - but <strong>misleading with imbalanced data!</strong></div>
</div>

<div class="grid grid-cols-2 gap-6">

<div class="bg-purple-50 p-2 rounded-lg border-1 border-purple-500">
<strong class="text-xl text-purple-800">Precision</strong>
<div class="my-3">

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

</div>
<div class="text-sm">Of predicted positives, how many are correct?</div>
</div>

<div class="bg-green-50 p-2 rounded-lg border-1 border-green-500">
<strong class="text-xl text-green-800">Recall (Sensitivity)</strong>
<div class="my-3">

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

</div>
<div class="text-sm">Of actual positives, how many did we catch?</div>
</div>

</div>

<div class="bg-orange-50 p-2 rounded-lg border-1 border-orange-500">
<strong class="text-xl text-orange-800">F1-Score</strong>
<div class="my-3">

$$
F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
$$

</div>
<div class="text-sm">Harmonic mean of precision and recall - balances both metrics</div>
</div>

</div>

---
layout: two-cols
---

# Precision vs Recall Trade-off

<div class="mt-4">

## Precision
<div class="text-sm text-gray-600 mb-2">"Of all positive predictions, how many are correct?"</div>

<div class="my-4 bg-blue-50 p-4 rounded">

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

</div>

<div class="bg-blue-100 p-3 rounded text-sm">
<strong class="text-blue-800">High Precision = Few false alarms</strong>
</div>

<div class="mt-3 p-3 bg-gray-50 rounded text-sm">
<strong>Example:</strong> Email spam filter
<div class="mt-1">→ Avoid marking important emails as spam!</div>
</div>

</div>

::right::

<div class="mt-4 ml-4">

## Recall
<div class="text-sm text-gray-600 mb-2">"Of all actual positives, how many did I catch?"</div>

<div class="my-4 bg-green-50 p-4 rounded">

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

</div>

<div class="bg-green-100 p-3 rounded text-sm">
<strong class="text-green-800">High Recall = Few missed detections</strong>
</div>

<div class="mt-3 p-3 bg-gray-50 rounded text-sm">
<strong>Example:</strong> Disease screening
<div class="mt-1">→ Catch all patients with disease!</div>
</div>

</div>

<div class="col-span-2 mt-6 bg-yellow-100 p-4 rounded border-2 border-yellow-500">
<strong>🏥 Medical Diagnosis Example:</strong>
<div class="grid grid-cols-2 gap-3 mt-2 text-sm">
<div><strong>High Recall</strong> → Aggressive screening (don't miss sick patients)</div>
<div><strong>High Precision</strong> → Conservative diagnosis (only when very confident)</div>
</div>
<div class="mt-2 text-center font-bold text-red-600">⚖️ Tradeoff: You usually can't maximize both!</div>
</div>

---
layout: default
---

# ROC Curve & AUC
## Performance Across All Thresholds

<div class="grid grid-cols-5 gap-6 mt-6">

<div class="col-span-3">
<img src="./figures/fig_roc_curve.png" class="w-75% rounded-lg shadow-lg" />
</div>

<div class="col-span-2">

<div class="space-y-3 text-sm">

<div class="bg-blue-50 p-3 rounded">
<strong>ROC Curve:</strong> Shows performance at all possible thresholds
</div>

<div class="bg-gray-50 p-3 rounded">
<strong>X-axis:</strong> False Positive Rate
</div>

<div class="bg-gray-50 p-3 rounded">
<strong>Y-axis:</strong> True Positive Rate (Recall)
</div>

<div class="bg-yellow-50 p-3 rounded">
<strong>Diagonal line:</strong> Random guessing (AUC = 0.5)
</div>

<div class="bg-green-50 p-4 rounded border-2 border-green-500">
<strong class="text-green-800">AUC (Area Under Curve):</strong>
<ul class="mt-2 space-y-1 text-xs">
<li>0.5 = Random guessing</li>
<li>0.7-0.8 = Acceptable</li>
<li>0.8-0.9 = Good</li>
<li>0.9-1.0 = Excellent</li>
<li>1.0 = Perfect classifier</li>
</ul>
</div>

</div>

</div>

</div>

<div class="mt-4 bg-purple-100 p-4 rounded text-center border-2 border-purple-500">
<strong class="text-lg">AUC lets you compare models without choosing a specific threshold!</strong>
</div>

---
layout: center
class: text-center
---

# Live Demo! 🚀
## Let's Build a Classifier

<div class="mt-8 space-y-4">

<div class="text-2xl font-bold text-blue-600">
Breast Cancer Classification
</div>

<div class="bg-blue-50 p-6 rounded-lg inline-block text-left">
<div class="space-y-2">
<div><strong>Dataset:</strong> Breast cancer (569 samples, 30 features)</div>
<div><strong>Task:</strong> Predict malignant (0) or benign (1)</div>
<div><strong>Tool:</strong> sklearn LogisticRegression</div>
</div>
</div>

<div class="text-xl mt-8">
Watch how simple this is with sklearn!
</div>

</div>

<div class="mt-12 text-6xl">
💻 → Google Colab
</div>

---
layout: center
class: text-center
---

# Demo Part 2
## Evaluating Our Classifier

<div class="mt-8">

<div class="grid grid-cols-2 gap-8 text-left max-w-4xl mx-auto">

<div class="bg-green-50 p-6 rounded-lg border-2 border-green-500">
<div class="text-xl font-bold text-green-800 mb-4">Part A: Basic Metrics</div>
<ul class="space-y-2">
<li>✓ Display confusion matrix</li>
<li>✓ Calculate accuracy, precision, recall</li>
<li>✓ Interpret the results</li>
</ul>
</div>

<div class="bg-purple-50 p-6 rounded-lg border-2 border-purple-500">
<div class="text-xl font-bold text-purple-800 mb-4">Part B: Threshold Tuning</div>
<ul class="space-y-2">
<li>• Default: threshold = 0.5</li>
<li>• Try threshold = 0.3 (aggressive)</li>
<li>• Try threshold = 0.7 (conservative)</li>
</ul>
</div>

</div>

<div class="mt-8 bg-yellow-100 p-4 rounded-lg border-2 border-yellow-500 inline-block">
<strong class="text-lg">⚡ See the precision-recall tradeoff in action!</strong>
</div>

</div>

<div class="mt-8 text-5xl">
💻 → Google Colab
</div>

---
layout: default
---

# Key Takeaways 🎯

<div class="mt-6 space-y-4">

<div class="flex items-start gap-4 bg-blue-50 p-4 rounded-lg">
<div class="text-3xl">1️⃣</div>
<div>
<strong class="text-xl text-blue-800">Classification predicts categories</strong>, not continuous values
</div>
</div>

<div class="flex items-start gap-4 bg-green-50 p-4 rounded-lg">
<div class="text-3xl">2️⃣</div>
<div>
<strong class="text-xl text-green-800">Logistic Regression = Linear model + Sigmoid</strong>
<div class="text-sm mt-1">• Outputs probabilities between 0 and 1</div>
<div class="text-sm">• Decision boundary is still <strong>linear</strong></div>
</div>
</div>

<div class="flex items-start gap-4 bg-purple-50 p-4 rounded-lg">
<div class="text-3xl">3️⃣</div>
<div>
<strong class="text-xl text-purple-800">Cross-Entropy Loss</strong> is the right loss function
<div class="text-sm mt-1">• Creates convex optimization → reliable training</div>
</div>
</div>

<div class="flex items-start gap-4 bg-orange-50 p-4 rounded-lg">
<div class="text-3xl">4️⃣</div>
<div>
<strong class="text-xl text-orange-800">Multiple metrics needed</strong> to evaluate classifiers
<div class="text-sm mt-1">• Accuracy misleads with imbalanced data</div>
<div class="text-sm">• Precision vs Recall tradeoff depends on your problem</div>
</div>
</div>

<div class="flex items-start gap-4 bg-pink-50 p-4 rounded-lg">
<div class="text-3xl">5️⃣</div>
<div>
<strong class="text-xl text-pink-800">sklearn makes it simple</strong> - just a few lines of code!
</div>
</div>

</div>

---
layout: two-cols
---

# When Does It Work?

<div class="mt-6">

## ✅ Works Well

<div class="bg-green-50 p-4 rounded-lg space-y-2 text-sm">
<div>• Linearly separable classes</div>
<div>• Need interpretable models</div>
<div>• Want probability estimates</div>
<div>• High-dimensional data</div>
</div>

<div class="mt-6">
<img src="./figures/fig_linear_vs_nonlinear.png" class="w-full rounded-lg shadow" />
</div>

</div>

::right::

<div class="mt-6 ml-4">

## ❌ Struggles With

<div class="bg-red-50 p-4 rounded-lg space-y-2 text-sm">
<div>• Non-linear decision boundaries</div>
<div>• Complex feature interactions</div>
<div>• Classes mixed in complex ways</div>
</div>

<div class="mt-6 bg-blue-100 p-4 rounded-lg border-2 border-blue-500">
<strong class="text-blue-800">Solutions:</strong>
<ul class="mt-2 space-y-1 text-sm">
<li>• Feature engineering (polynomial features)</li>
<li>• Try non-linear models</li>
<li>• Next week: LDA/QDA for curved boundaries!</li>
</ul>
</div>

</div>

---
layout: center
class: text-center
---

# Looking Ahead 🔮

<div class="mt-4">

<div class="grid grid-cols-2 gap-8 max-w-4xl mx-auto">

<div class="bg-green-100 p-4 rounded-lg border-2 border-green-500">
<div class="text-3xl mb-4">✅</div>
<strong class="text-xl text-green-800">This Week</strong>
<div class="mt-4 text-lg">Logistic Regression</div>
<div class="text-sm text-gray-600">Linear decision boundaries</div>
</div>

<div class="bg-blue-100 p-4 rounded-lg border-2 border-blue-500">
<div class="text-3xl mb-4">🔜</div>
<strong class="text-xl text-blue-800">Next Week</strong>
<div class="mt-4 text-lg font-bold">LDA & QDA</div>
<div class="text-sm text-gray-600 mt-2">
<div>• Linear Discriminant Analysis</div>
<div>• Quadratic boundaries (curved!)</div>
<div>• When to use each method</div>
</div>
</div>

</div>

</div>

<div class="mt-8 text-6xl">
Questions? 🙋‍♀️🙋‍♂️
</div>

<div class="mt-8 space-y-2 text-lg">
<div>📝 Try the lab exercises this week</div>
<div>🔬 Experiment with different datasets</div>
<div>💬 Come to office hours if you need help!</div>
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

<img src="./feedback-qr.svg" alt="Feedback QR code for Lesson 04 lecture" class="w-44 mx-auto rounded-lg shadow" />

<p class="text-sm mt-4 opacity-70">MPS311/439 · 2026-27 · Lesson 04 lecture</p>
<!-- COURSE_FEEDBACK_QR:END -->
