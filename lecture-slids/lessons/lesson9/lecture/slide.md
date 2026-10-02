---
theme: default
background: https://cover.sli.dev
class: text-center
highlighter: shiki
lineNumbers: false
info: |
  ## Lesson 9: Neural Networks
  MPS311/439 Machine Learning - Dr. Wei Xing
drawings:
  persist: false
transition: slide-left
title: 'Lesson 9: Neural Networks'
routerMode: hash
mdc: true
---

# Lesson 9: Neural Networks

## From Logistic Regression to Deep Learning

<div class="pt-12">
  <span class="text-xl">
    MPS311/439 Machine Learning<br>
    Dr. Wei Xing<br>
    2026–27
  </span>
</div>

---
layout: two-cols
---

# Where We Left Off

## Our Building Block: Logistic Regression

<div class="mt-4">

**The Formula:**

$$\hat{y} = \sigma(\mathbf{w}^T \mathbf{x} + b)$$

**What it does:**
- Takes weighted sum of inputs
- Applies sigmoid activation  
- Creates **one decision boundary**
- Perfect for linearly separable problems

</div>

<div class="mt-8 p-4 bg-blue-50 rounded">
  <strong>Key insight:</strong> One unit, one line, one decision
</div>

::right::

<div class="mt-12 ml-8">
  <img src="/figures/fig5_activation_functions.png" class="w-100" />
  <p class="text-sm text-gray-600 mt-2">Sigmoid activation function</p>
</div>

<div class="mt-8 text-center text-xl font-bold text-blue-600">
  But what if one line isn't enough?
</div>

---

# The XOR Challenge

A Problem Logistic Regression Cannot Solve

<div class="grid grid-cols-2 gap-8 mt-1">

<div>

**The XOR Truth Table:**

<div class="p-1 bg-gray-50 rounded font-mono mt-1">

| x₁ | x₂ | Output |
|----|-------|--------|
| 0  | 0     | 0      |
| 0  | 1     | 1      |
| 1  | 0     | 1      |
| 1  | 1     | 0      |

</div>

<div class="mt-1 p-1 bg-yellow-50 rounded border-2 border-yellow-400">
  <strong>Key Insight:</strong><br/>
  This is <span class="text-red-600 font-bold">not linearly separable</span><br/>
  We need something more powerful!
</div>

</div>

<div>
  <img src="/figures/fig1_xor_problem_viz.png" class="w-full" />
  <p class="text-sm text-gray-600 text-center mt-2">
    No single straight line can separate these classes!
  </p>
</div>

</div>

---

# Intuitive Solution

What If We Use Multiple Decision Boundaries?

<div class="mt-1">
  <img src="/figures/fig2_single_vs_multiple_units.png" class="w-50% mx-auto overflow-hidden" />
</div>

<div class="grid grid-cols-3 gap-4 mt-0">

<div class="p-4 bg-blue-50 rounded">
  <strong>Question 1:</strong><br/>
  What if we had <span class="text-blue-600 font-bold">two</span> logistic units working together?
</div>

<div class="p-4 bg-green-50 rounded">
  <strong>Question 2:</strong><br/>
  One detects "x₁ AND NOT x₂"<br/>
  Another detects "x₂ AND NOT x₁"
</div>

<div class="p-4 bg-purple-50 rounded">
  <strong>Question 3:</strong><br/>
  Then combine their outputs to make the final decision!
</div>

</div>

<div class="mt-6 text-center text-2xl font-bold text-purple-600">
  This is the core idea of <span class="underline">Neural Networks</span>
</div>

---

# Introducing Neural Networks

Neural Networks = Stacked Logistic Units

<div class="mt-2">
  <img src="/figures/fig3_xor_solution_network.png" class="w-80 max-w-3xl mx-auto" />
</div>

<div class="grid grid-cols-3 gap-6 mt-1">

<div class="p-4 bg-blue-100 rounded text-center">
  <div class="text-lg font-bold mb-2">Input Layer</div>
  <div>Your features</div>
  <div class="font-mono text-sm">(x₁, x₂, ..., xₙ)</div>
</div>

<div class="p-2 bg-green-100 rounded text-center">
  <div class="text-lg font-bold mb-2">Hidden Layer</div>
  <div>Intermediate processing</div>
  <div>Learns representations</div>
</div>

<div class="p-2 bg-orange-100 rounded text-center">
  <div class="text-lg font-bold mb-2">Output Layer</div>
  <div>Final prediction</div>
  <div class="font-mono text-sm">ŷ</div>
</div>

</div>

<div class="mt-2 p-2 bg-purple-50 rounded text-center">
  <strong>Key Insight:</strong> Each hidden unit is just logistic regression!<br/>
  Power comes from <span class="text-purple-600 font-bold">composition</span>
</div>

---

# Architecture Fundamentals

<!-- Neural Network Structure -->

<div class="mt-0">
  <img src="/figures/fig4_nn_architecture_labeled.png" class="w-100 max-w-4xl mx-auto" />
</div>

<div class="grid grid-cols-2 gap-6 mt--6">

<div>

**Key Terminology:**

<div class="space-y-1 mt--2">
  <div class="p-1 bg-gray-50 rounded">
    <strong>Neuron/Unit:</strong> Single computational unit
  </div>
  <div class="p-1 bg-gray-50 rounded">
    <strong>Weights (W):</strong> Parameters connecting layers
  </div>
  <div class="p-1 bg-gray-50 rounded">
    <strong>Biases (b):</strong> Offset parameters
  </div>
  <div class="p-1 bg-gray-50 rounded">
    <strong>Activation:</strong> Output after activation function
  </div>
</div>

</div>

<div>

**Mathematical Notation:**

<div class="p-0 bg-blue-50 rounded mt--2">

Layer $l$ computation:

$$a^{(l)} = \sigma(W^{(l)} a^{(l-1)} + b^{(l)})$$

</div>

<div class="mt-1 p-1 bg-yellow-50 rounded text-sm">
  <strong>Note:</strong> "Deep" = multiple hidden layers<br/>
  (We'll start with 1)
</div>

</div>

</div>

---
layout: two-cols
---

# Why Non-linearity Matters

<!-- The Critical Ingredient -->

<div class="mt-0">

**Without activation functions:**

<div class="p-1 bg-red-50 rounded mt-4">

$$W^{(2)}(W^{(1)}\mathbf{x}) = (W^{(2)}W^{(1)})\mathbf{x}$$

Just linear regression!<br/>
All layers collapse to one!

</div>

<div class="p-1 bg-green-50 rounded mt-2">
  <strong class="text-green-700">Non-linearity gives networks their power</strong>
</div>

**Common Choices:**

<div class="mt-4 text-sm">

| Function | Use Case |
|----------|----------|
| **ReLU** | Hidden layers (default!) |
| **Sigmoid** | Binary output |
| **Tanh** | Hidden (zero-centered) |

</div>

</div>

::right::

<div class="ml-8 mt-8">
  <img src="/figures/fig5_activation_functions.png" class="w-full" />
  
  <div class="mt-6 p-3 bg-blue-50 rounded">
    <strong>Rule of thumb:</strong><br/>
    Start with ReLU for hidden layers
  </div>
</div>



---

# Mathematical Formulation: Forward Propagation

<!-- **For a network with L layers:** -->

<div class="mt-0">

**Initialize:**

<div class="p-0 bg-gray-50 rounded font-mono">

a⁽⁰⁾ = x  (input)

</div>

</div>

<div class="mt-1">

**For each layer l = 1 to L:**

<div class="p-0 bg-blue-50 rounded mt-4">

**Pre-activation:**

$$z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}$$

**Activation:**

$$a^{(l)} = \sigma(z^{(l)})$$

</div>

</div>

<div class="mt-1">

**Final output:** $\hat{y} = a^{(L)}$

</div>

<div class="mt-1 p-0 bg-yellow-50 rounded text-sm">

**Matrix Shapes:** If layer $l-1$ has $n_{l-1}$ units and layer $l$ has $n_l$ units:

- $W^{(l)}$ has shape $(n_l, n_{l-1})$
- $b^{(l)}$ has shape $(n_l, 1)$

</div>

---

# Implementation with Keras

Solving XOR with ~10 Lines of Code!
```python
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

# XOR dataset
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0, 1, 1, 0])

# Build network
model = Sequential([
    Dense(4, input_dim=2, activation='relu'),  # Hidden layer
    Dense(1, activation='sigmoid')             # Output layer
])

# Compile and train
model.compile(optimizer='adam', loss='binary_crossentropy')
model.fit(X, y, epochs=1000, verbose=0)

# Predict
print(model.predict(X))
# Output: [[0.02], [0.98], [0.98], [0.02]] ✓
```

<div class="mt-4 grid grid-cols-3 gap-4">
  <div class="p-3 bg-green-50 rounded">✅ Keras handles complexity</div>
  <div class="p-3 bg-blue-50 rounded">✅ Focus on architecture</div>
  <div class="p-3 bg-purple-50 rounded">✅ Fundamentals help usage</div>
</div>

---
layout: two-cols
---

# Training Process

<!-- How Neural Networks Learn -->

<div class="mt-1">

**The Process:**

<div class="space-y-2 mt-4">

<div class="p-1 bg-blue-50 rounded">
  <strong>1.</strong> Forward pass: Compute predictions
</div>

<div class="p-1 bg-blue-50 rounded">
  <strong>2.</strong> Calculate loss: Measure error
</div>

<div class="p-1 bg-blue-50 rounded">
  <strong>3.</strong> Backward pass: Compute gradients
</div>

<div class="p-1 bg-blue-50 rounded">
  <strong>4.</strong> Update weights: Gradient descent
</div>

<div class="p-1 bg-blue-50 rounded">
  <strong>5.</strong> Repeat until convergence
</div>

</div>

**Loss Functions:**

<div class="mt-4 text-sm">
  <div class="p-2 bg-gray-50 rounded mb-1">
    Binary: <code>binary_crossentropy</code>
  </div>
  <div class="p-2 bg-gray-50 rounded mb-1">
    Regression: <code>mean_squared_error</code>
  </div>
  <div class="p-2 bg-gray-50 rounded">
    Multi-class: <code>categorical_crossentropy</code>
  </div>
</div>

</div>

::right::

<div class="ml-6">
  <img src="/figures/fig7_training_curves.png" class="w-full" />
  
  <div class="grid grid-cols-3 gap-2 mt-4 text-sm">
    <div class="p-2 bg-green-50 rounded">
      ✅ <strong>Good:</strong><br/>Both decrease
    </div>
    <div class="p-2 bg-yellow-50 rounded">
      ⚠️ <strong>Overfitting:</strong><br/>Val increases
    </div>
    <div class="p-2 bg-red-50 rounded">
      ⚠️ <strong>Underfitting:</strong><br/>Both high
    </div>
  </div>
  
  <div class="mt-4 p-3 bg-blue-50 rounded text-center font-bold">
    Always monitor training curves!
  </div>
</div>

---

# Practical Guidelines

<!-- When and How to Use Neural Networks -->

<div class="grid grid-cols-3 gap-4 mt-6">

<div class="p-2 bg-orange-50 rounded">

**How many hidden layers?**

- Start with **1 hidden layer**
- Add more only if needed
- Most problems: 1-2 layers

</div>

<div class="p-2 bg-green-50 rounded">

**How many hidden units?**

- Start small: 4-8 units
- Between input and output size
- Increase if underfitting

</div>

<div class="p-2 bg-purple-50 rounded">

**When to use NNs?**

- ✅ Complex patterns
- ✅ Large datasets (1000+)
- ✅ Simpler models fail
- ❌ Small data, linear, interpretability

</div>

</div>

<div class="mt-6">

**Your Workflow:**

<div class="p-4 bg-blue-50 rounded text-center font-mono text-lg mt-2">
  Normalize → Start simple → Train → Monitor → Iterate
</div>

</div>

<div class="mt-4 grid grid-cols-2 gap-4 text-sm">
  <div class="p-3 bg-red-50 rounded">
    <strong>Common Pitfall:</strong> Forgetting to normalize features
  </div>
  <div class="p-3 bg-yellow-50 rounded">
    <strong>Common Pitfall:</strong> Not splitting train/test data
  </div>
</div>

---

# Backpropagation
Understanding How Gradients Are Computed

<div class="grid grid-cols-2 gap-6 mt-1">

<div>

**The Challenge:**

- Millions of parameters
- Need gradients: $\frac{\partial L}{\partial w_{ij}}$

**The Solution: Chain Rule**

<div class="p-0 bg-blue-50 rounded mt-4">

**Key Insight:**

<div class="mt-2">

$$\frac{\partial L}{\partial w^{(1)}} = \frac{\partial L}{\partial a^{(3)}} \times \frac{\partial a^{(3)}}{\partial a^{(2)}} \times \frac{\partial a^{(2)}}{\partial w^{(1)}}$$

</div>

<div class="mt-3 font-bold text-blue-700">
  Backpropagation: Reuse computations by going backwards
</div>

</div>

**Efficiency:**

<div class="grid grid-cols-2 gap-2 mt-0 text-sm">
  <div class="p-0.5 bg-red-100 rounded">Naive: O(n²)</div>
  <div class="p-0.5 bg-green-100 rounded">Backprop: O(n)</div>
</div>

</div>

<div>
  <img src="/figures/fig8_backprop_flow.png" class="w-full" />
  
  <div class="mt-4 p-3 bg-yellow-50 rounded text-sm text-center">
    <strong>Note:</strong> See lecture notes for full derivation!
  </div>
</div>

</div>

---

# Advanced Techniques

Beyond Basic Networks

<div class="grid grid-cols-2 gap-4 mt-6">

<div class="p-4 bg-pink-50 rounded">

**L2 Regularization**

- Penalty: $\lambda \|W\|^2$
- Prevents large weights
- Keras: `kernel_regularizer=l2(0.01)`

</div>

<div class="p-4 bg-green-50 rounded">

**Dropout**

- Randomly drop neurons (p=0.5)
- Forces robust learning
- Keras: `Dropout(0.5)`

</div>

<div class="p-4 bg-teal-50 rounded">

**Batch Normalization**

- Normalizes activations
- Faster, stable training
- Keras: `BatchNormalization()`

</div>

<div class="p-4 bg-yellow-50 rounded">

**Optimizers**

- **Adam**: Best default
- **SGD + Momentum**: Classic
- **RMSprop**: Good for RNNs

</div>

</div>

<div class="mt-6 p-4 bg-blue-50 rounded text-center">
  <strong>Details and code examples:</strong> See Lecture Notes Section 11
</div>

---

# Interactive Demo Time!

Let's See It In Action

<div class="grid grid-cols-2 gap-6 mt-2">

<div>

**Live Demonstrations:**

<div class="space-y-3 mt-4">

<div class="p-3 bg-blue-50 rounded">
  <strong>🎯 Demo 1:</strong> XOR Decision Boundary Evolution<br/>
  <span class="text-sm">Watch network learn in real-time</span>
</div>

<div class="p-3 bg-green-50 rounded">
  <strong>📊 Demo 2:</strong> Architecture Explorer<br/>
  <span class="text-sm">Adjust layers/units, see effects</span>
</div>

<div class="p-3 bg-purple-50 rounded">
  <strong>🔧 Demo 3:</strong> Activation Function Comparison<br/>
  <span class="text-sm">Compare ReLU, sigmoid, tanh</span>
</div>

<div class="p-3 bg-orange-50 rounded">
  <strong>📈 Demo 4:</strong> Training Dynamics Visualizer<br/>
  <span class="text-sm">Monitor curves, detect overfitting</span>
</div>

</div>

</div>

<div>

<div class="p-6 bg-gradient-to-br from-blue-100 to-purple-100 rounded-lg text-center h-full flex flex-col justify-center">
  <div class="text-3xl mb-4">🚀</div>
  <!-- <div class="text-xl font-bold mb-4">Open Google Colab Now!</div> -->
  <a 
    href="https://playground.tensorflow.org" 
    target="_blank" 
    rel="noopener noreferrer"
    class="text-3xl mb-4 hover:text-blue-600"
  >
    [playground.tensorflow.org]
  </a>
  
  <div class="text-lg font-semibold text-purple-700">
    Play with interactive demo!
  <div class="flex justify-center mt-4">
    <img src="./qrcode_playground.tensorflow.org.png" class="w-55"  />
  </div>
  </div>
  


</div>

</div>

</div>

---

# Summary: What We Learned

The Journey from Logistic Regression to Neural Networks

<div class="p-6 bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg mt-4 text-center text-xl font-semibold">
  Logistic Regression → XOR Problem → Stack Units → Neural Networks
</div>

<div class="grid grid-cols-2 gap-6 mt-6">

<div>

**Core Concepts:**

<div class="space-y-2 mt-3">
  <div class="flex items-start">
    <span class="text-green-600 mr-2">✅</span>
    <span>Neural networks = stacked logistic units</span>
  </div>
  <div class="flex items-start">
    <span class="text-green-600 mr-2">✅</span>
    <span>Hidden layers learn representations</span>
  </div>
  <div class="flex items-start">
    <span class="text-green-600 mr-2">✅</span>
    <span>Activation functions provide non-linearity</span>
  </div>
  <div class="flex items-start">
    <span class="text-green-600 mr-2">✅</span>
    <span>Forward propagation: layer-by-layer</span>
  </div>
  <div class="flex items-start">
    <span class="text-green-600 mr-2">✅</span>
    <span>Training: gradient descent + backprop</span>
  </div>
  <div class="flex items-start">
    <span class="text-green-600 mr-2">✅</span>
    <span>Keras makes it simple (~10 lines!)</span>
  </div>
</div>

</div>

<div>

**Key Principles:**

<div class="space-y-3 mt-3">

<div class="p-3 bg-blue-50 rounded">
  <strong>1.</strong> Start simple: 1 layer, few units, ReLU + Adam
</div>

<div class="p-3 bg-green-50 rounded">
  <strong>2.</strong> Monitor curves: diagnose issues early
</div>

<div class="p-3 bg-purple-50 rounded">
  <strong>3.</strong> Iterate: adjust based on performance
</div>

<div class="p-3 bg-orange-50 rounded">
  <strong>4.</strong> Understand fundamentals: debug better
</div>

</div>

</div>

</div>

---
layout: two-cols
---

# Learning Outcomes

What You Can Now Do

**MPS311 Students:**

<div class="mt-4 space-y-2">

<div class="flex items-start">
  <span class="text-green-600 mr-2">✅</span>
  <span>Build neural networks with Keras</span>
</div>

<div class="flex items-start">
  <span class="text-green-600 mr-2">✅</span>
  <span>Choose architecture (layers, units, activations)</span>
</div>

<div class="flex items-start">
  <span class="text-green-600 mr-2">✅</span>
  <span>Select loss functions and optimizers</span>
</div>

<div class="flex items-start">
  <span class="text-green-600 mr-2">✅</span>
  <span>Train and evaluate networks</span>
</div>

<div class="flex items-start">
  <span class="text-green-600 mr-2">✅</span>
  <span>Interpret training curves</span>
</div>

<div class="flex items-start">
  <span class="text-green-600 mr-2">✅</span>
  <span>Diagnose common issues</span>
</div>

<div class="flex items-start">
  <span class="text-green-600 mr-2">✅</span>
  <span>Apply to real problems</span>
</div>

</div>

<div class="mt-4 p-3 bg-blue-50 rounded font-mono text-sm">
  Normalize → Build → Train →<br/> Monitor → Adjust → Deploy
</div>

::right::

<div class="ml-6">

**MPS439 Students:**

<div class="mt-4 space-y-2">

<div class="p-2 bg-purple-50 rounded">
  <strong>All MPS311 skills, PLUS:</strong>
</div>

<div class="flex items-start">
  <span class="text-purple-600 mr-2">✅</span>
  <span>Understand backpropagation algorithm</span>
</div>

<div class="flex items-start">
  <span class="text-purple-600 mr-2">✅</span>
  <span>Derive gradients using chain rule</span>
</div>

<div class="flex items-start">
  <span class="text-purple-600 mr-2">✅</span>
  <span>Implement from scratch</span>
</div>

<div class="flex items-start">
  <span class="text-purple-600 mr-2">✅</span>
  <span>Apply regularization (L1/L2, dropout)</span>
</div>

<div class="flex items-start">
  <span class="text-purple-600 mr-2">✅</span>
  <span>Use batch normalization</span>
</div>

<div class="flex items-start">
  <span class="text-purple-600 mr-2">✅</span>
  <span>Compare optimizers effectively</span>
</div>

<div class="flex items-start">
  <span class="text-purple-600 mr-2">✅</span>
  <span>Debug gradient issues</span>
</div>

<div class="flex items-start">
  <span class="text-purple-600 mr-2">✅</span>
  <span>Design custom architectures</span>
</div>

</div>

<div class="mt-4 p-3 bg-purple-50 rounded text-sm">
  <strong>Next level:</strong> Research implementation, novel designs
</div>

</div>

---
layout: center
class: text-center
---

# Resources & Next Steps

<div class="grid grid-cols-2 gap-8 mt-8 text-left">

<div>

**Today's Materials:**

<div class="space-y-2 mt-4">
  <div>📝 Lecture notes with full derivations</div>
  <div>💻 Interactive Colab notebooks</div>
  <div>🖼️ All figures and visualizations</div>
  <div>📚 Code examples (Keras + from-scratch)</div>
</div>

**Practice Recommendations:**

<div class="space-y-2 mt-4 text-sm">
  <div>1. Work through XOR example yourself</div>
  <div>2. Try on real dataset (Iris, MNIST)</div>
  <div>3. Experiment with architectures</div>
  <div>4. Read Sections 10-11 (MPS439)</div>
</div>

</div>

<div>

<div class="p-6 bg-orange-50 rounded-lg border-2 border-orange-300">
  <div class="text-lg text-center mb-4">
    <strong>Key Reminder:</strong>
  </div>
  <div class="text-center">
    "Neural networks are powerful,<br/>but not magic.<br/><br/>
    They're compositions of simple<br/>operations we understand.<br/><br/>
    <span class="text-xl font-bold text-orange-700">Master the fundamentals,<br/>then scale up!</span>
  </div>
</div>

</div>

</div>

<div class="mt-12 text-3xl">
  Questions?
</div>

<!-- COURSE_FEEDBACK_QR:START -->
---
layout: center
class: text-center
---

# 30-second feedback

<p class="text-3xl mb-4">What would help you learn better next time?</p>

<p class="text-2xl mb-4">Scan to share anonymous feedback on today's lecture.</p>

<img src="./feedback-qr.svg" alt="Feedback QR code for Lesson 09 lecture" class="w-44 mx-auto rounded-lg shadow" />

<p class="text-sm mt-4 opacity-70">MPS311/439 · 2026-27 · Lesson 09 lecture</p>
<!-- COURSE_FEEDBACK_QR:END -->
