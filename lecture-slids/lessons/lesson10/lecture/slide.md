---
theme: default
background: https://cover.sli.dev
class: text-center
highlighter: shiki
lineNumbers: false
info: |
  ## Lesson 10: Convolutional Neural Networks
  MPS311/439 Machine Learning - Dr. Wei Xing
drawings:
  persist: false
transition: slide-left
title: 'Lesson 10: Convolutional Neural Networks'
routerMode: hash
mdc: true
---

# Lesson 10: Convolutional Neural Networks

## Learning Spatial Patterns in Images

<div class="pt-12">
  <span class="text-xl">
    MPS311/439 Machine Learning<br>
    Dr. Wei Xing<br>
    2026–27
  </span>
</div>

---

# Where We Left Off

<div class="grid grid-cols-2 gap-8">

<div>

## Last Week: Neural Networks

- Built fully-connected neural networks
- Stacked layers: Input → Hidden → Output
- Key insight: Multiple layers of transformations

<div class="mt-4">

$$y = \text{ReLU}(W_2 \cdot \text{ReLU}(W_1 \cdot x))$$

</div>

- Universal approximation with depth

</div>

<div>

## Today: Specialized for Images

- **CNNs** = Convolutional Neural Networks
- Same building blocks (weighted sums, activations)
- But organized for **spatial data**
- Revolutionary impact on computer vision

<div class="mt-4 p-4 bg-blue-50 rounded">
We're still using weighted sums and activations - just arranged more intelligently!
</div>

</div>

</div>

---

# The Challenge: Parameter Explosion

<div class="text-center">

<!-- ## The Problem: Too Many Parameters! -->

<div class="mt-2">

**MNIST digit:** 28×28 = 784 pixels -> First hidden layer (100 neurons): 784 × 100 = **78,400 parameters**

**Color image:** 224×224×3 = 150,528 pixels -> With 100 neurons: **15,052,800 parameters!** 😱

</div>

<img src="/figures/fig1_parameter_explosion.png" class="mx-auto mt-0" style="width: 65%;">

<div class="mt-1 p-1 bg-red-50 rounded inline-block">
And this is just the FIRST layer! We haven't even started training...
</div>

</div>

---

# Why Does This Fail?

<div class="grid grid-cols-2 gap-8">

<div>

## What We're Doing Wrong

<div class="mt-4">

❌ Flattening images into vectors

❌ Treating pixels as independent

❌ Connecting **every pixel** to **every neuron**

❌ Ignoring spatial relationships

</div>

<div class="mt-8 text-center">
<img src="/figures/fig2_convolution_operation.png" style="width: 100%; opacity: 0.5;">
</div>

</div>

<div>

## What We're Missing

<div class="mt-4">

✅ Images have **spatial structure**

✅ Nearby pixels are related

✅ Patterns **repeat** across locations

✅ Edges, textures appear everywhere

</div>

<div class="mt-8 p-4 bg-yellow-50 rounded">
<strong>Key Insight:</strong><br>
To detect an edge, we only need neighboring pixels, not pixels from across the image!
</div>

</div>

</div>

---

# The Key Insight

<div class="text-left">

## What if we only looked at **local regions**?

<div class="grid grid-cols-2 gap-12 mt-8">

<div>

### 1. Local Receptive Fields

- Each neuron connects to a **small region**
- Example: 3×3 or 5×5 neighborhood
- Not the entire image!

</div>

<div>

### 2. Parameter Sharing

- Use **same weights** across entire image
- Same edge detector works everywhere
- Dramatic parameter reduction

</div>

</div>

<div class="mt-12 p-6 bg-green-50 rounded text-lg">
<strong>Analogy:</strong> Like using the same 'pattern detector' to scan the entire image, position by position
</div>

</div>

---

# Solution Preview: Convolutional Neural Networks

<div>

<!-- ## The Solution: Convolutional Neural Networks (CNNs) -->

<div class="mt-2">

**Core Concept:**
- Small **filter** (or **kernel**) slides across image
- At each position: compute weighted sum
- Still our familiar operation, but applied **locally** and **repeatedly**

</div>

<div class="mt-3 p-1 bg-blue-50 rounded">

**Mathematical Connection:**

Same operation as before, organized differently:

$$y_{ij} = \sum_{m}\sum_{n} x_{i+m,j+n} \cdot w_{mn} + b$$

Just a weighted sum, but local!

</div>

<div class="mt-3 grid grid-cols-3 gap-4 text-center">

<div class="p-1 bg-green-100 rounded">
✓ Dramatically fewer parameters
</div>

<div class="p-1 bg-green-100 rounded">
✓ Preserves spatial structure
</div>

<div class="p-1 bg-green-100 rounded">
✓ Translation invariant
</div>

</div>

</div>

---

# Understanding Convolutions

<div class="text-center">

<!-- ## How Convolution Works: Sliding Window -->

<img src="/figures/fig2_convolution_operation.png" class="mx-auto mt-0" style="width: 85%;">

<div class="mt-2 grid grid-cols-5 gap-3 text-sm">

<div class="p-2 bg-blue-50 rounded">
1. Place 3×3 filter on image
</div>

<div class="p-2 bg-blue-50 rounded">
2. Multiply element-wise
</div>

<div class="p-2 bg-blue-50 rounded">
3. Sum all products
</div>

<div class="p-2 bg-blue-50 rounded">
4. Slide to next position
</div>

<div class="p-2 bg-blue-50 rounded">
5. Repeat across image
</div>

</div>

<div class="mt-4 text-base">
Each output value captures a local pattern at that position
</div>

</div>

---

# Filter Examples: Different Patterns

<div class="text-center">

<!-- ## Different Filters Detect Different Patterns -->

<img src="/figures/fig3_filter_examples.png" class="mx-auto mt-0" style="width: 65%;">

<div class="mt-0 grid grid-cols-3 gap-4">

<div class="p-0 bg-blue-50 text-sm rounded">

Vertical Edge:

$$\begin{bmatrix}-1 & 0 & 1\\-1 & 0 & 1\\-1 & 0 & 1\end{bmatrix}$$

</div>

<div class="p-0 bg-green-50 text-sm rounded">

Horizontal Edge:

$$\begin{bmatrix}-1 & -1 & -1\\0 & 0 & 0\\1 & 1 & 1\end{bmatrix}$$

</div>

<div class="p-0 bg-purple-50 text-sm rounded">

Blur/Smooth:

$$\frac{1}{9}\begin{bmatrix}1 & 1 & 1\\1 & 1 & 1\\1 & 1 & 1\end{bmatrix}$$

</div>

</div>

<div class="mt-1 p-0 bg-yellow-100 rounded text-lg font-bold">
The Magic: In CNNs, we LEARN these filter values through training!
</div>

</div>

---

# Convolutional Layers: Multiple Filters

<div class="text-left">

<!-- ## Convolutional Layer = Multiple Filters -->

<img src="/figures/fig4_multiple_filters.png" class="mx-auto mt-4" style="width: 52%;">

<div class="mt-0 grid text-left grid-cols-3 gap-6">

<div class="p-0 bg-red-50 text-base rounded">

**Key Concepts:**
- One filter → one **feature map**
- Use 32, 64, or 128 filters
- Each detects different patterns

</div>

<div class="p-0 bg-green-50 text-base rounded">

**Dimension Example:**

Input: 28×28×**1**
Apply 32 filters (3×3)
Output: 28×28×**32**

</div>

<div class="p-0 bg-purple-50 text-base rounded">

**What We Get:**
- 32 different views
- Highlighting different features
- Stack of feature maps

</div>

</div>

</div>

---

# Parameter Efficiency: The Big Win

<div>

## Dramatic Parameter Reduction

<div class="grid grid-cols-2 gap-8 mt-8">

<div class="p-6 bg-red-50 rounded">

### Fully-Connected Approach

<div class="mt-4">

28×28 image → 100 neurons

Parameters: 784 × 100

</div>

<div class="text-4xl font-bold text-red-600 mt-4">
= 78,400
</div>

</div>

<div class="p-6 bg-green-50 rounded">

### Convolutional Approach

<div class="mt-4">

32 filters, 3×3 size

Parameters: (3×3×1 + 1) × 32

</div>

<div class="text-4xl font-bold text-green-600 mt-4">
= 320
</div>

</div>

</div>

<div class="mt-8 text-center p-6 bg-yellow-100 rounded text-2xl font-bold">
~250× fewer parameters with BETTER performance!
</div>

<div class="mt-6 text-center text-lg">

**Why it works:**
Parameter sharing • Local connections • Learns reusable patterns

</div>

</div>

---

# Pooling Layers: Reduce Dimensions Gradually

<div>

<!-- ## Pooling -->


<img src="/figures/fig5_pooling_operation.png" style="width: 80%;">



<div class="grid grid-cols-3 gap-6 mt-4">

  <div class="p-4 bg-blue-50 rounded text-center">
    <strong>Reduce Spatial Dimensions</strong>
    <ul class="text-left mt-2 list-disc ml-4">
      <li>Computational efficiency</li>
      <li>More manageable size</li>
    </ul>
  </div>

  <div class="p-4 bg-green-50 rounded text-center">
    <strong>Translation Invariance</strong>
    <ul class="text-left mt-2 list-disc ml-4">
      <li>Small shifts don't matter much</li>
      <li>"Cat is still a cat"</li>
    </ul>
  </div>

  <div class="p-4 bg-yellow-50 rounded text-center">
    <strong>No Learnable Parameters</strong>
    <ul class="text-left mt-2 list-disc ml-4">
      <li>Just a fixed operation</li>
      <li>Max or Average pooling</li>
    </ul>
  </div>

</div>

<div>

<!-- <img src="/figures/fig5_pooling_operation.png" style="width: 100%;"> -->

<!-- <div class="mt-4 p-3 bg-blue-50 rounded text-center">
Common: 2×2 max pooling, stride 2<br>
Reduces by half: 28×28 → 14×14
</div> -->

</div>

</div>


---

# Complete CNN Architecture

<div class="text-center">

<!-- ## Putting It All Together: CNN for MNIST -->

<img src="/figures/fig6_cnn_architecture.png" class="mx-auto mt-0" style="width: 65%; ">

<div class="mt--10 p-0 bg-blue-50 rounded text-xl">

**The Pattern:**: Input → \[**Conv** → **ReLU** → **Pool**\] × N → **Flatten** → Dense → Output

</div>

<div class="mt--3 grid grid-cols-3 gap-4 text-sm">

<div class="p-0 bg-green-100 rounded">
Total: ~421K parameters
</div>

<div class="p-0 bg-red-100 rounded">
FC network: ~1.3M parameters
</div>

<div class="p-0 bg-yellow-100 rounded">
68% reduction!
</div>

</div>

</div>

---

# Implementation Part 1: Data Preparation

<div>

## Implementing CNNs in Keras: Step 1

```python
from tensorflow import keras
import numpy as np

# Load MNIST
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# Normalize to [0, 1]
X_train = X_train / 255.0
X_test = X_test / 255.0

# CRITICAL: Add channel dimension
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

print(f"New shape: {X_train.shape}")  # (60000, 28, 28, 1)
```

<div class="mt-6 grid grid-cols-2 gap-6">

<div class="p-4 bg-red-100 rounded">
<strong>⚠️ CRITICAL:</strong> CNNs expect<br>
(height, width, channels)
</div>

<div class="p-4 bg-blue-100 rounded">
Grayscale = 1 channel<br>
RGB = 3 channels
</div>

</div>

</div>

---

# Implementation Part 2: Building the Model

<div>

## Step 2: Build the CNN Architecture

```python
from tensorflow.keras import layers, models

model = models.Sequential([
    # First conv block
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    
    # Second conv block
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    
    # Classification head
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])
```

<div class="mt-4 grid grid-cols-4 gap-3 text-sm">

<div class="p-2 bg-blue-50 rounded">
<code>Conv2D(32, (3,3))</code><br>32 filters, 3×3
</div>

<div class="p-2 bg-green-50 rounded">
<code>MaxPooling2D((2,2))</code><br>2×2 pooling
</div>

<div class="p-2 bg-purple-50 rounded">
<code>Flatten()</code><br>3D → 1D
</div>

<div class="p-2 bg-orange-50 rounded">
<code>Dense(10)</code><br>10 classes
</div>

</div>

</div>

---

# Implementation Part 3: Train & Evaluate

<div class="grid grid-cols-2 gap-6">

<div>

## Step 3: Compile & Train

```python
# Compile
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train
history = model.fit(
    X_train, y_train,
    epochs=5,
    validation_split=0.2
)

# Evaluate
test_loss, test_acc = model.evaluate(
    X_test, y_test
)
print(f'Test accuracy: {test_acc:.4f}')
```

</div>

<div>

<img src="/figures/fig7_training_curves.png" style="width: 100%;">

<div class="mt-4 p-4 bg-green-100 rounded text-center text-xl font-bold">
~99% accuracy on MNIST!
</div>

<div class="mt-3 text-sm text-center">
From 0 to 99% in just 5 epochs
</div>

</div>

</div>

---

# What Do CNNs Learn?

<div class="text-center">

<!-- ## Hierarchical Feature Learning -->

<img src="/figures/fig8_hierarchical_features.png" class="mx-auto mt-4" style="width: 65%;">

<div class="mt-0 grid grid-cols-3 gap-4">

<div class="p-0 bg-blue-50 text-sm rounded">
<strong>Early Layers</strong><br>
Conv1, Conv2<br>
Simple edges, colors, orientations
</div>

<div class="p-0 bg-purple-50 text-sm rounded">
<strong>Middle Layers</strong><br>
Conv3, Conv4<br>
Textures, patterns, shapes
</div>

<div class="p-0 bg-orange-50 text-sm rounded">
<strong>Deep Layers</strong><br>
Conv5, Conv6<br>
Complex objects, parts
</div>

</div>

<div class="mt-1 p-0 bg-yellow-100 rounded text-base font-bold">
This hierarchy emerges AUTOMATICALLY from training!<br>
We never told it to learn edges first.
</div>

</div>

---

# Practical Tips

<div>

<!-- ## Practical Considerations -->

<div class="grid grid-cols-3 gap-6 mt-6">

<div>

### When to Use CNNs

<div class="mt-3 p-1 bg-blue-50 rounded space-y-2">

✅ Image data

✅ Spatial structure matters

✅ Patterns repeat across locations

<div class="mt-4"></div>

❌ Tabular data

❌ No spatial relationships

❌ Features have specific positions

</div>

</div>

<div>

### Fighting Overfitting

<div class="mt-3 p-1 bg-green-50 rounded space-y-2">

**Data augmentation**
- Rotate, flip, zoom

**Dropout layers**
- 0.25-0.5 dropout rate

**Early stopping**
- Monitor validation loss

**More data**
- Always helps!

</div>

</div>

<div>

### Computational Tips

<div class="mt-3 p-1 bg-yellow-50 rounded space-y-2">

**Use GPU**
- Google Colab (free!)
- 10-100× speedup

**Start simple**
- Small models first
- Scale up if needed

**Common choices**
- 3×3 filters most common
- Double filters after pooling
- 32 → 64 → 128

</div>

</div>

</div>

</div>

---

# Explore CNNs Interactively!

<div class="grid grid-cols-2 gap-8">

<div>

## 🎮 Try It Yourself!

<div class="mt-2 space-y-4">

<div class="p-1 bg-blue-50 rounded">

### CNN Explainer
**Interactive visualization of how CNNs work**

🔗 [**poloclub.github.io/cnn-explainer/**](https://poloclub.github.io/cnn-explainer/)

- See convolutions in real-time
- Explore each layer's output
- Understand the full pipeline
- Works in your browser!

</div>

<div class="p-1 bg-green-50 rounded">

### TensorFlow Playground
**Experiment with neural networks**

🔗 [**playground.tensorflow.org**](https://playground.tensorflow.org/)

</div>

<div class="p-1 bg-purple-50 rounded">

### Distill.pub
**Beautiful ML explanations**

🔗 **distill.pub**

</div>

</div>

</div>

<div class="text-center">

<div class="mt-8">

### Scan to Explore CNN Explainer

<div class="mt-2 flex justify-center">
  <img src="./qrcode_poloclub.github.io.png" style="width: 50%; margin-bottom: -30px;">
</div>

<div class="mt-4 text-lg font-bold text-blue-600">

[**poloclub.github.io/cnn-explainer/**](https://poloclub.github.io/cnn-explainer/)

</div>

<div class="mt-6 p-1 bg-yellow-100 rounded">
<strong>📱 Pull out your phone!</strong><br>
Spend 5-10 minutes exploring<br>
See everything we discussed in action
</div>

</div>

</div>

</div>

---

# Summary: What We Learned Today

<div>

## Key Takeaways from Lesson 10

<div class="mt-0 space-y-4">

<div class="p-0 bg-blue-50 rounded">

**1. Spatial Structure Matters**
<div class="grid grid-cols-3 gap-3 text-center text-base mt-3">
  <div class="p-2 bg-blue-100 rounded">Images aren't just flat vectors</div>
  <div class="p-2 bg-blue-100 rounded">Nearby pixels are related</div>
  <div class="p-2 bg-blue-100 rounded">Patterns repeat across locations</div>
</div>

</div>

<div class="p-0 bg-green-50 rounded">

**2. Convolution Operation**

Local weighted sum at each position:

$$y_{ij} = \sum_{m,n} x_{i+m,j+n} \cdot w_{mn} + b$$

Same operation as before, applied everywhere

</div>

<div class="p-0 bg-purple-50 rounded">

**3. Parameter Sharing** — 320 parameters vs 78,400 for same capacity — Dramatic efficiency gain

</div>

<div class="grid grid-cols-2 gap-4">

<div class="p-0 bg-orange-50 rounded">
<strong>4. Pooling</strong><br>
Dimension reduction • Translation invariance
</div>

<div class="p-0 bg-yellow-50 rounded">
<strong>5. Hierarchical Learning</strong><br>
Simple → Complex automatically
</div>

</div>

</div>

</div>

---

# What You Can Do Now

<div>

<!-- ## Your New Skills - What You Can Do -->

<div class="mt-1 space-y-3">

<div class="p-1 bg-green-50 rounded">
✅ <strong>Understand</strong> how convolutions work (weighted sums on local regions)
</div>

<div class="p-1 bg-green-50 rounded">
✅ <strong>Explain</strong> why CNNs are more efficient than fully-connected networks
</div>

<div class="p-1 bg-green-50 rounded">
✅ <strong>Build</strong> CNNs using Keras for image classification tasks
</div>

<div class="p-1 bg-green-50 rounded">
✅ <strong>Choose</strong> appropriate architecture parameters (number of filters: 32, 64, 128 • filter sizes: 3×3 most common • when to add pooling)
</div>

<div class="p-1 bg-green-50 rounded">
✅ <strong>Train and evaluate</strong> CNN models on your own image datasets
</div>

<div class="p-1 bg-green-50 rounded">
✅ <strong>Apply</strong> to real problems: MNIST, Fashion-MNIST, CIFAR-10
</div>

</div>

<div class="mt-2 p-1 bg-blue-100 rounded">

<strong>For MPS439 Students (Additional):</strong>
Understand backpropagation through conv layers • Use pre-trained models (VGG-16, ResNet) • Transfer learning

</div>

<div class="mt-4 text-center text-lg">
<strong>Next Steps:</strong> Practice with provided code • Try different architectures • Experiment with your own datasets!
</div>

</div>

<!-- COURSE_FEEDBACK_QR:START -->
---
layout: center
class: text-center
---

# 30-second feedback

<p class="text-xl mb-4">Scan this code to share anonymous feedback or post a question for this lecture.</p>

<img src="./feedback-qr.svg" alt="Feedback QR code for Lesson 10 lecture" class="w-44 mx-auto rounded-lg shadow" />

<p class="text-sm mt-4 opacity-70">MPS311/439 · 2026-27 · Lesson 10 lecture</p>
<!-- COURSE_FEEDBACK_QR:END -->
