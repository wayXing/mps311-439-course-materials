---
pagetitle: "Lab 9: Neural Networks - Classification and Regression"
---

# Lab 9: Neural Networks - Classification and Regression
## MPS311/439 - Machine Learning
**Lesson 9 Lab Session | Duration: 50 minutes**

---

## Introduction

Welcome to Lab 9! Today you'll explore **Neural Networks** - powerful models that can learn complex, non-linear patterns by stacking simple operations in layers.

**What you'll learn today:**
- How to build neural networks using Keras for classification and regression
- How architecture choices (layers, units) affect performance
- How different activation functions compare
- How to monitor training and detect overfitting
- **How neural networks compare to previous methods you've learned**

**Important comparisons:**
- Check your previous lab records for **Logistic Regression accuracy on Digits**
- Check your **Assignment 3 results for California Housing** (Ridge Regression)
- Today you'll see if neural networks can beat those results!

**Remember:** This lab uses a fill-in-the-blanks approach. Don't write code from scratch - just fill in the blanks marked with `____`. Focus on understanding the concepts!

---

## Setup: Import Libraries and Load Data

We'll use two datasets today:
1. **Digits dataset** (Classification): 1797 images of handwritten digits (0-9), each 8×8 pixels
2. **California Housing dataset** (Regression): Housing prices based on location/demographics

**Copy and run this code:**

```python
# Import packages
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam, SGD
from keras.regularizers import l2
import warnings
warnings.filterwarnings('ignore')

# Load Digits dataset
digits = load_digits()
X_digits = digits.data
y_digits = digits.target

print("="*60)
print("DIGITS DATASET (Classification)")
print("="*60)
print(f"Samples: {X_digits.shape[0]}")
print(f"Features: {X_digits.shape[1]} (8x8 pixel images)")
print(f"Classes: {len(np.unique(y_digits))} (digits 0-9)")

# Visualize some digits
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap='gray')
    ax.set_title(f'Label: {y_digits[i]}')
    ax.axis('off')
plt.tight_layout()
plt.show()

# Split and standardize
X_train, X_test, y_train, y_test = train_test_split(
    X_digits, y_digits, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"\nTraining samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")
print("Features standardized!")
```

**What's happening here?**
- We loaded 1797 handwritten digit images (8×8 = 64 features each)
- We split into train (80%) and test (20%) sets
- We standardized features (important for neural networks!)
- This is the same Digits dataset you used with Logistic Regression before

---

## Part 1: Your First Neural Network for Classification (8 minutes)

**Background:** A neural network for multi-class classification needs:
- Input layer: 64 features (our pixel values)
- Hidden layer(s): Learn patterns with ReLU activation
- Output layer: 10 units (one per digit) with softmax activation

Let's build a simple 1-hidden-layer network!

### Task 1.1: Build the network

**Fill in the blanks below:**

```python
# Create a simple neural network
model = Sequential([
    Dense(____, input_dim=64, activation='____'),  # Hidden layer
    Dense(____, activation='softmax')              # Output layer
])

print("Model created!")
model.summary()
```

**Hints:**
- Hidden layer: try `16` units with `'relu'` activation
- Output layer: `10` units (one per digit class) with `'softmax'`

**AI Help:** Ask ChatGPT: "How do I build a neural network for multi-class classification with Keras?"

### Task 1.2: Compile the model

**Fill in the blanks:**

```python
# Compile the model
model.compile(
    optimizer='____',
    loss='____',
    metrics=['accuracy']
)

print("Model compiled!")
```

**Hints:**
- Optimizer: use `'adam'` (adaptive learning rate, works well)
- Loss: use `'sparse_categorical_crossentropy'` (for integer labels)

**Note:** We use `sparse_categorical_crossentropy` because our labels are integers (0, 1, 2, ..., 9), not one-hot encoded.

### Task 1.3: Train the model

**Fill in the blanks:**

```python
# Train the model
history = model.fit(
    X_train, y_train,
    epochs=____,
    batch_size=32,
    verbose=1
)

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
```

**Hint:**
- Use `epochs=50` (how many times to go through all training data)

### Task 1.4: Compare with your previous results

**Questions:**

1. What accuracy did you get with Logistic Regression on Digits in your previous lab?
```
Logistic Regression accuracy: _____%
```

2. What accuracy did the neural network achieve?
```
Neural Network accuracy: _____%
```

3. Did the neural network improve over Logistic Regression?
```
Answer: _______________________________________________
```

**Key insight:** Even a simple 1-layer neural network can often match or beat logistic regression!

---

## Part 2: Experiment with Architecture (10 minutes)

**Background:** The architecture (number of layers, units per layer) dramatically affects the network's capacity to learn. Let's experiment!

### Task 2.1: Try different numbers of hidden units

**Fill in the blanks and run each variation:**

**Variation A: Small network (8 units)**
```python
model_small = Sequential([
    Dense(____, input_dim=64, activation='relu'),
    Dense(10, activation='softmax')
])
model_small.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history_small = model_small.fit(X_train, y_train, epochs=50, verbose=0)
_, acc_small = model_small.evaluate(X_test, y_test, verbose=0)
print(f"Small network (8 units): {acc_small:.4f}")
```

**Variation B: Medium network (32 units)**
```python
model_medium = Sequential([
    Dense(____, input_dim=64, activation='relu'),
    Dense(10, activation='softmax')
])
model_medium.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history_medium = model_medium.fit(X_train, y_train, epochs=50, verbose=0)
_, acc_medium = model_medium.evaluate(X_test, y_test, verbose=0)
print(f"Medium network (32 units): {acc_medium:.4f}")
```

**Variation C: Large network (64 units)**
```python
model_large = Sequential([
    Dense(____, input_dim=64, activation='relu'),
    Dense(10, activation='softmax')
])
model_large.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history_large = model_large.fit(X_train, y_train, epochs=50, verbose=0)
_, acc_large = model_large.evaluate(X_test, y_test, verbose=0)
print(f"Large network (64 units): {acc_large:.4f}")
```

**Hints:**
- Fill in `8`, `32`, and `64` respectively

### Task 2.2: Try adding a second hidden layer

**Fill in the blanks:**

```python
# Two-layer network
model_deep = Sequential([
    Dense(32, input_dim=64, activation='relu'),  # First hidden layer
    Dense(____, activation='____'),               # Second hidden layer
    Dense(10, activation='softmax')              # Output layer
])
model_deep.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history_deep = model_deep.fit(X_train, y_train, epochs=50, verbose=0)
_, acc_deep = model_deep.evaluate(X_test, y_test, verbose=0)
print(f"Deep network (32→16): {acc_deep:.4f}")
```

**Hints:**
- Second hidden layer: use `16` units with `'relu'` activation

### Task 2.3: Compare all architectures

**Fill in your results:**

```python
print("\n" + "="*60)
print("ARCHITECTURE COMPARISON")
print("="*60)
print(f"Small (8 units):        {acc_small:.4f}")
print(f"Medium (32 units):      {acc_medium:.4f}")
print(f"Large (64 units):       {acc_large:.4f}")
print(f"Deep (32→16 units):     {acc_deep:.4f}")
print("="*60)
```

**Questions:**

1. Which architecture performed best?
```
Best architecture: _______________________________________________
```

2. Did adding more units always improve accuracy?
```
Answer: _______________________________________________
```

3. Did the two-layer network beat the one-layer networks?
```
Answer: _______________________________________________
```

**Key insight:** More capacity (units/layers) can help, but there's diminishing returns. Start simple!

---

## Part 3: Activation Functions Matter (10 minutes)

**Background:** The activation function in hidden layers provides non-linearity. Let's compare ReLU, sigmoid, and tanh.

### Task 3.1: Train with different activations

**Fill in the blanks for each activation:**

**ReLU (what we've been using)**
```python
model_relu = Sequential([
    Dense(32, input_dim=64, activation='____'),
    Dense(10, activation='softmax')
])
model_relu.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history_relu = model_relu.fit(X_train, y_train, epochs=50, verbose=0)
_, acc_relu = model_relu.evaluate(X_test, y_test, verbose=0)
print(f"ReLU activation: {acc_relu:.4f}")
```

**Sigmoid**
```python
model_sigmoid = Sequential([
    Dense(32, input_dim=64, activation='____'),
    Dense(10, activation='softmax')
])
model_sigmoid.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history_sigmoid = model_sigmoid.fit(X_train, y_train, epochs=50, verbose=0)
_, acc_sigmoid = model_sigmoid.evaluate(X_test, y_test, verbose=0)
print(f"Sigmoid activation: {acc_sigmoid:.4f}")
```

**Tanh**
```python
model_tanh = Sequential([
    Dense(32, input_dim=64, activation='____'),
    Dense(10, activation='softmax')
])
model_tanh.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history_tanh = model_tanh.fit(X_train, y_train, epochs=50, verbose=0)
_, acc_tanh = model_tanh.evaluate(X_test, y_test, verbose=0)
print(f"Tanh activation: {acc_tanh:.4f}")
```

**Hints:**
- Use `'relu'`, `'sigmoid'`, and `'tanh'` respectively

**AI Help:** Ask ChatGPT: "What are the differences between ReLU, sigmoid, and tanh activation functions?"

### Task 3.2: Compare training curves

**Copy and run this code:**

```python
# Plot training curves
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history_relu.history['loss'], label='ReLU')
plt.plot(history_sigmoid.history['loss'], label='Sigmoid')
plt.plot(history_tanh.history['loss'], label='Tanh')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Comparison')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history_relu.history['accuracy'], label='ReLU')
plt.plot(history_sigmoid.history['accuracy'], label='Sigmoid')
plt.plot(history_tanh.history['accuracy'], label='Tanh')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy Comparison')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
```

### Task 3.3: Analyze results

**Fill in the comparison:**

```python
print("\n" + "="*60)
print("ACTIVATION FUNCTION COMPARISON")
print("="*60)
print(f"ReLU:    {acc_relu:.4f}")
print(f"Sigmoid: {acc_sigmoid:.4f}")
print(f"Tanh:    {acc_tanh:.4f}")
print("="*60)
```

**Questions:**

1. Which activation function performed best?
```
Best activation: _______________________________________________
```

2. Looking at the training curves, which activation learned fastest?
```
Answer: _______________________________________________
```

3. Why is ReLU the default choice for hidden layers?
```
Answer: _______________________________________________
___________________________________________________________
```

**Key insight:** ReLU usually trains faster and works well. Sigmoid can suffer from vanishing gradients.

---

## Part 4: Training Dynamics & Overfitting (10 minutes)

**Background:** Monitoring training and validation loss helps detect overfitting. When validation loss increases while training loss decreases, you're overfitting!

### Task 4.1: Train with validation split

**Fill in the blanks:**

```python
# Build a model
model_monitored = Sequential([
    Dense(64, input_dim=64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(10, activation='softmax')
])
model_monitored.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train with validation split
history_monitored = model_monitored.fit(
    X_train, y_train,
    epochs=____,
    batch_size=32,
    validation_split=____,
    verbose=0
)

print("Training complete!")
```

**Hints:**
- Use `epochs=100` (more epochs to see overfitting)
- Use `validation_split=0.2` (20% of training data for validation)

**AI Help:** Ask ChatGPT: "What is validation_split in Keras and why is it important?"

### Task 4.2: Plot training dynamics

**Fill in the blanks:**

```python
# Plot training and validation curves
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history_monitored.history['____'], label='Training Loss')
plt.plot(history_monitored.history['____'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Dynamics - Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history_monitored.history['accuracy'], label='Training Accuracy')
plt.plot(history_monitored.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Dynamics - Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
```

**Hints:**
- Training loss: `'loss'`
- Validation loss: `'val_loss'`

### Task 4.3: Detect overfitting

**Questions:**

1. At approximately which epoch does overfitting begin (validation loss starts increasing)?
```
Overfitting begins around epoch: ____
```

2. What is the gap between final training and validation accuracy?
```
Gap: _____%
```

3. If you saw this pattern in a real project, what would you do?
```
Answer: _______________________________________________
___________________________________________________________
```

**Key insight:** Always use validation data! Stop training when validation loss increases (early stopping).

---

## Part 5: Neural Network for Regression (6 minutes)

**Background:** Neural networks aren't just for classification! Let's use them for regression on the California Housing dataset.

### Task 5.1: Load and prepare housing data

**Copy and run this code:**

```python
print("\n" + "="*60)
print("CALIFORNIA HOUSING DATASET (Regression)")
print("="*60)

# Load data
housing = fetch_california_housing()
X_housing = housing.data
y_housing = housing.target

print(f"Samples: {X_housing.shape[0]}")
print(f"Features: {X_housing.shape[1]}")
print(f"Target: House prices (in $100,000s)")

# Split and standardize
X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(
    X_housing, y_housing, test_size=0.2, random_state=42
)

scaler_h = StandardScaler()
X_train_h = scaler_h.fit_transform(X_train_h)
X_test_h = scaler_h.transform(X_test_h)

print(f"\nTraining samples: {X_train_h.shape[0]}")
print(f"Test samples: {X_test_h.shape[0]}")
```

### Task 5.2: Build a regression network

**Fill in the blanks:**

```python
# Neural network for regression
model_regression = Sequential([
    Dense(64, input_dim=8, activation='relu'),
    Dense(32, activation='relu'),
    Dense(____, activation='____')  # Output layer for regression
])

model_regression.compile(
    optimizer='adam',
    loss='____',  # Loss for regression
    metrics=['mae']
)

print("Regression model created!")
model_regression.summary()
```

**Hints:**
- Output layer: `1` unit (single price prediction) with `'linear'` activation (or `None`)
- Loss: `'mse'` (mean squared error for regression)

**AI Help:** Ask ChatGPT: "How do I build a neural network for regression in Keras?"

### Task 5.3: Train and evaluate

**Fill in the blanks:**

```python
# Train
history_regression = model_regression.fit(
    X_train_h, y_train_h,
    epochs=50,
    validation_split=0.2,
    verbose=____
)

# Evaluate
from sklearn.metrics import r2_score, mean_squared_error

y_pred = model_regression.predict(X_test_h)
r2 = r2_score(y_test_h, y_pred)
mse = mean_squared_error(y_test_h, y_pred)
rmse = np.sqrt(mse)

print(f"\nNeural Network Results:")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")
```

**Hint:**
- Use `verbose=0` for clean output

### Task 5.4: Compare with your Assignment 3 results

**Questions:**

1. What R² score did you get with Ridge Regression in Assignment 3?
```
Ridge Regression R²: ____
```

2. What R² score did the neural network achieve?
```
Neural Network R²: ____
```

3. Did the neural network improve over Ridge Regression?
```
Answer: _______________________________________________
```

4. Neural networks can do regression too! When would you prefer NN over Ridge Regression?
```
Answer: _______________________________________________
___________________________________________________________
```

**Key insight:** Neural networks work for both classification AND regression. They excel when relationships are highly non-linear!

---

## Reflection Questions (3 minutes)

Answer these questions based on what you've learned today:

**Question 1:** You've now used Logistic Regression and Neural Networks on Digits. What advantage does the neural network have?

```
Your answer: _______________________________________________
___________________________________________________________
```

**Question 2:** You tried networks with 8, 32, and 64 hidden units. If you had very limited training data (say, 100 samples), which would you choose and why?

```
Your answer: _______________________________________________
___________________________________________________________
```

**Question 3:** Why do we always standardize features before training neural networks?

```
Your answer: _______________________________________________
___________________________________________________________
```

**Question 4:** Looking at your training curves in Part 4, you saw overfitting happen. Name two things you could try to reduce overfitting (hint: think about what we'll cover in Part 6!).

```
Your answer: _______________________________________________
___________________________________________________________
```

---

## Summary

**What you learned today:**
- ✅ How to build neural networks for classification and regression using Keras
- ✅ How architecture (layers, units) affects model capacity and performance
- ✅ How activation functions (ReLU, sigmoid, tanh) compare
- ✅ How to monitor training dynamics and detect overfitting
- ✅ **How neural networks compare to previous methods** (Logistic/Ridge Regression)

**Key takeaways:**
- Neural networks can beat simpler methods on complex, non-linear problems
- Start simple (1 layer, moderate units, ReLU, Adam) and increase complexity if needed
- Always use validation data to monitor overfitting
- ReLU is the default choice for hidden layers
- Neural networks work for both classification and regression!

**Performance comparison:**
- Logistic Regression → Neural Network on Digits: Usually improves
- Ridge Regression → Neural Network on Housing: Comparable or better
- The gap widens on more complex problems!

---

## Part 6: Advanced Techniques - For MPS439 Students Only (30 minutes)

**Background:** You've built basic neural networks. Now let's explore advanced techniques to prevent overfitting and improve training: **L2 Regularization**, **Dropout**, and **Optimizer Comparison**.

These are the techniques from lecture Section 11 that make neural networks work in practice!

---

### Task 6.1: Understanding L2 Regularization

**Background:** L2 regularization (Ridge for neural networks) adds a penalty for large weights: 

$$\text{Loss}_{\text{total}} = \text{Loss}_{\text{data}} + \lambda \sum_{l} \|W^{(l)}\|^2$$

This encourages smaller, more distributed weights and reduces overfitting.

**Fill in the blanks:**

```python
# Build model WITHOUT regularization
model_no_reg = Sequential([
    Dense(64, input_dim=64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(10, activation='softmax')
])
model_no_reg.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Build model WITH L2 regularization
model_l2 = Sequential([
    Dense(64, input_dim=64, activation='relu', kernel_regularizer=l2(____)),
    Dense(32, activation='relu', kernel_regularizer=l2(____)),
    Dense(10, activation='softmax')
])
model_l2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("Models created!")
```

**Hints:**
- Use `l2(0.01)` for both hidden layers (lambda = 0.01)

**AI Help:** Ask ChatGPT: "How does L2 regularization work in neural networks?"

### Task 6.2: Train and compare regularization effect

**Fill in the blanks:**

```python
# Train both models
history_no_reg = model_no_reg.fit(
    X_train, y_train, 
    epochs=100, 
    validation_split=0.2, 
    verbose=0
)

history_l2 = model_l2.fit(
    X_train, y_train, 
    epochs=____, 
    validation_split=____, 
    verbose=0
)

# Plot comparison
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history_no_reg.history['val_loss'], label='No Regularization')
plt.plot(history_l2.history['____'], label='With L2 Regularization')
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.title('Effect of L2 Regularization')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history_no_reg.history['val_accuracy'], label='No Regularization')
plt.plot(history_l2.history['val_accuracy'], label='With L2 Regularization')
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.title('Validation Accuracy Comparison')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
```

**Hints:**
- Use same settings: `epochs=100`, `validation_split=0.2`
- Validation loss history: `'val_loss'`

### Task 6.3: Examine weight magnitudes

**Fill in the blanks:**

```python
# Extract weights from first hidden layer
weights_no_reg = model_no_reg.layers[____].get_weights()[0]
weights_l2 = model_l2.layers[0].get_weights()[0]

# Compute statistics
print("\n" + "="*60)
print("WEIGHT MAGNITUDE COMPARISON")
print("="*60)
print(f"No Regularization:")
print(f"  Max |weight|: {np.max(np.abs(weights_no_reg)):.4f}")
print(f"  Mean |weight|: {np.mean(np.abs(weights_no_reg)):.4f}")
print(f"  Sum(weight²): {np.sum(weights_no_reg**2):.4f}")

print(f"\nWith L2 Regularization:")
print(f"  Max |weight|: {np.max(np.abs(____)):.4f}")
print(f"  Mean |weight|: {np.mean(np.abs(____)):.4f}")
print(f"  Sum(weight²): {np.sum(____**2):.4f}")
print("="*60)
```

**Hints:**
- First layer index: `0`
- Use `weights_l2` for L2 regularized weights

**Questions:**

1. Are the weights smaller with L2 regularization?
```
Answer: _______________________________________________
```

2. Does L2 regularization help prevent overfitting (check the validation curves)?
```
Answer: _______________________________________________
```

3. Why is L2 called "weight decay"?
```
Answer: _______________________________________________
___________________________________________________________
```

---

### Task 6.4: Understanding Dropout

**Background:** Dropout randomly sets a fraction of neurons to zero during training. This forces the network to learn robust features that don't depend on any single neuron.

**Fill in the blanks:**

```python
# Build model WITH dropout
model_dropout = Sequential([
    Dense(64, input_dim=64, activation='relu'),
    Dropout(____),  # Drop 50% of neurons randomly
    Dense(32, activation='relu'),
    Dropout(____),  # Drop 30% of neurons randomly
    Dense(10, activation='softmax')
])
model_dropout.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train
history_dropout = model_dropout.fit(
    X_train, y_train,
    epochs=100,
    validation_split=0.2,
    verbose=0
)

print("Dropout model trained!")
```

**Hints:**
- First dropout: `0.5` (50% dropout rate)
- Second dropout: `0.3` (30% dropout rate)

**AI Help:** Ask ChatGPT: "How does dropout prevent overfitting in neural networks?"

### Task 6.5: Compare all three approaches

**Copy and run this code:**

```python
# Compare all three: No regularization, L2, Dropout
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history_no_reg.history['val_loss'], label='No Regularization', linewidth=2)
plt.plot(history_l2.history['val_loss'], label='L2 Regularization', linewidth=2)
plt.plot(history_dropout.history['val_loss'], label='Dropout', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.title('Regularization Techniques Comparison - Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history_no_reg.history['val_accuracy'], label='No Regularization', linewidth=2)
plt.plot(history_l2.history['val_accuracy'], label='L2 Regularization', linewidth=2)
plt.plot(history_dropout.history['val_accuracy'], label='Dropout', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.title('Regularization Techniques Comparison - Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Print final results
_, acc_no_reg = model_no_reg.evaluate(X_test, y_test, verbose=0)
_, acc_l2 = model_l2.evaluate(X_test, y_test, verbose=0)
_, acc_dropout = model_dropout.evaluate(X_test, y_test, verbose=0)

print("\n" + "="*60)
print("FINAL TEST ACCURACY COMPARISON")
print("="*60)
print(f"No Regularization: {acc_no_reg:.4f}")
print(f"L2 Regularization: {acc_l2:.4f}")
print(f"Dropout:           {acc_dropout:.4f}")
print("="*60)
```

**Questions:**

1. Which regularization technique gave the best test accuracy?
```
Answer: _______________________________________________
```

2. Looking at the validation curves, which technique prevented overfitting most effectively?
```
Answer: _______________________________________________
```

---

### Task 6.6: Comparing Optimizers

**Background:** The optimizer determines how weights are updated. Let's compare SGD (basic), SGD with momentum, and Adam (adaptive learning rates).

**Fill in the blanks:**

```python
# Build three identical models with different optimizers

# Model 1: SGD
model_sgd = Sequential([
    Dense(32, input_dim=64, activation='relu'),
    Dense(10, activation='softmax')
])
model_sgd.compile(
    optimizer=SGD(learning_rate=____),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Model 2: SGD with Momentum
model_momentum = Sequential([
    Dense(32, input_dim=64, activation='relu'),
    Dense(10, activation='softmax')
])
model_momentum.compile(
    optimizer=SGD(learning_rate=0.01, momentum=____),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Model 3: Adam
model_adam = Sequential([
    Dense(32, input_dim=64, activation='relu'),
    Dense(10, activation='softmax')
])
model_adam.compile(
    optimizer=____,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("Three models with different optimizers created!")
```

**Hints:**
- SGD learning rate: `0.01`
- Momentum: `0.9` (typical value)
- Adam optimizer: `'adam'` or `Adam()` (uses default learning rate)

**AI Help:** Ask ChatGPT: "What are the differences between SGD,SGD with momentum, and Adam optimizers?"

### Task 6.7: Train and compare convergence

**Fill in the blanks:**

```python
# Train all three
print("Training with different optimizers...")

history_sgd = model_sgd.fit(X_train, y_train, epochs=50, validation_split=0.2, verbose=0)
history_momentum = model_momentum.fit(____, ____, epochs=____, validation_split=____, verbose=0)
history_adam = model_adam.fit(X_train, y_train, epochs=50, validation_split=0.2, verbose=0)

print("Training complete!")
```

**Hints:**
- All use same data: `X_train`, `y_train`
- All use same settings: `epochs=50`, `validation_split=0.2`

### Task 6.8: Visualize optimizer comparison

**Copy and run this code:**

```python
# Plot training curves
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history_sgd.history['loss'], label='SGD', linewidth=2)
plt.plot(history_momentum.history['loss'], label='SGD + Momentum', linewidth=2)
plt.plot(history_adam.history['loss'], label='Adam', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Optimizer Comparison - Training Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history_sgd.history['val_accuracy'], label='SGD', linewidth=2)
plt.plot(history_momentum.history['val_accuracy'], label='SGD + Momentum', linewidth=2)
plt.plot(history_adam.history['val_accuracy'], label='Adam', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.title('Optimizer Comparison - Validation Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Final comparison
_, acc_sgd = model_sgd.evaluate(X_test, y_test, verbose=0)
_, acc_momentum = model_momentum.evaluate(X_test, y_test, verbose=0)
_, acc_adam = model_adam.evaluate(X_test, y_test, verbose=0)

print("\n" + "="*60)
print("OPTIMIZER FINAL TEST ACCURACY")
print("="*60)
print(f"SGD:              {acc_sgd:.4f}")
print(f"SGD + Momentum:   {acc_momentum:.4f}")
print(f"Adam:             {acc_adam:.4f}")
print("="*60)
```

**Questions:**

1. Which optimizer converged fastest (reached good accuracy quickest)?
```
Answer: _______________________________________________
```

2. Which optimizer achieved the best final test accuracy?
```
Answer: _______________________________________________
```

3. Why is Adam often the default choice?
```
Answer: _______________________________________________
___________________________________________________________
```

---

### Task 6.9: Combining Techniques

**Background:** In practice, you often combine multiple techniques. Let's build a "production-ready" model with everything!

**Fill in the blanks:**

```python
# Build a model with L2, Dropout, and Adam
model_best = Sequential([
    Dense(64, input_dim=64, activation='relu', kernel_regularizer=____),
    Dropout(____),
    Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.3),
    Dense(10, activation='softmax')
])

model_best.compile(
    optimizer=____,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train with early stopping awareness (we'll just run fixed epochs)
history_best = model_best.fit(
    X_train, y_train,
    epochs=100,
    validation_split=0.2,
    verbose=0
)

_, acc_best = model_best.evaluate(X_test, y_test, verbose=0)
print(f"\nBest Model Test Accuracy: {acc_best:.4f}")
```

**Hints:**
- Use `l2(0.01)` for regularization
- Use `0.5` for first dropout
- Use `'adam'` optimizer

### Task 6.10: Final comparison

**Copy and run this code:**

```python
# Compare your basic model from Part 1 with this advanced model
print("\n" + "="*60)
print("BASIC vs ADVANCED MODEL")
print("="*60)
print(f"Basic model (Part 1):     {test_acc:.4f}")
print(f"Advanced model (Part 6):  {acc_best:.4f}")
print(f"Improvement:              {(acc_best - test_acc)*100:.2f}%")
print("="*60)
```

---

### Advanced Reflection Questions

**Question 1:** You applied L2 regularization, dropout, and compared optimizers. Which technique had the biggest impact on preventing overfitting in your experiments?

```
Your answer: _______________________________________________
___________________________________________________________
```

**Question 2:** Why do we apply dropout during training but not during testing/prediction?

```
Your answer: _______________________________________________
___________________________________________________________
```

**Question 3:** You found that Adam converges faster than SGD. Does this mean Adam is always better? When might you still use SGD?

```
Your answer: _______________________________________________
___________________________________________________________
```

**Question 4:** In your "best model" (Task 6.9), you combined L2 regularization and dropout. Can you explain how these two techniques work differently to prevent overfitting?

```
Your answer: _______________________________________________
___________________________________________________________
___________________________________________________________
```

**Question 5:** Looking at all your experiments today, what would be your recommended "starting point" architecture and training setup for a new classification problem?

```
Your answer: _______________________________________________
___________________________________________________________
___________________________________________________________
```

---

**Congratulations!** You've mastered advanced neural network techniques and understand how to build production-ready models with regularization, dropout, and proper optimizer selection!

---

**End of Lab Worksheet**

<!-- COURSE_FEEDBACK_QR:START -->
---

## 30-second feedback

Scan the code to share anonymous feedback or post a question for this lab. It opens the correct **Lesson 09 lab** record automatically.

![Feedback QR code for Lesson 09 lab](./feedback-qr.png){fig-align="center" width="180px"}
<!-- COURSE_FEEDBACK_QR:END -->
