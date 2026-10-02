---
pagetitle: "Lab 10: Convolutional Neural Networks"
---

# Lab 10: Convolutional Neural Networks
## MPS311/439 - Machine Learning
**Lesson 10 Lab Session | Duration: 50 minutes**

---

## Introduction

Welcome to Lab 10! Today you'll build your first **Convolutional Neural Network (CNN)** - the architecture that revolutionized computer vision!

**What you'll learn today:**
- How to build a CNN using Keras (Conv2D, MaxPooling2D, Flatten, Dense)
- How to train a CNN on image data
- What patterns CNN filters actually learn
- How architecture choices affect performance
- How to use pre-trained models (MPS439)

**Remember:** This lab uses a fill-in-the-blanks approach. Don't write code from scratch - just fill in the blanks marked with `____`. Focus on understanding CNNs!

**Dataset:** We'll use **Fashion-MNIST** - 70,000 images of clothing items (28×28 grayscale). It has 10 categories: T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot.

---

## Setup: Import Libraries and Load Data

CNNs expect image data in a specific format: (height, width, channels). For grayscale images, channels = 1. For color images, channels = 3 (RGB).

**IMPORTANT:** The critical step for CNNs is reshaping data to add the channel dimension!

**Copy and run this code:**

```python
# Import packages
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers, models

# Load Fashion-MNIST
(X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()

# Class names
class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Normalize pixel values to [0, 1]
X_train = X_train / 255.0
X_test = X_test / 255.0

# CRITICAL: Reshape to add channel dimension
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")
print(f"Image shape: {X_train.shape[1:]}")
print(f"Number of classes: {len(class_names)}")
print(f"\nData ready! Shape is now (samples, height, width, channels)")
```

**What's happening here?**
- We loaded 60,000 training images and 10,000 test images
- Each image is 28×28 pixels, grayscale
- We normalized pixels from [0, 255] to [0, 1] for better training
- We reshaped from (28, 28) to (28, 28, 1) - CNNs need the channel dimension!

**Let's look at some examples:**

```python
# Show first 10 images
plt.figure(figsize=(12, 3))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(X_train[i].reshape(28, 28), cmap='gray')
    plt.title(class_names[y_train[i]])
    plt.axis('off')
plt.tight_layout()
plt.show()
```

---

## Part 1: Build Your First CNN (10 minutes)

**Background:** A CNN has three types of layers:
1. **Conv2D** - applies filters to detect patterns (edges, textures)
2. **MaxPooling2D** - reduces dimensions, keeps important features
3. **Dense** - final classification layers (same as before!)

The typical pattern is: Conv → Pool → Conv → Pool → Flatten → Dense → Output

### Task 1.1: Create the CNN architecture

**Fill in the blanks below:**

```python
# Create CNN model
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    # First convolutional block
    layers.Conv2D(____, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Second convolutional block
    layers.Conv2D(____, (3, 3), activation='____'),
    layers.MaxPooling2D((2, 2)),
    
    # Flatten and dense layers
    layers.____(),
    layers.Dense(128, activation='relu'),
    layers.Dense(____, activation='softmax')
])

print("Model created!")
model.summary()
```

**Hints:**
- First Conv2D: use 32 filters
- Second Conv2D: use 64 filters (double after pooling)
- Second Conv2D activation: use 'relu'
- Layer to flatten: `Flatten`
- Output layer: 10 units (one per class)

**AI Help:** Ask ChatGPT: "What does Conv2D and MaxPooling2D do in a CNN?"

**Question:** Look at the model summary. How many parameters are in the first Conv2D layer?

```
Number of parameters: ____________
```

### Task 1.2: Understand the architecture

**Fill in the blanks by reading the model summary:**

```python
# After first Conv2D layer, shape is: (____, ____, ____)
# After first MaxPooling2D, shape is: (____, ____, ____)
# After second Conv2D layer, shape is: (____, ____, ____)
# After second MaxPooling2D, shape is: (____, ____, ____)
# After Flatten, shape is: (____)

print("Check your answers by looking at the model.summary() output above!")
```

**Hint:** The summary shows "Output Shape" for each layer

**Question:** Why do the spatial dimensions (height, width) get smaller while the channels get larger?

```
Your answer: _______________________________________________
___________________________________________________________
```

---

## Part 2: Train the CNN (8 minutes)

**Background:** Training a CNN is the same as training a regular neural network - we use an optimizer, a loss function, and train for multiple epochs.

### Task 2.1: Compile the model

**Fill in the blanks:**

```python
# Compile model
model.compile(
    optimizer='____',
    loss='____',
    metrics=['accuracy']
)

print("Model compiled!")
```

**Hints:**
- Optimizer: use 'adam' (works well for most cases)
- Loss: use 'sparse_categorical_crossentropy' (for integer labels)

**AI Help:** Ask ChatGPT: "What's the difference between categorical_crossentropy and sparse_categorical_crossentropy?"

### Task 2.2: Train the model

**Fill in the blanks:**

```python
# Train model
history = model.fit(
    X_train, y_train,
    epochs=____,
    validation_split=0.2,
    batch_size=128,
    verbose=1
)

print("\nTraining complete!")
```

**Hint:**
- Use 5 epochs (more takes too long for lab)

**What's happening?**
- The model sees 80% of training data, keeps 20% for validation
- Each epoch processes all training data once
- Batch size = 128 means we update weights after every 128 images

**Question:** Watch the training output. Does validation accuracy increase each epoch?

```
Your observation: _______________________________________________
```

### Task 2.3: Plot training curves

**Fill in the blanks:**

```python
# Plot accuracy and loss
plt.figure(figsize=(12, 4))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['____'], label='Training')
plt.plot(history.history['____'], label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.legend()
plt.grid(True)

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['____'], label='Training')
plt.plot(history.history['____'], label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
```

**Hints:**
- Training accuracy: 'accuracy'
- Validation accuracy: 'val_accuracy'
- Training loss: 'loss'
- Validation loss: 'val_loss'

**Questions:**

1. Is there a gap between training and validation accuracy? Is the model overfitting?
```
Your answer: _______________________________________________
```

2. What would you do if the gap was very large?
```
Your answer: _______________________________________________
```

---

## Part 3: Make Predictions (8 minutes)

**Background:** Now let's test our trained CNN! We'll make predictions on the test set and visualize some results.

### Task 3.1: Evaluate on test set

**Fill in the blanks:**

```python
# Evaluate on test data
test_loss, test_acc = model.____(X_test, y_test, verbose=0)

print(f"Test accuracy: {test_acc:.4f}")
print(f"Test loss: {test_loss:.4f}")
```

**Hint:**
- Method to evaluate: `.evaluate()`

**Question:** How does test accuracy compare to validation accuracy? Are they similar?

```
Your answer: _______________________________________________
```

### Task 3.2: Make predictions

**Fill in the blanks:**

```python
# Predict on test set
predictions = model.____(X_test)

# Get predicted class (highest probability)
predicted_classes = np.argmax(____, axis=1)

print(f"Predictions shape: {predictions.shape}")
print(f"First prediction probabilities: {predictions[0]}")
print(f"Predicted class: {predicted_classes[0]} ({class_names[predicted_classes[0]]})")
print(f"True class: {y_test[0]} ({class_names[y_test[0]]})")
```

**Hints:**
- Method to predict: `.predict()`
- Find max along axis=1 (across classes)

**AI Help:** Ask ChatGPT: "What does argmax do in NumPy?"

### Task 3.3: Visualize correct predictions

**Copy and run this code:**

```python
# Show 10 correct predictions
correct_indices = np.where(predicted_classes == y_test)[0]

plt.figure(figsize=(12, 3))
for i in range(10):
    idx = correct_indices[i]
    plt.subplot(2, 5, i+1)
    plt.imshow(X_test[idx].reshape(28, 28), cmap='gray')
    plt.title(f'True: {class_names[y_test[idx]]}\nPred: {class_names[predicted_classes[idx]]}')
    plt.axis('off')
plt.suptitle('Correct Predictions', fontsize=14, y=1.02)
plt.tight_layout()
plt.show()
```

### Task 3.4: Visualize incorrect predictions

**Fill in the blanks:**

```python
# Show 10 incorrect predictions
incorrect_indices = np.where(____ != y_test)[0]

plt.figure(figsize=(12, 3))
for i in range(10):
    idx = incorrect_indices[i]
    plt.subplot(2, 5, i+1)
    plt.imshow(X_test[idx].reshape(28, 28), cmap='gray')
    plt.title(f'True: {class_names[y_test[idx]]}\nPred: {class_names[____[idx]]}')
    plt.axis('off')
plt.suptitle('Incorrect Predictions', fontsize=14, y=1.02)
plt.tight_layout()
plt.show()
```

**Hints:**
- Compare `predicted_classes` with `y_test`
- Use `predicted_classes` for predicted labels

**Questions:**

1. What types of clothing does the model confuse? (e.g., Shirt vs T-shirt?)
```
Your answer: _______________________________________________
___________________________________________________________
```

2. Can you understand why it made these mistakes by looking at the images?
```
Your answer: _______________________________________________
```

---

## Part 4: Understanding Filters (8 minutes)

**Background:** The first convolutional layer learns to detect basic patterns like edges and simple shapes. Let's visualize what the 32 filters in our first layer actually learned!

### Task 4.1: Extract filter weights

**Fill in the blanks:**

```python
# Get weights from first Conv2D layer
first_layer = model.layers[____]
filters, biases = first_layer.____

print(f"Filter shape: {filters.shape}")
print(f"Number of filters: {filters.shape[3]}")
print(f"Filter size: {filters.shape[0]}x{filters.shape[1]}")
```

**Hints:**
- First Conv2D is layer index 0
- Method to get weights: `.get_weights()`

**AI Help:** Ask ChatGPT: "How do I extract and visualize CNN filter weights in Keras?"

### Task 4.2: Visualize the filters

**Fill in the blanks:**

```python
# Plot first 32 filters
plt.figure(figsize=(12, 6))
for i in range(32):
    plt.subplot(4, 8, i+1)
    # Get the i-th filter
    f = filters[:, :, ____, ____]
    plt.imshow(f, cmap='gray')
    plt.axis('off')
plt.suptitle('32 Learned Filters from First Conv2D Layer', fontsize=14)
plt.tight_layout()
plt.show()
```

**Hints:**
- Input channel index: 0 (grayscale has only 1 input channel)
- Filter index: i

**Questions:**

1. Can you see edge detectors (vertical, horizontal, diagonal) in some filters?
```
Your answer: _______________________________________________
```

2. Some filters look like noise. Why might that happen?
```
Your answer: _______________________________________________
```

### Task 4.3: Apply a filter to an image

Let's manually apply one filter to see what it detects!

**Fill in the blanks:**

```python
# Pick a test image
test_img = X_test[0]

# Get one filter (let's use filter 5)
one_filter = filters[:, :, 0, 5]

# Apply convolution manually using correlation (approximate)
from scipy.ndimage import correlate
feature_map = correlate(test_img.reshape(28, 28), ____, mode='constant')

# Show results
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(test_img.reshape(28, 28), cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(____, cmap='gray')
plt.title('Filter (3x3)')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(____, cmap='viridis')
plt.title('Feature Map\n(What filter detected)')
plt.axis('off')
plt.colorbar()

plt.tight_layout()
plt.show()
```

**Hints:**
- Correlate with `one_filter`
- Plot `one_filter` for the filter
- Plot `feature_map` for the output

**Question:** What pattern did this filter detect in the image? (edges, textures, etc.)

```
Your answer: _______________________________________________
```

---

## Part 5: Architecture Experiments (8 minutes)

**Background:** CNN performance depends on architecture choices. Let's experiment with different configurations and see what happens!

### Task 5.1: Try fewer filters

**Fill in the blanks:**

```python
# Create smaller CNN (16 and 32 filters instead of 32 and 64)
model_small = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(____, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(____, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model_small.compile(optimizer='adam', 
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])

# Train
history_small = model_small.fit(X_train, y_train, epochs=____, 
                                validation_split=0.2, batch_size=128, verbose=0)

# Evaluate
_, test_acc_small = model_small.evaluate(X_test, y_test, verbose=0)
print(f"Small model test accuracy: {test_acc_small:.4f}")
print(f"Original model test accuracy: {test_acc:.4f}")
```

**Hints:**
- Use 16 filters in first Conv2D
- Use 32 filters in second Conv2D
- Train for 5 epochs

### Task 5.2: Try more filters

**Fill in the blanks:**

```python
# Create larger CNN (64 and 128 filters)
model_large = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(____, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(____, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model_large.compile(optimizer='adam',
                   loss='sparse_categorical_crossentropy', 
                   metrics=['accuracy'])

# Train
history_large = model_large.fit(X_train, y_train, epochs=5,
                                validation_split=0.2, batch_size=128, verbose=0)

# Evaluate
_, test_acc_large = model_large.evaluate(X_test, y_test, verbose=0)
print(f"Large model test accuracy: {test_acc_large:.4f}")
```

**Hints:**
- Use 64 filters in first Conv2D
- Use 128 filters in second Conv2D

### Task 5.3: Compare all models

**Copy and run this code:**

```python
# Compare parameter counts
print("\nModel Comparison:")
print(f"Small model  - Parameters: {model_small.count_params():,} - Test Acc: {test_acc_small:.4f}")
print(f"Original model - Parameters: {model.count_params():,} - Test Acc: {test_acc:.4f}")
print(f"Large model  - Parameters: {model_large.count_params():,} - Test Acc: {test_acc_large:.4f}")

# Plot comparison
models_names = ['Small\n(16,32)', 'Original\n(32,64)', 'Large\n(64,128)']
accuracies = [test_acc_small, test_acc, test_acc_large]
params = [model_small.count_params(), model.count_params(), model_large.count_params()]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].bar(models_names, accuracies, color=['blue', 'green', 'red'])
axes[0].set_ylabel('Test Accuracy')
axes[0].set_title('Model Accuracy Comparison')
axes[0].set_ylim(0.8, 0.95)
axes[0].grid(True, alpha=0.3)

axes[1].bar(models_names, params, color=['blue', 'green', 'red'])
axes[1].set_ylabel('Number of Parameters')
axes[1].set_title('Model Size Comparison')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

**Questions:**

1. Does more parameters always mean better accuracy?
```
Your answer: _______________________________________________
```

2. If you had limited compute resources, which model would you choose and why?
```
Your answer: _______________________________________________
___________________________________________________________
```

3. What's the trade-off between model size and accuracy?
```
Your answer: _______________________________________________
```

---

## Reflection Questions (5 minutes)

Answer these questions based on what you've learned today:

**Question 1:** Why do we add the channel dimension when reshaping data for CNNs? What would happen if we forgot?

```
Your answer: _______________________________________________
___________________________________________________________
```

**Question 2:** Explain in your own words what MaxPooling2D does and why it's useful.

```
Your answer: _______________________________________________
___________________________________________________________
```

**Question 3:** The first Conv2D layer has 320 parameters (32 filters × (3×3 + 1 bias)). Why is this so much fewer than a fully-connected layer?

```
Your answer: _______________________________________________
___________________________________________________________
```

**Question 4:** You trained a CNN that gets 95% training accuracy but only 75% validation accuracy. What's happening and what would you do?

```
Your answer: _______________________________________________
___________________________________________________________
```

**Question 5:** If you had color images (RGB) instead of grayscale, how would the architecture change?

```
Your answer: _______________________________________________
___________________________________________________________
```

---

## Summary

**What you learned today:**
- ✅ How to build CNNs using Conv2D, MaxPooling2D, and Dense layers
- ✅ How to reshape image data for CNNs (adding channel dimension)
- ✅ How to train and evaluate CNNs on Fashion-MNIST
- ✅ What filters actually learn (edges, patterns, textures)
- ✅ How architecture choices (number of filters) affect performance

**Key takeaways:**
- CNNs need data shaped as (height, width, channels)
- Conv2D applies learned filters to detect patterns
- MaxPooling2D reduces dimensions and provides translation invariance
- More filters = more capacity but also more parameters
- First layers learn simple patterns (edges), deeper layers learn complex patterns
- Always normalize pixel values to [0, 1] for better training!

---

## Part 6: Advanced Challenge - Transfer Learning (For MPS439 Students Only) (30 minutes)

**Background:** Training large CNNs from scratch requires lots of data and time. **Transfer learning** lets us use a model pre-trained on millions of images (ImageNet) and adapt it to our task. This is how real-world CNN applications work!

We'll use **VGG16**, a famous CNN architecture trained on ImageNet. We'll:
1. Load the pre-trained VGG16 (without the top classification layer)
2. Freeze its weights (we won't retrain the feature extraction)
3. Add our own classification head for Fashion-MNIST
4. Train only the new layers

### Task 6.1: Prepare data for VGG16

VGG16 was originally trained on 224×224 RGB images, but with `include_top=False` it also accepts smaller images (at least 32×32). We use 64×64 here so the exercise runs on a typical laptop, then convert Fashion-MNIST from grayscale to RGB.

**Fill in the blanks:**

```python
# Use a smaller VGG16-compatible size for a practical lab runtime
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

# Resize images to 64x64 and convert to RGB
def prepare_for_vgg(images):
    # Resize to 64x64
    from tensorflow.image import resize
    images_resized = resize(images, [____, ____])
    
    # Convert grayscale to RGB by repeating channel 3 times
    images_rgb = np.repeat(images_resized, ____, axis=-1)
    
    # VGG16 preprocessing
    images_processed = preprocess_input(images_rgb)
    
    return images_processed

# Use subset for speed (VGG16 is large!)
X_train_vgg = prepare_for_vgg(X_train[:1000])
X_test_vgg = prepare_for_vgg(X_test[:200])
y_train_vgg = y_train[:1000]
y_test_vgg = y_test[:200]

print(f"Training samples: {X_train_vgg.shape[0]}")
print(f"Image shape: {X_train_vgg.shape[1:]}")
print(f"Ready for VGG16!")
```

**Hints:**
- Resize to 64 by 64
- Repeat 3 times for RGB

**AI Help:** Ask ChatGPT: "How do I use transfer learning with VGG16 in Keras?"

### Task 6.2: Load pre-trained VGG16

**Fill in the blanks:**

```python
# Load VGG16 without top layers (no classification head)
base_model = VGG16(
    weights='____',
    include_top=____,
    input_shape=(64, 64, 3)
)

# Freeze base model weights
base_model.trainable = ____

print(f"VGG16 loaded!")
print(f"Number of layers: {len(base_model.layers)}")
print(f"Trainable: {base_model.trainable}")
```

**Hints:**
- Use 'imagenet' weights
- Set include_top=False (we'll add our own)
- Set trainable=False (freeze the weights)

### Task 6.3: Add custom classification head

**Fill in the blanks:**

```python
# Build model with VGG16 base and custom head
model_vgg = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(____, activation='relu'),
    layers.Dense(____, activation='softmax')
])

model_vgg.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\nModel architecture:")
model_vgg.summary()
```

**Hints:**
- Dense layer: use 256 units
- Output layer: 10 units (10 classes)

**Question:** How many parameters are trainable vs non-trainable?

```
Trainable parameters: ____________
Non-trainable parameters: ____________
```

### Task 6.4: Train with transfer learning

**Fill in the blanks:**

```python
# Train only the new layers
history_vgg = model_vgg.fit(
    X_train_vgg, y_train_vgg,
    epochs=____,
    validation_split=0.2,
    batch_size=32,
    verbose=1
)

# Evaluate
_, test_acc_vgg = model_vgg.evaluate(X_test_vgg, y_test_vgg, verbose=0)
print(f"\nTransfer learning test accuracy: {test_acc_vgg:.4f}")
```

**Hint:**
- Use 3 epochs (VGG16 is slow!)

**Question:** How does this accuracy compare to training from scratch? Is it better or worse with only 1000 training samples?

```
Your answer: _______________________________________________
___________________________________________________________
```

### Task 6.5: Fine-tuning (optional - if time permits)

Now let's unfreeze the last few layers of VGG16 and fine-tune them.

**Fill in the blanks:**

```python
# Unfreeze last 4 layers of VGG16
base_model.trainable = True
for layer in base_model.layers[:-4]:
    layer.trainable = ____

# Recompile with lower learning rate
model_vgg.compile(
    optimizer=keras.optimizers.Adam(learning_rate=____),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print(f"Trainable layers: {sum([1 for layer in model_vgg.layers if layer.trainable])}")

# Fine-tune
history_finetune = model_vgg.fit(
    X_train_vgg, y_train_vgg,
    epochs=2,
    validation_split=0.2,
    batch_size=32,
    verbose=1
)

# Evaluate
_, test_acc_finetune = model_vgg.evaluate(X_test_vgg, y_test_vgg, verbose=0)
print(f"\nFine-tuned test accuracy: {test_acc_finetune:.4f}")
```

**Hints:**
- Keep first layers frozen: trainable=False
- Use low learning rate: 1e-5 (or 0.00001)

### Task 6.6: Compare all approaches

**Copy and run this code:**

```python
# Compare training from scratch vs transfer learning
print("\n" + "="*60)
print("COMPARISON OF APPROACHES")
print("="*60)
print(f"Small CNN (from scratch, 60k samples): {test_acc:.4f}")
print(f"Transfer Learning (5k samples):        {test_acc_vgg:.4f}")
if 'test_acc_finetune' in locals():
    print(f"Fine-tuned (5k samples):               {test_acc_finetune:.4f}")
print("="*60)

# Plot training curves
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history_vgg.history['accuracy'], label='Training')
plt.plot(history_vgg.history['val_accuracy'], label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Transfer Learning - Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history_vgg.history['loss'], label='Training')
plt.plot(history_vgg.history['val_loss'], label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Transfer Learning - Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
```

### Task 6.7: Visualize VGG16 predictions

**Fill in the blanks:**

```python
# Make predictions with VGG16
predictions_vgg = model_vgg.____(X_test_vgg)
predicted_classes_vgg = np.argmax(____, axis=1)

# Show some predictions
plt.figure(figsize=(12, 6))
for i in range(12):
    plt.subplot(3, 4, i+1)
    # Get original image (before resizing)
    orig_img = X_test[i].reshape(28, 28)
    plt.imshow(orig_img, cmap='gray')
    
    true_label = class_names[y_test_vgg[i]]
    pred_label = class_names[____[i]]
    correct = y_test_vgg[i] == predicted_classes_vgg[i]
    
    color = 'green' if correct else 'red'
    plt.title(f'True: {true_label}\nPred: {pred_label}', color=color, fontsize=9)
    plt.axis('off')

plt.suptitle('VGG16 Transfer Learning Predictions', fontsize=14)
plt.tight_layout()
plt.show()
```

**Hints:**
- Use `.predict()` method
- Use `predictions_vgg` for argmax
- Use `predicted_classes_vgg` for predicted label

### Task 6.8: Advanced reflection

**Question 1:** Why did we freeze the VGG16 weights initially instead of training all layers?

```
Your answer: _______________________________________________
___________________________________________________________
```

**Question 2:** Transfer learning can work with only 1000 samples. Why is this possible?

```
Your answer: _______________________________________________
___________________________________________________________
```

**Question 3:** VGG16 was trained on ImageNet (real-world objects). Fashion-MNIST is quite different (simple clothing). Why does transfer learning still help?

```
Your answer: _______________________________________________
___________________________________________________________
```

**Question 4:** When would you NOT want to use transfer learning?

```
Your answer: _______________________________________________
___________________________________________________________
```

**Question 5:** What are the trade-offs between training from scratch vs transfer learning?

```
Your answer: _______________________________________________
___________________________________________________________
```

---

**Congratulations!** You've used a state-of-the-art pre-trained CNN and adapted it to a new task. This is exactly how modern computer vision applications are built!

---

**End of Lab Worksheet**

<!-- COURSE_FEEDBACK_QR:START -->
---

## 30-second feedback

What would help you learn better next time?

Scan to share anonymous feedback on today's lab.

![Feedback QR code for Lesson 10 lab](./feedback-qr.png){fig-align="center" width="180px"}
<!-- COURSE_FEEDBACK_QR:END -->
