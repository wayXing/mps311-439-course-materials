---
pagetitle: "Lab 3: Polynomial Features & Regularization"
---

# Lab 3: Polynomial Features & Regularization
## MPS311/439 - Machine Learning
**Lesson 3 Lab Session | Duration: 50 minutes**

**Objective**: Learn to create polynomial features, observe overfitting, and apply regularization techniques

---

## Setup: Import Libraries & Load Data (5 minutes)

First, let's import everything we need and prepare our dataset. **Copy and run this code block:**

```python
# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import r2_score

# Load California housing data (we'll use a subset to keep it simple)
housing = fetch_california_housing()
X_full = housing.data
y_full = housing.target

# Select just 2 features to keep things simple: MedInc (income) and HouseAge
# MedInc is column 0, HouseAge is column 1
X = X_full[:, [0, 1]]  
y = y_full

# Take a smaller sample (500 points) for faster computation
np.random.seed(42)
indices = np.random.choice(len(X), size=500, replace=False)
X = X[indices]
y = y[indices]

# Split into train and test (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")
print(f"Features: {housing.feature_names[:2]}")
print("\nSetup complete! ✓")
```

**What just happened?**
- We loaded California housing data
- Selected 2 features: median income and house age
- Used 500 houses (smaller dataset = faster learning)
- Split into 80% training, 20% test

---

## Part 1: Polynomial Features & Overfitting (15 minutes)

### Background
Remember from the lecture: we can make our model fit curves by creating polynomial features. For example, if we have feature `x`, we can create `x²`, `x³`, etc.

### Task 1.1: Create Polynomial Features (Degree 1 - Baseline)

Let's start with degree 1 (no polynomials, just the original features).

```python
# Create polynomial features of degree 1 (baseline - no actual polynomials)
poly1 = PolynomialFeatures(degree=1, include_bias=False)
X_train_poly1 = poly1.fit_transform(X_train)
X_test_poly1 = poly1.transform(X_test)

# Fit a linear regression model
model1 = LinearRegression()
model1.fit(X_train_poly1, y_train)

# Calculate R² scores
train_r2_deg1 = model1.score(X_train_poly1, y_train)
test_r2_deg1 = model1.score(X_test_poly1, y_test)

print(f"Degree 1 - Train R²: {train_r2_deg1:.3f}")
print(f"Degree 1 - Test R²:  {test_r2_deg1:.3f}")
```

**Your turn**: Run this code and record the R² scores below:
- Train R²: _________
- Test R²: _________

---

### Task 1.2: Create Polynomial Features (Degree 2)

Now let's add squared terms (degree 2).

**YOUR TASK**: Complete the code below by filling in the blanks.

```python
# Create polynomial features of degree 2
poly2 = PolynomialFeatures(degree=____, include_bias=False)  # Fill in the degree
X_train_poly2 = poly2.fit_transform(X_train)
X_test_poly2 = poly2.transform(X_test)

# Fit a linear regression model
model2 = LinearRegression()
model2.fit(X_train_poly2, y_train)

# Calculate R² scores
train_r2_deg2 = model2.score(X_train_poly2, y_train)
test_r2_deg2 = model2.score(X_test_poly2, y_test)

print(f"Degree 2 - Train R²: {train_r2_deg2:.3f}")
print(f"Degree 2 - Test R²:  {test_r2_deg2:.3f}")
```

**Hint**: Look at the code above (Task 1.1) - you only need to change one number!

**Record your results**:
- Train R²: _________
- Test R²: _________

**Question**: Did the R² scores improve compared to Degree 1? _________

---

### Task 1.3: Create Polynomial Features (Degree 5 - High Complexity)

Now let's try degree 5 to see what happens with more complexity.

**YOUR TASK**: Complete the code below (following the same pattern).

```python
# Create polynomial features of degree 5
poly5 = PolynomialFeatures(degree=____, include_bias=False)  # Fill in the degree
X_train_poly5 = poly5.fit_transform(X_train)
X_test_poly5 = poly5.transform(X_test)

# Fit a linear regression model
model5 = LinearRegression()
model5.fit(____, y_train)  # Fill in: what data should we fit on?

# Calculate R² scores
train_r2_deg5 = model5.score(X_train_poly5, y_train)
test_r2_deg5 = model5.score(____, y_test)  # Fill in: what data should we test on?

print(f"Degree 5 - Train R²: {train_r2_deg5:.3f}")
print(f"Degree 5 - Test R²:  {test_r2_deg5:.3f}")
```

**Hint**: For `.fit()` use the training data with degree 5 polynomials. For `.score()` use the test data with degree 5 polynomials.

**Record your results**:
- Train R²: _________
- Test R²: _________

---

### Task 1.4: Observe Overfitting

**YOUR TASK**: Look at your three sets of results and answer these questions:

1. **Which model has the HIGHEST training R²?** 
   - [ ] Degree 1  
   - [ ] Degree 2  
   - [ ] Degree 5

2. **Which model has the HIGHEST test R²?**
   - [ ] Degree 1  
   - [ ] Degree 2  
   - [ ] Degree 5

3. **Which model shows signs of overfitting (big gap between train and test R²)?**
   - [ ] Degree 1  
   - [ ] Degree 2  
   - [ ] Degree 5

**Key Insight**: As we increase polynomial degree, the model fits the training data better and better. But at some point (degree 5), it starts to memorize noise, and test performance gets **worse**. This is overfitting!

---

## Part 2: Regularization - Ridge & Lasso (15 minutes)

### Background
From the lecture, we learned that regularization helps prevent overfitting by penalizing large coefficients. But **we must scale our data first** before applying regularization!

### Task 2.1: Scale the Data

**Copy and run this code** to scale your degree 5 polynomial features:

```python
# Create the scaler
scaler = StandardScaler()

# Fit on training data and transform both train and test
X_train_poly5_scaled = scaler.fit_transform(X_train_poly5)
X_test_poly5_scaled = scaler.transform(X_test_poly5)

print("Data scaled! ✓")
print(f"Original data mean: {X_train_poly5[0, 0]:.2f}")
print(f"Scaled data mean: {X_train_poly5_scaled[:, 0].mean():.2f}")
print(f"Scaled data std: {X_train_poly5_scaled[:, 0].std():.2f}")
```

**What you should see**: The scaled data should have mean ≈ 0 and standard deviation ≈ 1.

---

### Task 2.2: Apply Ridge Regression

Ridge regularization penalizes the sum of squared coefficients. The penalty strength is controlled by `alpha` (which is λ in the lecture).

**YOUR TASK**: Complete the code below.

```python
# Create Ridge model with alpha = 1.0
ridge_model = Ridge(alpha=1.0)

# Fit on SCALED training data
ridge_model.fit(X_train_poly5_scaled, y_train)

# Calculate R² scores
train_r2_ridge = ridge_model.score(____, y_train)  # Fill in: use scaled training data
test_r2_ridge = ridge_model.score(____, y_test)    # Fill in: use scaled test data

print(f"Ridge (α=1.0) - Train R²: {train_r2_ridge:.3f}")
print(f"Ridge (α=1.0) - Test R²:  {test_r2_ridge:.3f}")
```

**Hint**: Use the scaled versions of the data (the ones with `_scaled` in the name).

**Record your results**:
- Train R²: _________
- Test R²: _________

**Question**: Is the gap between train and test R² smaller than the unregularized degree 5 model (from Task 1.3)? _________

---

### Task 2.3: Apply Lasso Regression

Lasso penalizes the sum of absolute values of coefficients. It can shrink coefficients exactly to zero!

**YOUR TASK**: Complete the code below (follow the same pattern as Ridge).

```python
# Create Lasso model with alpha = 1.0
lasso_model = Lasso(alpha=____)  # Fill in the alpha value

# Fit on scaled training data
lasso_model.fit(____, y_train)  # Fill in: what data to fit on?

# Calculate R² scores
train_r2_lasso = lasso_model.score(X_train_poly5_scaled, y_train)
test_r2_lasso = lasso_model.score(X_test_poly5_scaled, y_test)

print(f"Lasso (α=1.0) - Train R²: {train_r2_lasso:.3f}")
print(f"Lasso (α=1.0) - Test R²:  {test_r2_lasso:.3f}")
```

**Hint**: Use `alpha=1.0` and fit on the scaled training data.

**Record your results**:
- Train R²: _________
- Test R²: _________

---

### Task 2.4: Compare Coefficients

Let's see how regularization affected the coefficients.

**Copy and run this code**:

```python
print("Number of features (degree 5 polynomials):", len(model5.coef_))
print("\nCoefficients comparison:")
print(f"Unregularized (some): {model5.coef_[:5]}")
print(f"Ridge (some):         {ridge_model.coef_[:5]}")
print(f"Lasso (some):         {lasso_model.coef_[:5]}")

# Count how many coefficients are exactly zero in Lasso
num_zero_lasso = np.sum(lasso_model.coef_ == 0)
print(f"\nNumber of zero coefficients in Lasso: {num_zero_lasso} out of {len(lasso_model.coef_)}")
```

**Question**: Do the regularized models have smaller coefficients than the unregularized model? _________

---

## Part 3: Cross-Validation & Finding the Best Alpha (15 minutes)

### Background
How do we find the best value of `alpha`? We use cross-validation on the **training set only** to test different values.

### Task 3.1: Test Multiple Alpha Values with Ridge

We'll test several alpha values and see which performs best using 5-fold cross-validation.

**Copy and run this code**:

```python
# Alpha values to test
alphas = [0.1, 1, 10, 100]

print("Testing Ridge with different alphas...")
print("=" * 50)

ridge_results = []

for alpha in alphas:
    # Create model
    ridge = Ridge(alpha=alpha)
    
    # Do 5-fold cross-validation on training data
    cv_scores = cross_val_score(ridge, X_train_poly5_scaled, y_train, 
                                 cv=5, scoring='r2')
    
    # Calculate mean CV score
    mean_cv_score = cv_scores.mean()
    ridge_results.append(mean_cv_score)
    
    print(f"Alpha = {alpha:6.1f}  |  Mean CV R² = {mean_cv_score:.3f}")

print("=" * 50)

# Find best alpha
best_ridge_alpha = alphas[np.argmax(ridge_results)]
print(f"\nBest Ridge alpha: {best_ridge_alpha}")
```

**Record the best alpha for Ridge**: _________

---

### Task 3.2: Test Multiple Alpha Values with Lasso

Now let's do the same for Lasso.

**YOUR TASK**: Complete the code below (follow the Ridge pattern above).

```python
# Alpha values to test (same as before)
alphas = [0.1, 1, 10, 100]

print("Testing Lasso with different alphas...")
print("=" * 50)

lasso_results = []

for alpha in alphas:
    # Create model
    lasso = Lasso(alpha=alpha)
    
    # Do 5-fold cross-validation on training data
    cv_scores = cross_val_score(____, X_train_poly5_scaled, y_train,  # Fill in: model name
                                 cv=5, scoring='r2')
    
    # Calculate mean CV score
    mean_cv_score = cv_scores.mean()
    lasso_results.append(mean_cv_score)
    
    print(f"Alpha = {alpha:6.1f}  |  Mean CV R² = {mean_cv_score:.3f}")

print("=" * 50)

# Find best alpha
best_lasso_alpha = alphas[np.argmax(lasso_results)]
print(f"\nBest Lasso alpha: {best_lasso_alpha}")
```

**Hint**: Replace the blank with the model variable name (the Lasso model you created).

**Record the best alpha for Lasso**: _________

---

### Task 3.3: Build Final Models with Best Alpha

Now let's train final models using the best alpha values we found.

**YOUR TASK**: Complete the code.

```python
# Train final Ridge model with best alpha
final_ridge = Ridge(alpha=best_ridge_alpha)
final_ridge.fit(X_train_poly5_scaled, y_train)

# Train final Lasso model with best alpha
final_lasso = Lasso(alpha=____)  # Fill in: use the best lasso alpha
final_lasso.fit(____, y_train)   # Fill in: use scaled training data

# Evaluate on test set
ridge_test_r2 = final_ridge.score(X_test_poly5_scaled, y_test)
lasso_test_r2 = final_lasso.score(X_test_poly5_scaled, y_test)

print(f"Final Ridge (α={best_ridge_alpha}) Test R²: {ridge_test_r2:.3f}")
print(f"Final Lasso (α={best_lasso_alpha}) Test R²: {lasso_test_r2:.3f}")
```

**Hint**: Use `best_lasso_alpha` for the alpha value and `X_train_poly5_scaled` for the training data.

---

### Task 3.4: Create a Comparison Table

Let's summarize everything we've learned!

**Copy and run this code**:

```python
print("\n" + "="*60)
print("FINAL MODEL COMPARISON")
print("="*60)
print(f"{'Model':<30} {'Train R²':>12} {'Test R²':>12}")
print("-"*60)
print(f"{'Degree 1 (baseline)':<30} {train_r2_deg1:>12.3f} {test_r2_deg1:>12.3f}")
print(f"{'Degree 2':<30} {train_r2_deg2:>12.3f} {test_r2_deg2:>12.3f}")
print(f"{'Degree 5 (no regularization)':<30} {train_r2_deg5:>12.3f} {test_r2_deg5:>12.3f}")
print(f"{'Ridge (α={best_ridge_alpha})':<30} {ridge_test_r2:>12.3f} {ridge_test_r2:>12.3f}")
print(f"{'Lasso (α={best_lasso_alpha})':<30} {lasso_test_r2:>12.3f} {lasso_test_r2:>12.3f}")
print("="*60)
```

---

## Reflection Questions (5 minutes)

Answer these based on your results:

1. **Which model performed best on the test set?**
   
   Answer: _______________________________________________

2. **What happened when we went from degree 2 to degree 5 without regularization?**
   
   Answer: _______________________________________________

3. **How did Ridge and Lasso help with the overfitting problem?**
   
   Answer: _______________________________________________

4. **Why did we need to scale the data before using Ridge/Lasso?**
   
   Answer: _______________________________________________

---

## Summary: What You Learned Today ✓

- ✅ How to create polynomial features using `PolynomialFeatures`
- ✅ How to observe overfitting (high training R², low test R²)
- ✅ How to scale data using `StandardScaler` (required before regularization!)
- ✅ How to apply Ridge and Lasso regularization
- ✅ How to use `cross_val_score` to find the best hyperparameter (alpha)
- ✅ How to compare multiple models

**These skills are essential for Assignment 1!**

---

## Optional Challenge (If you have extra time)

Try visualizing how coefficients change with different alpha values:

```python
# Test many alpha values
alphas_test = [0.01, 0.1, 1, 10, 100, 1000]
coef_ridge = []
coef_lasso = []

for alpha in alphas_test:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train_poly5_scaled, y_train)
    coef_ridge.append(ridge.coef_)
    
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_train_poly5_scaled, y_train)
    coef_lasso.append(lasso.coef_)

# Plot
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(alphas_test, coef_ridge)
plt.xscale('log')
plt.xlabel('Alpha')
plt.ylabel('Coefficient value')
plt.title('Ridge Coefficients vs Alpha')

plt.subplot(1, 2, 2)
plt.plot(alphas_test, coef_lasso)
plt.xscale('log')
plt.xlabel('Alpha')
plt.ylabel('Coefficient value')
plt.title('Lasso Coefficients vs Alpha')

plt.tight_layout()
plt.show()
```

**What do you notice?** Ridge coefficients shrink smoothly, while Lasso coefficients can drop to exactly zero!

---

**Lab Complete!** 🎉

Make sure you understand each step - you'll need these techniques for your assignment!



# Part 4: For MPS439 Students Only - Implementing Ridge from Scratch (15 minutes)

### Background: Understanding What's Happening Inside Ridge

In the lecture, you learned that Ridge regression minimizes:

$$ J(\mathbf{w}) = \text{MSE} + \lambda \sum_{j=1}^{p} w_j^2 $$

But how does sklearn actually find the optimal weights? It uses **gradient descent** - an iterative algorithm that takes small steps toward the minimum.

**The gradient descent update rule:**

$$ w_j^{new} = w_j^{old} - \eta \cdot \frac{\partial J}{\partial w_j} $$

where:
- $\eta$ is the **learning rate** (step size)
- $\frac{\partial J}{\partial w_j}$ is the **gradient** (direction of steepest increase)

For Ridge regression, the gradient is:

$$ \frac{\partial J}{\partial w_j} = -\frac{2}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i) \cdot x_{ij} + 2\lambda w_j $$

**In plain English:** 
- The first term says "adjust weights to reduce prediction error"
- The second term says "but also push weights toward zero"

---

### Task 4.1: Implement Gradient Descent for Ridge

**YOUR TASK**: Complete the missing parts in this Ridge implementation.

```python
def ridge_gradient_descent(X, y, alpha=1.0, learning_rate=0.01, n_iterations=1000):
    """
    Implement Ridge regression using gradient descent.
    
    Parameters:
    -----------
    X : array, shape (n_samples, n_features)
        Training data (must be scaled!)
    y : array, shape (n_samples,)
        Target values
    alpha : float
        Regularization strength (lambda in the lecture)
    learning_rate : float
        Step size for gradient descent (eta)
    n_iterations : int
        Number of iterations to run
    
    Returns:
    --------
    weights : array, shape (n_features,)
        Learned coefficients
    losses : list
        Loss at each iteration (for plotting convergence)
    """
    n_samples, n_features = X.shape
    
    # Initialize weights to zeros
    weights = np.zeros(n_features)
    losses = []
    
    for iteration in range(n_iterations):
        # 1. Make predictions
        y_pred = X @ weights  # Matrix multiplication: X times weights
        
        # 2. Calculate error
        error = y_pred - y
        
        # 3. Calculate MSE loss (without regularization, just for monitoring)
        mse = np.mean(error ** 2)
        
        # 4. Calculate regularization term
        reg_term = alpha * np.sum(weights ** 2)
        
        # 5. Total loss (MSE + regularization)
        total_loss = mse + reg_term
        losses.append(total_loss)
        
        # 6. Calculate gradient
        # Gradient = (2/n) * X^T * error + 2 * alpha * weights
        mse_gradient = (2 / n_samples) * (X.T @ error)
        reg_gradient = 2 * alpha * weights
        gradient = mse_gradient + ____  # FILL IN: add the regularization gradient
        
        # 7. Update weights
        weights = weights - ____ * gradient  # FILL IN: multiply by learning rate
    
    return weights, losses

# Test your implementation
print("Training our Ridge implementation...")
my_weights, my_losses = ridge_gradient_descent(
    X_train_poly5_scaled, 
    y_train, 
    alpha=1.0,
    learning_rate=0.01,
    n_iterations=1000
)

print("Training complete! ✓")
```

**Hints**: 
- For the first blank: add `reg_gradient` to complete the total gradient
- For the second blank: multiply gradient by `learning_rate`

---

### Task 4.2: Visualize Convergence

Let's see if our gradient descent actually worked - the loss should decrease over iterations.

**Copy and run this code**:

```python
# Plot how loss decreased during training
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(my_losses)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss vs Iteration (Full)')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(my_losses[100:])  # Skip first 100 iterations to see detail
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss vs Iteration (After 100 iterations)')
plt.grid(True)

plt.tight_layout()
plt.show()

print(f"Initial loss: {my_losses[0]:.3f}")
print(f"Final loss: {my_losses[-1]:.3f}")
print(f"Loss reduction: {my_losses[0] - my_losses[-1]:.3f}")
```

**Question**: Does the loss curve show convergence (flattening out)? _________

---

### Task 4.3: Compare with sklearn's Ridge

Now let's see how close our implementation is to sklearn's optimized version.

**YOUR TASK**: Complete the comparison code.

```python
# Train sklearn's Ridge for comparison
sklearn_ridge = Ridge(alpha=1.0)
sklearn_ridge.fit(____, y_train)  # FILL IN: use scaled training data

# Compare coefficients
print("\nCoefficient Comparison:")
print("="*60)
print(f"{'Feature':<15} {'Our Ridge':>15} {'sklearn Ridge':>15} {'Difference':>15}")
print("-"*60)

for i in range(min(5, len(my_weights))):  # Show first 5 features
    diff = abs(my_weights[i] - sklearn_ridge.coef_[i])
    print(f"Feature {i:<8} {my_weights[i]:>15.4f} {sklearn_ridge.coef_[i]:>15.4f} {diff:>15.4f}")

print("="*60)

# Calculate R² on test set
my_predictions = X_test_poly5_scaled @ my_weights
my_r2 = r2_score(y_test, my_predictions)

sklearn_r2 = sklearn_ridge.score(X_test_poly5_scaled, y_test)

print(f"\nTest R² - Our Ridge:    {my_r2:.4f}")
print(f"Test R² - sklearn Ridge: {sklearn_r2:.4f}")
print(f"Difference:              {abs(my_r2 - sklearn_r2):.4f}")
```

**Hint**: Use `X_train_poly5_scaled` for training sklearn's Ridge.

**Record your observations**:
- Are the coefficients similar? _________
- Is the R² similar? _________
- If there's a difference, why might that be? _________

---

### Task 4.4: Experiment with Learning Rate

The learning rate is crucial for gradient descent. Let's see what happens with different values.

**Copy and run this code**:

```python
# Test different learning rates
learning_rates = [0.001, 0.005, 0.01]

plt.figure(figsize=(12, 4))

for i, lr in enumerate(learning_rates):
    weights, losses = ridge_gradient_descent(
        X_train_poly5_scaled, 
        y_train, 
        alpha=1.0,
        learning_rate=lr,
        n_iterations=1000
    )
    
    plt.subplot(1, 3, i+1)
    plt.plot(losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title(f'Learning Rate = {lr}')
    plt.grid(True)
    
    # Calculate final R²
    predictions = X_test_poly5_scaled @ weights
    r2 = r2_score(y_test, predictions)
    plt.text(0.5, 0.95, f'Final R² = {r2:.3f}', 
             transform=plt.gca().transAxes, 
             verticalalignment='top')

plt.tight_layout()
plt.show()
```

**Questions**:
1. Which learning rate converged fastest? _________
2. Which learning rate gives the best final R²? _________
3. What would happen if the learning rate were too large? _________

---

### Task 4.5: Understanding the Regularization Effect

Let's visualize how the regularization term (alpha) affects the weights during training.

**Copy and run this code**:

```python
# Train with different alpha values and track weight magnitudes
alphas_test = [0.0, 0.1, 1.0, 10.0]
final_weights = []

for alpha in alphas_test:
    weights, _ = ridge_gradient_descent(
        X_train_poly5_scaled, 
        y_train, 
        alpha=alpha,
        learning_rate=0.01,
        n_iterations=1000
    )
    final_weights.append(weights)

# Plot coefficient magnitudes
plt.figure(figsize=(10, 5))

x = np.arange(len(final_weights[0]))
width = 0.2

for i, alpha in enumerate(alphas_test):
    plt.bar(x + i*width, np.abs(final_weights[i]), width, 
            label=f'α={alpha}', alpha=0.7)

plt.xlabel('Feature Index')
plt.ylabel('Absolute Coefficient Value')
plt.title('How Alpha Affects Coefficient Magnitudes')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Print summary statistics
print("\nCoefficient Magnitude Summary:")
print("="*60)
print(f"{'Alpha':<10} {'Max |coef|':>15} {'Mean |coef|':>15} {'Sum |coef|²':>15}")
print("-"*60)
for i, alpha in enumerate(alphas_test):
    max_coef = np.max(np.abs(final_weights[i]))
    mean_coef = np.mean(np.abs(final_weights[i]))
    sum_sq = np.sum(final_weights[i] ** 2)
    print(f"{alpha:<10.1f} {max_coef:>15.4f} {mean_coef:>15.4f} {sum_sq:>15.4f}")
print("="*60)
```

**Questions**:
1. What happens to coefficient magnitudes as alpha increases? _________
2. Does alpha=0 (no regularization) give the largest coefficients? _________
3. Why does Ridge "shrink" coefficients toward zero? _________

---

## Advanced Challenge: Implement Lasso (Optional)

If you have extra time and want to tackle Lasso for your assignment, here's the key difference:

**Lasso gradient** has a non-smooth component:

$$ \frac{\partial J}{\partial w_j} = -\frac{2}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i) \cdot x_{ij} + \lambda \cdot \text{sign}(w_j) $$

Where $\text{sign}(w_j)$ is:
- +1 if $w_j > 0$
- -1 if $w_j < 0$  
- 0 if $w_j = 0$

**Try implementing this**:

```python
def lasso_gradient_descent(X, y, alpha=1.0, learning_rate=0.01, n_iterations=1000):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    losses = []
    
    for iteration in range(n_iterations):
        y_pred = X @ weights
        error = y_pred - y
        
        mse = np.mean(error ** 2)
        reg_term = alpha * np.sum(np.abs(weights))
        total_loss = mse + reg_term
        losses.append(total_loss)
        
        # MSE gradient
        mse_gradient = (2 / n_samples) * (X.T @ error)
        
        # Lasso regularization gradient (subgradient)
        reg_gradient = alpha * np.sign(weights)
        
        gradient = mse_gradient + reg_gradient
        weights = weights - learning_rate * gradient
        
        # Soft thresholding (helps push small weights to exactly zero)
        threshold = learning_rate * alpha
        weights = np.sign(weights) * np.maximum(np.abs(weights) - threshold, 0)
    
    return weights, losses

# Test it
lasso_weights, lasso_losses = lasso_gradient_descent(
    X_train_poly5_scaled, y_train, alpha=1.0, learning_rate=0.01, n_iterations=1000
)

print(f"Number of zero coefficients in our Lasso: {np.sum(lasso_weights == 0)}")
print(f"Number of non-zero coefficients: {np.sum(lasso_weights != 0)}")
```

---

## Summary: What MPS439 Students Learned ✓

**Beyond the core content, you now understand:**

- ✅ **The math behind Ridge regression** - how the regularization term affects the gradient
- ✅ **Gradient descent algorithm** - the iterative optimization process
- ✅ **Learning rate tuning** - how step size affects convergence
- ✅ **Implementation from scratch** - not just using sklearn as a black box
- ✅ **Coefficient shrinkage mechanics** - why regularization reduces weight magnitudes

**This prepares you for Assignment 1 Options A or B:**
- **Option A**: You can now implement and analyze Ridge regression
- **Option B**: You have the foundation to extend to Lasso (see the optional challenge)

**Key insights:**
1. sklearn's Ridge uses a closed-form solution (faster than our gradient descent)
2. Gradient descent gives you control and understanding of the optimization process
3. The regularization term literally adds to the gradient, pulling weights toward zero
4. Different hyperparameters (alpha, learning rate) dramatically affect results

---

## Reflection for MPS439 Students

Based on what you implemented today:

1. **Why does Ridge lead to coefficient shrinkage?**
   
   Answer: _________________________________________________

2. **What's the role of the learning rate in gradient descent?**
   
   Answer: _________________________________________________

3. **How would you explain the difference between Ridge and Lasso's regularization to someone?**
   
   Answer: _________________________________________________

4. **For your assignment, which implementation option interests you most (A, B, or C) and why?**
   
   Answer: _________________________________________________

---

**MPS439 Section Complete!** 🎉

You now have the mathematical and coding foundation to implement regularized regression from scratch. This deeper understanding will help you in the assignment and in future machine learning work!

<!-- COURSE_FEEDBACK_QR:START -->
---

## 30-second feedback

What would help you learn better next time?

Scan to share anonymous feedback on today's lab.

![Feedback QR code for Lesson 03 lab](./feedback-qr.png){fig-align="center" width="180px"}
<!-- COURSE_FEEDBACK_QR:END -->
