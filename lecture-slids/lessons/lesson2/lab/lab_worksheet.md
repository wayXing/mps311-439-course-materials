---
pagetitle: "Lab 2: Linear Regression"
---

# Lab 2: Linear Regression
## MPS311/439 - Machine Learning
**Lesson 2 Lab Session | Duration: 50 minutes**

---

## Introduction

Today, you'll build your first **predictive model** using scikit-learn. By the end of this session, you'll be able to predict diabetes progression in patients using real medical data and judge the evidence for that prediction.

### What You'll Learn

- Load and explore a real medical dataset  
- Build a linear regression model using scikit-learn  
- Make predictions on new patient data  
- Measure how well your model performs  
- Understand why we split data into training and testing sets  
- Compare different features to find the best predictors  

### Before You Start

**Environment:** Use Google Colab or Jupyter Notebook  
**Dataset:** Diabetes dataset (built into scikit-learn)  
**Required Libraries:** numpy, matplotlib, sklearn (all pre-installed in Colab)

**Remember:** This lab uses a fill-in-the-blanks approach. Complete only the parts marked with `____`; the surrounding code shows the workflow. Focus on what each modelling step does rather than memorising syntax.

---

## Part 0: Setup and Data Exploration (5 minutes)

### Goal
Load the diabetes dataset and understand what we're working with.

### Background
The diabetes dataset contains measurements from 442 patients. Our task is to predict **disease progression** one year after baseline (a numerical value indicating severity). We have 10 features: age, sex, BMI, blood pressure, and six blood serum measurements.

### Your Tasks

**Task 0.1:** Import libraries and load the data

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import ____

diabetes = ____()

# Extract the feature matrix and target vector
X = diabetes.____
y = diabetes.____
```

**Task 0.2:** Explore the data structure

```python
print("X shape:", X.____)
print("y shape:", y.____)
print("\nFeatures:", diabetes.____)
print("\nDescription:")
print(diabetes.____[:500])
```

**Task 0.3:** Create a scatter plot of BMI vs disease progression

```python
bmi = X[:, ____]

plt.figure(figsize=(8, 5))
plt.____(bmi, y, alpha=0.5)
plt.xlabel('BMI')
plt.ylabel('Disease Progression')
plt.title('BMI vs Diabetes Progression')
plt.grid(True, alpha=0.3)
plt.____()
```

**Questions to Answer:**

- How many patients are in the dataset?
- How many features do we have?
- Do you see a relationship between BMI and disease progression?

> Patients: __________  Features: __________
>
> Relationship observed: _______________________________________________

### Hints
- Hint: Use `X[:, 2]` to extract column index 2
- Hint: `plt.scatter(x_data, y_data, alpha=0.5)` creates a scatter plot
- Hint: Feature indices start at 0

### AI Help
- "How do I load the diabetes dataset from sklearn?"
- "How do I extract a single column from a numpy array?"
- "How do I create a scatter plot with matplotlib?"

---

## Part 1: Your First Linear Regression Model (15 minutes)

### Goal
Build a simple linear regression model using just ONE feature (BMI).

### Background
Linear regression finds the line of best fit: **ŷ = w × x + b**

The workflow: Prepare -> Create -> Fit -> Predict -> Visualize

### Your Tasks

**Task 1.1:** Prepare the data (reshape for sklearn)

```python
from sklearn.linear_model import ____

# sklearn expects a 2D feature matrix
X_bmi = X[:, ____].reshape(____, ____)

print("Original shape:", X[:, 2].shape)
print("Reshaped:", X_bmi.____)
```

**Task 1.2:** Create and train the model

```python
model = ____()
model.____(X_bmi, ____)

print("Weight (w):", model.____[0])
print("Bias (b):", model.____)
```

**Task 1.3:** Make predictions

```python
y_pred = model.____(____)

print("First 3 predictions:", y_pred[:____])
print("First 3 actual:", y[:____])
```

**Task 1.4:** Visualize the fitted line

```python
plt.figure(figsize=(10, 6))
plt.____(X_bmi, y, alpha=0.5)
plt.____(X_bmi, y_pred, color='red', linewidth=2)
plt.xlabel('BMI')
plt.ylabel('Disease Progression')
plt.title(f'Fitted Line: y = {model.coef_[0]:.1f} * BMI + {model.intercept_:.1f}')
plt.____()
```

**Questions to Answer:**

- What is the weight? What does it mean?
- What is the bias? What does it represent?
- Does the line fit the data well?

> Weight and interpretation: ___________________________________________
>
> Bias and interpretation: _____________________________________________
>
> Evidence about fit: __________________________________________________

### Hints
- Hint: `.reshape(-1, 1)` converts (442,) to (442, 1)
- Hint: `model.coef_` gives weights, `model.intercept_` gives bias
- Hint: `model.predict(X)` returns predictions
- Hint: Use `plt.scatter()` for points and `plt.plot()` for lines

### AI Help
- "Why does sklearn need reshape(-1, 1)?"
- "How do I fit a LinearRegression model?"
- "How do I access learned parameters from sklearn model?"

---

## Part 2: Measuring Model Performance (10 minutes)

### Goal
Quantify model performance using MSE and R² score.

### Background
- **MSE (Mean Squared Error):** Average of squared errors. Lower is better.
- **R² Score:** Proportion of variance explained. Ranges from -∞ to 1. Higher is better.

### Your Tasks

**Task 2.1:** Calculate MSE manually

```python
residuals = ____ - ____
squared_residuals = residuals ** ____
mse_manual = np.____(squared_residuals)

print("MSE (manual):", mse_manual)
print("RMSE:", np.____(mse_manual))
```

**Task 2.2:** Calculate metrics using sklearn

```python
from sklearn.metrics import mean_squared_error, r2_score

mse_sklearn = ____(y, y_pred)
r2 = ____(y, y_pred)

print("MSE (sklearn):", mse_sklearn)
print("R² score:", r2)
```

**Task 2.3:** Try a different feature (blood pressure)

```python
X_bp = X[:, ____].reshape(-1, 1)

model_bp = ____()
model_bp.____(X_bp, y)
y_pred_bp = model_bp.____(X_bp)

mse_bp = mean_squared_error(y, ____)
r2_bp = r2_score(y, ____)

print("BMI          - MSE:", mse_sklearn, "R²:", r2)
print("Blood Press  - MSE:", mse_bp, "R²:", r2_bp)
```

**Questions to Answer:**

- Which feature is a better predictor: BMI or blood pressure?
- What does an R² of 0.03 mean?
- Why isn't R² close to 1.0?

> Better single feature: _______________________________________________
>
> Interpretation of R²: ________________________________________________
>
> Why performance is limited: _________________________________________

### Hints
- Hint: MSE = np.mean((y - y_pred)**2)
- Hint: Lower MSE = better model
- Hint: R² close to 1 = excellent fit, close to 0 = poor fit
- Hint: Feature index 3 is blood pressure

### AI Help
- "What's the difference between MSE and R² score?"
- "How do I calculate MSE manually in numpy?"
- "How do I use sklearn's mean_squared_error function?"

---

## Part 3: Multiple Features & Train/Test Split (15 minutes)

### Goal
Use multiple features and properly evaluate using train/test split.

### Background
**Why multiple features?** Real predictions depend on multiple factors.  
**Why train/test split?** To check if the model generalizes to new data, not just memorizes training data.

### Your Tasks

**Task 3.1:** Split the data

```python
from sklearn.model_selection import ____

# Select BMI (2), blood pressure (3), and feature 8
X_multi = X[:, [____, ____, ____]]

X_train, X_test, y_train, y_test = train_test_split(
    X_multi, y, test_size=____, random_state=____
)

print("Training set:", X_train.shape[0], "samples")
print("Test set:", X_test.shape[0], "samples")
```

**Task 3.2:** Train model on training data

```python
model_multi = ____()
model_multi.____(____, ____)

print("Weights:", model_multi.____)
print("Bias:", model_multi.____)
```

**Task 3.3:** Evaluate on both train and test sets

```python
y_train_pred = model_multi.____(X_train)
y_test_pred = model_multi.____(X_test)

train_mse = mean_squared_error(____, y_train_pred)
test_mse = mean_squared_error(____, y_test_pred)

train_r2 = r2_score(y_train, ____)
test_r2 = r2_score(y_test, ____)

print("Training - MSE:", train_mse, "R²:", train_r2)
print("Test     - MSE:", test_mse, "R²:", test_r2)
```

**Task 3.4:** Visualize predictions

```python
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(____, ____, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Training Set')

plt.subplot(1, 2, 2)
plt.scatter(____, ____, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Test Set')

plt.tight_layout()
plt.____()
```

**Questions to Answer:**

- Is test MSE higher or lower than training MSE?
- Is this expected? Why?
- How much better is the 3-feature model compared to single feature?

> Train versus test MSE: _______________________________________________
>
> Is this expected? Why? _______________________________________________
>
> Improvement over one feature: _______________________________________

### Hints
- Hint: `train_test_split(X, y, test_size=0.2, random_state=42)`
- Hint: Use `X[:, [2, 3, 8]]` to select multiple columns
- Hint: Always fit on training data only
- Hint: Evaluate on both train and test
- Hint: Test error is usually slightly higher (this is normal!)

### AI Help
- "Why do we split data into train and test?"
- "How do I use train_test_split with 80/20 ratio?"
- "How do I select multiple columns from numpy array?"
- "Why is test error higher than training error?"

---

## Part 4: Exploration & Mini-Challenge (5 minutes)

### Goal
Experiment independently to find the best model!

### Challenges

**Challenge 4.1:** Find the best single feature

```python
best_mse = float('inf')
best_feature_idx = None
mse_list = []

for i in range(____):
    X_feature = X[:, ____].reshape(-1, 1)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_feature, y, test_size=0.2, random_state=42
    )

    model = ____()
    model.____(X_tr, y_tr)
    y_pred = model.____(X_te)

    mse = mean_squared_error(____, ____)
    mse_list.append(mse)

    if mse < best_mse:
        best_mse = ____
        best_feature_idx = ____

print("Best feature:", diabetes.feature_names[____])
print("Test MSE:", best_mse)
```

**Challenge 4.2:** Use ALL features

```python
X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(
    ____, ____, test_size=0.2, random_state=42
)

model_all = ____()
model_all.____(X_train_all, y_train_all)
y_test_pred_all = model_all.____(X_test_all)

mse_all = mean_squared_error(____, y_test_pred_all)
r2_all = r2_score(____, y_test_pred_all)

print("All features - MSE:", mse_all, "R²:", r2_all)
print("3 features   - MSE:", test_mse, "R²:", test_r2)
print("Best single  - MSE:", best_mse)
```

**Challenge 4.3:** Beat the benchmark
Can you get test R² > 0.50 using any combination of features?

```python
# Choose a set of feature indices, then reuse the split-fit-predict-evaluate
# workflow above. Record the test R² before changing the feature set again.
feature_indices = [____]
```

### Hints
- Hint: Loop with `for i in range(10):`
- Hint: Track best with variables: `best_mse = float('inf')`, `best_feature = None`
- Hint: For all features, use `X` directly

### AI Help
- "How do I loop through features and find the best one?"
- "How do I keep track of minimum value in a loop?"

---

## Summary Checklist

By the end of this lab, you should be able to:

- [ ] Load the diabetes dataset and explore it
- [ ] Reshape data for sklearn (using .reshape(-1, 1))
- [ ] Create and fit a LinearRegression model
- [ ] Make predictions using .predict()
- [ ] Calculate MSE and R² score
- [ ] Split data using train_test_split
- [ ] Evaluate on both training and test sets
- [ ] Compare different feature combinations
- [ ] Interpret model weights and performance metrics

### What's Next?

**Next lesson:** Feature engineering and regularization (Ridge & Lasso)

**Reflection Questions:**

1. Why is train/test split crucial?
2. Does more features always mean better performance?
3. What would happen if we trained and tested on the same data?

---

## Advanced Section (MPS439 Students Only)

Continue to implement linear regression from scratch using:

1. The Normal Equation
2. Gradient Descent

```{=latex}
\newpage
```

---

## Part 5A: Implementing the Normal Equation (15 minutes)

### Goal
Implement the mathematical solution: **w = (X^T X)^(-1) X^T y**

### Your Tasks

**Task 5A.1:** Implement normal equation function

```python
def normal_equation(X, y):
    """
    Compute optimal weights using the normal equation.
    Returns: w (array with bias as first element)
    """
    n = X.shape[____]
    
    X_with_bias = np.column_stack([np.____(n), X])
    
    XtX = X_with_bias.____ @ X_with_bias
    
    XtX_inv = np.linalg.____(XtX)
    
    Xty = X_with_bias.____ @ y
    
    w = ____ @ ____
    
    return w

X_multi = X[:, [____, ____, ____]]
w_normal = normal_equation(____, ____)

print("Normal Equation:")
print("Bias:", w_normal[0])
print("Weights:", w_normal[1:])

model_compare = ____()
model_compare.____(X_multi, y)

print("\nsklearn:")
print("Bias:", model_compare.____)
print("Weights:", model_compare.____)
```

**Task 5A.2:** Make predictions with your implementation

```python
def predict_normal(X, w):
    """Make predictions using normal equation weights"""
    n = X.shape[0]
    X_with_bias = np.column_stack([np.____(n), X])
    return X_with_bias @ ____

y_pred_normal = predict_normal(____, ____)
y_pred_sklearn = model_compare.____(X_multi)

mse_normal = np.mean((y - y_pred_normal) ** 2)
mse_sklearn = mean_squared_error(y, ____)

print("MSE (Normal Eq):", mse_normal)
print("MSE (sklearn):", mse_sklearn)
print("Match:", np.allclose(____, ____))
```

### Hints
- Hint: `np.column_stack([np.ones(n), X])` adds bias column
- Hint: Use `np.linalg.inv()` for matrix inverse
- Hint: Use `@` operator for matrix multiplication
- Hint: Weights should match sklearn to ~4 decimal places

### AI Help
- "How do I implement the normal equation in numpy?"
- "What does np.column_stack do?"
- "Why might matrix inversion fail?"

---

## Part 5B: Implementing Gradient Descent (15 minutes)

### Goal
Implement iterative optimization: **w = w - α × gradient**

### Your Tasks

**Task 5B.1:** Implement gradient descent

```python
def gradient_descent(X, y, learning_rate=0.5, iterations=1000):
    """
    Perform gradient descent to learn weights.
    Returns: w (weights), losses (list of MSE per iteration)
    """
    n, p = X.shape
    
    X_with_bias = np.column_stack([np.____(n), X])
    
    w = np.____(p + 1)
    
    losses = []
    
    for i in range(____):
        y_pred = X_with_bias @ ____
        
        loss = np.mean((y - y_pred) ** ____)
        
        losses.____(loss)
        
        gradient = (2/n) * X_with_bias.____ @ (y_pred - y)
        
        w = w - ____ * gradient
        
        if (i + 1) % 200 == 0:
            print(f"Iteration {i+1}: Loss = {loss:.2f}")
    
    return w, losses

w_gd, losses = gradient_descent(
    X_multi, y, learning_rate=____, iterations=____
)

print("Bias:", w_gd[0])
print("Weights:", w_gd[1:])
```

```{=latex}
\newpage
```

**Task 5B.2:** Visualize convergence

```python
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(____)
plt.xlabel('Iteration')
plt.ylabel('MSE Loss')
plt.title('Full Convergence')

plt.subplot(1, 2, 2)
plt.plot(losses[____:])
plt.xlabel('Iteration')
plt.ylabel('MSE Loss')
plt.title('Convergence (zoomed)')

plt.tight_layout()
plt.show()
```

**Task 5B.3:** Compare all three methods

```python
y_pred_gd = predict_normal(X_multi, ____)
mse_gd = np.mean((y - ____) ** 2)

print("Normal Equation  - MSE:", mse_normal)
print("sklearn          - MSE:", mse_sklearn)
print("Gradient Descent - MSE:", mse_gd)
```

**Task 5B.4:** Experiment with learning rates

```python
learning_rates = [0.01, 0.1, 0.5, 1.0]
plt.figure(figsize=(10, 5))

for lr in ____:
    _, losses_lr = gradient_descent(
        X_multi, y, learning_rate=____, iterations=500
    )
    plt.plot(losses_lr, label=f'α = {lr}')

plt.xlabel('Iteration')
plt.ylabel('MSE Loss')
plt.title('Learning Rate Comparison')
plt.legend()
plt.yscale('log')
plt.show()
```

### Hints
- Hint: Start with learning_rate=0.5, iterations=1000
- Hint: Gradient = (2/n) * X^T @ (y_pred - y)
- Hint: If loss increases, reduce learning rate
- Hint: Loss should decrease monotonically

### AI Help
- "How do I implement gradient descent for linear regression?"
- "Why is my gradient descent loss increasing?"
- "How do I choose a good learning rate?"
- "When should I use gradient descent vs normal equation?"

---

## Submission (if required)

Save your completed notebook as: `Lab2_YourName.ipynb`

Include:

- All completed code cells
- Output from all cells
- Answers to reflection questions

**Good luck and have fun with your first ML model!**

<!-- COURSE_FEEDBACK_QR:START -->
---

## 30-second feedback

Scan the code to share anonymous feedback or post a question for this lab. It opens the correct **Lesson 02 lab** record automatically.

![Feedback QR code for Lesson 02 lab](./feedback-qr.png){fig-align="center" width="180px"}
<!-- COURSE_FEEDBACK_QR:END -->
