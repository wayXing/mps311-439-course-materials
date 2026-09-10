---
pagetitle: "Lab 2: Linear Regression"
---

# Lab 2: Linear Regression
## MPS311/439 - Machine Learning
**Lesson 2 Lab Session | Duration: 50 minutes**

---

## Introduction

Today, you'll build your first **predictive model** from scratch. By the end of this session, you'll be able to predict diabetes progression in patients using real medical data.

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

---

## Part 0: Setup and Data Exploration (5 minutes)

### Goal
Load the diabetes dataset and understand what we're working with.

### Background
The diabetes dataset contains measurements from 442 patients. Our task is to predict **disease progression** one year after baseline (a numerical value indicating severity). We have 10 features: age, sex, BMI, blood pressure, and six blood serum measurements.

### Your Tasks

**Task 0.1:** Import libraries and load the data

```python
# TODO: Import numpy, matplotlib.pyplot, and load_diabetes from sklearn.datasets



# TODO: Load the diabetes dataset using load_diabetes()


# TODO: Extract features (X) and target (y) from the dataset
X = 
y = 
```

**Task 0.2:** Explore the data structure

```python
# TODO: Print the shape of X and y



# TODO: Print the feature names


# TODO: Print the first 500 characters of the dataset description

```

**Task 0.3:** Create a scatter plot of BMI vs disease progression

```python
# TODO: Extract the BMI column (feature index 2) from X
bmi = 

# TODO: Create a figure with size (8, 5)


# TODO: Create a scatter plot of BMI vs target (y)
# Use alpha=0.5 and color='steelblue'


# TODO: Add labels and title



# TODO: Add a grid and show the plot


```

**Questions to Answer:**
- How many patients are in the dataset?
- How many features do we have?
- Do you see a relationship between BMI and disease progression?

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

The workflow: Prepare → Create → Fit → Predict → Visualize

### Your Tasks

**Task 1.1:** Prepare the data (reshape for sklearn)

```python
# TODO: Import LinearRegression from sklearn.linear_model


# TODO: Extract BMI and reshape it to 2D array (442, 1)
# sklearn requires 2D arrays, use .reshape(-1, 1)
X_bmi = 

# TODO: Print the shape before and after reshaping


```

**Task 1.2:** Create and train the model

```python
# TODO: Create a LinearRegression model


# TODO: Fit the model using X_bmi and y


# TODO: Print the learned weight (slope) and bias (intercept)
print("Weight (w):", )
print("Bias (b):", )
```

**Task 1.3:** Make predictions

```python
# TODO: Use the trained model to predict y values for all X_bmi


# TODO: Print predictions for the first 3 patients


```

**Task 1.4:** Visualize the fitted line

```python
# TODO: Create a figure with size (10, 6)


# TODO: Plot scatter of actual data (X_bmi vs y)


# TODO: Plot the fitted line (X_bmi vs y_pred)
# Use color='red' and linewidth=2


# TODO: Add labels, title, legend, and grid




```

**Questions to Answer:**
- What is the weight? What does it mean?
- What is the bias? What does it represent?
- Does the line fit the data well?

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
# TODO: Calculate residuals (errors): y_true - y_pred
residuals = 

# TODO: Square the residuals


# TODO: Calculate the mean of squared residuals (MSE)


# TODO: Print MSE and RMSE (square root of MSE)


```

**Task 2.2:** Calculate metrics using sklearn

```python
# TODO: Import mean_squared_error and r2_score from sklearn.metrics


# TODO: Calculate MSE using sklearn function


# TODO: Calculate R² score


# TODO: Print both metrics


```

**Task 2.3:** Try a different feature (blood pressure)

```python
# TODO: Extract blood pressure (feature index 3) and reshape


# TODO: Create and fit a new model


# TODO: Make predictions


# TODO: Calculate MSE and R² for this model


# TODO: Print comparison with BMI model


```

**Questions to Answer:**
- Which feature is a better predictor: BMI or blood pressure?
- What does an R² of 0.03 mean?
- Why isn't R² close to 1.0?

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
# TODO: Import train_test_split from sklearn.model_selection


# TODO: Select 3 features: BMI (2), blood pressure (3), and feature 8
X_multi = 

# TODO: Split data into train (80%) and test (20%) sets
# Use random_state=42 for reproducibility
X_train, X_test, y_train, y_test = 

# TODO: Print the sizes of train and test sets


```

**Task 3.2:** Train model on training data

```python
# TODO: Create a new LinearRegression model


# TODO: Fit on TRAINING data only (X_train, y_train)


# TODO: Print the learned weights and bias


```

**Task 3.3:** Evaluate on both train and test sets

```python
# TODO: Make predictions on training set


# TODO: Make predictions on test set


# TODO: Calculate MSE for training set


# TODO: Calculate MSE for test set


# TODO: Calculate R² for training set


# TODO: Calculate R² for test set


# TODO: Print all metrics in a formatted way


```

**Task 3.4:** Visualize predictions

```python
# TODO: Create a figure with 2 subplots side by side


# Subplot 1: Training predictions
# TODO: Scatter plot of y_train vs y_train_pred


# TODO: Plot perfect prediction line (y=y)


# TODO: Add labels and title


# Subplot 2: Test predictions  
# TODO: Scatter plot of y_test vs y_test_pred


# TODO: Plot perfect prediction line


# TODO: Add labels and title


# TODO: Show the plot

```

**Questions to Answer:**
- Is test MSE higher or lower than training MSE?
- Is this expected? Why?
- How much better is the 3-feature model compared to single feature?

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
# TODO: Write a loop to test all 10 features
# For each feature:
#   1. Extract and reshape
#   2. Split into train/test
#   3. Fit model
#   4. Calculate test MSE
#   5. Track which feature gives lowest MSE

# Your code here:




# TODO: Print which feature is best

```

**Challenge 4.2:** Use ALL features

```python
# TODO: Use all 10 features (X directly, no slicing needed)


# TODO: Split into train/test


# TODO: Train and evaluate


# TODO: Compare with 3-feature model


```

**Challenge 4.3:** Beat the benchmark
Can you get test R² > 0.50 using any combination of features?

```python
# Your experiments here:




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

**Next week:** Feature engineering and regularization (Ridge & Lasso)

**Reflection Questions:**
1. Why is train/test split crucial?
2. Does more features always mean better performance?
3. What would happen if we trained and tested on the same data?

---

## Advanced Section (MPS439 Students Only)

Continue to implement linear regression from scratch using:
1. The Normal Equation
2. Gradient Descent

See the advanced section below.

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
    # TODO: Get number of samples
    n = 
    
    # TODO: Add column of ones to X for bias
    X_with_bias = 
    
    # TODO: Compute X^T X (use @ or np.dot)
    XtX = 
    
    # TODO: Compute inverse of X^T X
    XtX_inv = 
    
    # TODO: Compute X^T y
    Xty = 
    
    # TODO: Compute final weights: (X^T X)^-1 X^T y
    w = 
    
    return w

# TODO: Test on 3-feature model
X_multi = 
w_normal = 

# TODO: Print results and compare with sklearn

```

**Task 5A.2:** Make predictions with your implementation

```python
def predict_normal(X, w):
    """Make predictions using normal equation weights"""
    # TODO: Add bias column to X
    
    
    # TODO: Compute predictions: X @ w
    
    
    return 

# TODO: Make predictions


# TODO: Calculate MSE and compare with sklearn

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
def gradient_descent(X, y, learning_rate=0.01, iterations=1000):
    """
    Perform gradient descent to learn weights.
    Returns: w (weights), losses (list of MSE per iteration)
    """
    n, p = X.shape
    
    # TODO: Add bias column
    X_with_bias = 
    
    # TODO: Initialize weights to zero
    w = 
    
    # TODO: Create empty list to track losses
    losses = []
    
    # TODO: Loop for 'iterations' times
    for i in range(iterations):
        # TODO: Compute predictions
        y_pred = 
        
        # TODO: Compute MSE loss
        loss = 
        
        # TODO: Append loss to losses list
        
        
        # TODO: Compute gradient: (2/n) * X^T @ (y_pred - y)
        gradient = 
        
        # TODO: Update weights: w = w - learning_rate * gradient
        w = 
        
        # TODO: Print progress every 100 iterations
        if (i + 1) % 100 == 0:
            print(f"Iteration {i+1}: Loss = {loss:.2f}")
    
    return w, losses

# TODO: Run gradient descent
w_gd, losses = 

# TODO: Print final weights

```

**Task 5B.2:** Visualize convergence

```python
# TODO: Plot loss vs iterations (2 subplots: full and zoomed)




```

**Task 5B.3:** Compare all three methods

```python
# TODO: Compute MSE for sklearn, normal equation, and gradient descent




# TODO: Print comparison


```

**Task 5B.4:** Experiment with learning rates

```python
# TODO: Try learning rates: [0.01, 0.1, 0.5, 1.0]
# Plot loss curves for each on the same plot




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
