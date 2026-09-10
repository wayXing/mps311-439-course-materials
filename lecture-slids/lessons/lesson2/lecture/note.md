# Lesson 2: Linear Regression
## Predicting Continuous Outcomes with Straight Lines

**MPS311/439 Machine Learning**  
**Dr. Wei Xing**  
**University of Sheffield**  
**Academic year 2026–27**

---

## Table of Contents
1. [Introduction: The Power of Prediction](#1-introduction-the-power-of-prediction)
2. [Defining the Regression Problem](#2-defining-the-regression-problem)
3. [Building Our First Model: The Linear Equation](#3-building-our-first-model-the-linear-equation)
4. [Measuring Error: The Loss Function](#4-measuring-error-the-loss-function)
5. [Training with Calculus: Single Feature Case](#5-training-with-calculus-single-feature-case)
6. [Scaling Up: Multiple Linear Regression](#6-scaling-up-multiple-linear-regression)
7. [Implementation in Python](#7-implementation-in-python)
8. [Summary and Next Steps](#8-summary-and-next-steps)

---

## 1. Introduction: The Power of Prediction

### The Fundamental Question

How does Rightmove estimate a house's price without anyone visiting it? How can Netflix predict what show you'll enjoy next? How do businesses forecast quarterly sales? The answer lies in one of the most fundamental concepts in machine learning: **<span style="color: #2E86AB; font-weight: bold;">prediction</span>**.

### Real-World Applications

Prediction systems are embedded throughout modern life:

- **🏠 Real Estate:** Automated property valuations
- **🚗 Transportation:** Uber/Lyft fare estimation
- **💰 Finance:** Stock price forecasting, credit risk assessment
- **⚡ Energy:** Demand forecasting for power grids
- **🎓 Education:** Student performance prediction
- **🏥 Healthcare:** Disease progression modeling

All of these tasks share a common structure: we want to predict a **<span style="color: #E63946; font-weight: bold;">numerical value</span>** based on some available information.

---

## 2. Defining the Regression Problem

### Core Terminology

Let's establish our foundational vocabulary:

> **<span style="color: #2E86AB; font-weight: bold;">Features (X):</span>** The input information we have available. These are the measurements or attributes we'll use to make predictions.

> **<span style="color: #E63946; font-weight: bold;">Target (y):</span>** The numerical value we want to predict. This is also called the "dependent variable" or "output."

> **<span style="color: #06A77D; font-weight: bold;">Regression:</span>** The task of predicting a continuous numerical value (as opposed to classification, where we predict categories).

### Example: House Price Prediction

Let's work with a concrete example throughout these notes:

| House ID | Size (sq ft) | Bedrooms | Age (years) | **Price (£)** |
|----------|--------------|----------|-------------|---------------|
| 1        | 1,200        | 2        | 10          | **185,000**   |
| 2        | 1,800        | 3        | 5           | **265,000**   |
| 3        | 2,400        | 4        | 15          | **320,000**   |
| 4        | 950          | 1        | 25          | **145,000**   |

- **Features:** Size, Bedrooms, Age
- **Target:** Price (what we want to predict)

### The Learning Framework

Our goal is to learn a **<span style="color: #F77F00; font-weight: bold;">model</span>** (function) that maps features to predictions:

```
f: X → y
```

This function should work not just on houses we've seen before, but on **new, unseen houses** as well. This ability to work on new data is called **<span style="color: #2E86AB; font-weight: bold;">generalization</span>**.

---

## 3. Building Our First Model: The Linear Equation

### Visual Intuition

Let's start simple. Suppose we only use **one feature**: the size of the house. If we plot our data:

```
Price (£)
   │
340k│                              ●
   │
280k│                    ●
   │         ●
220k│                          
   │   ●         
160k│  
   │
100k│
   └─────────────────────────────────── Size (sq ft)
      800   1200   1600   2000   2400
```

What pattern do you see? The points roughly follow an **upward trend**. The simplest way to capture this pattern is with a **straight line**.

### The Model Equation

Remember from school mathematics: the equation of a line is **y = mx + c**, where:
- m is the slope
- c is the y-intercept

In machine learning, we use slightly different notation:

$$\boxed{\hat{y} = w_1 x_1 + w_0}$$

Let's break down each component:

| Symbol | Name | Interpretation | School Math Equivalent |
|--------|------|----------------|------------------------|
| $\hat{y}$ | **Predicted value** | Our model's estimate (y-hat) | y |
| $x_1$ | **Feature** | Input value (e.g., house size) | x |
| $w_1$ | **Weight** | How much influence the feature has | m (slope) |
| $w_0$ | **Bias** | Baseline prediction when feature = 0 | c (intercept) |

> **<span style="color: #E63946; font-weight: bold;">Key Insight:</span>** The values $w_0$ and $w_1$ are called **parameters**. These are the numbers our model needs to **learn** from the data.

### Interpreting the Parameters

Let's say we learn that $w_1 = 120$ and $w_0 = 40,000$ for house prices in pounds and size in square feet:

$$\hat{y} = 120x_1 + 40,000$$

**What does this mean?**

- **$w_1 = 120$:** For every additional square foot, the price increases by £120
- **$w_0 = 40,000$:** A hypothetical house of 0 sq ft would be worth £40,000 (this represents baseline costs like land, permissions, etc.)

**Example prediction:**  
For a 1,500 sq ft house:
$$\hat{y} = 120(1500) + 40,000 = 180,000 + 40,000 = £220,000$$

---

## 4. Measuring Error: The Loss Function

### The Challenge

We could draw infinitely many different lines through our data points:

```
Price
   │
   │        Line A (steep)
   │       ╱
   │   ●  ╱    ● 
   │  ╱● ╱   ●        Line B (gentle)
   │ ╱  ╱ ● ╱
   │╱  ╱  ╱
   │  ╱──╱─────────
   │ ╱  
   └─────────────── Size
```

**Question:** Which line is "best"? How do we measure how well a line fits our data?

### Measuring Individual Errors

For each data point $i$, we can measure the **<span style="color: #E63946; font-weight: bold;">residual</span>** (error):

$$\text{Error}_i = y_i - \hat{y}_i$$

Where:
- $y_i$ is the **actual** price
- $\hat{y}_i = w_1 x_i + w_0$ is our **predicted** price

```
Price
   │
   │    Actual point (yi)
   │         ●
   │         │← Residual (error)
   │         │
   │         ×  Prediction (ŷi)
   │        ╱
   │       ╱ Line
   │      ╱
   └────────────── Size
```

### The Mean Squared Error (MSE)

We can't just add up all the errors because positive and negative errors would cancel out. Instead, we **square** each error and then average:

$$\boxed{J(w_0, w_1) = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 = \frac{1}{n} \sum_{i=1}^{n} (y_i - (w_1 x_i + w_0))^2}$$

Where:
- **$J$** is the loss function (also called cost function)
- **$n$** is the number of data points
- The **summation** goes over all training examples

> **<span style="color: #2E86AB; font-weight: bold;">Why square the errors?</span>**
> 1. Squaring ensures all errors are positive (no cancellation)
> 2. Larger errors get penalized more heavily (squared penalty)
> 3. The resulting function is smooth and differentiable (important for optimization)
> 4. It has a unique minimum (convex function)

### Example Calculation

Suppose we have 3 houses and our current parameters are $w_0 = 50,000$, $w_1 = 100$:

| House | Size ($x_i$) | Actual Price ($y_i$) | Predicted ($\hat{y}_i$) | Error $(y_i - \hat{y}_i)$ | Squared Error |
|-------|--------------|----------------------|-------------------------|---------------------------|---------------|
| 1     | 1,200        | 185,000              | 170,000                 | 15,000                    | 225,000,000   |
| 2     | 1,800        | 265,000              | 230,000                 | 35,000                    | 1,225,000,000 |
| 3     | 2,400        | 320,000              | 290,000                 | 30,000                    | 900,000,000   |

$$J = \frac{225,000,000 + 1,225,000,000 + 900,000,000}{3} = 783,333,333$$

Our goal during **<span style="color: #06A77D; font-weight: bold;">training</span>** is to find the values of $w_0$ and $w_1$ that **minimize** this loss $J$.

---

## 5. Training with Calculus: Single Feature Case

### The Optimization Problem

**<span style="color: #F77F00; font-weight: bold;">Training</span>** means finding the optimal parameters:

$$\min_{w_0, w_1} J(w_0, w_1)$$

From calculus, we know that minima occur where the derivative equals zero. Since $J$ depends on two parameters, we need **partial derivatives**.

### Computing the Partial Derivatives

#### Step 1: Derivative with respect to $w_0$ (bias)

Starting with:
$$J(w_0, w_1) = \frac{1}{n} \sum_{i=1}^{n} (y_i - (w_1 x_i + w_0))^2$$

Taking the partial derivative with respect to $w_0$:

$$\frac{\partial J}{\partial w_0} = \frac{1}{n} \sum_{i=1}^{n} 2(y_i - (w_1 x_i + w_0)) \cdot (-1)$$

$$\boxed{\frac{\partial J}{\partial w_0} = \frac{2}{n} \sum_{i=1}^{n} -(y_i - (w_1 x_i + w_0))}$$

#### Step 2: Derivative with respect to $w_1$ (weight)

Taking the partial derivative with respect to $w_1$:

$$\frac{\partial J}{\partial w_1} = \frac{1}{n} \sum_{i=1}^{n} 2(y_i - (w_1 x_i + w_0)) \cdot (-x_i)$$

$$\boxed{\frac{\partial J}{\partial w_1} = \frac{2}{n} \sum_{i=1}^{n} -x_i(y_i - (w_1 x_i + w_0))}$$

### Solving for Optimal Parameters

Setting both partial derivatives to zero:

$$\frac{\partial J}{\partial w_0} = 0 \quad \text{and} \quad \frac{\partial J}{\partial w_1} = 0$$

This gives us a **system of two linear equations** with two unknowns. We can solve this system algebraically to find $w_0^*$ and $w_1^*$ (the optimal values).

### The Limitation

This approach works beautifully for **one feature**. But what if we have:
- 10 features? → 10 equations with 10 unknowns
- 100 features? → 100 equations with 100 unknowns
- 1,000 features? → Computationally prohibitive!

**We need a better approach.** Enter: **<span style="color: #E63946; font-weight: bold;">Linear Algebra</span>**.

---

## 6. Scaling Up: Multiple Linear Regression

### From Scalars to Vectors

Instead of working with individual numbers, we'll package everything into vectors and matrices. This allows us to handle any number of features elegantly.

### Notation for Multiple Features

Suppose we have $p$ features for each house:

$$\hat{y} = w_0 + w_1 x_1 + w_2 x_2 + \cdots + w_p x_p$$

We can write this more compactly using **vector notation**:

**Weight vector:**
$$\mathbf{w} = \begin{bmatrix} w_0 \\ w_1 \\ w_2 \\ \vdots \\ w_p \end{bmatrix}$$

**Feature vector** (note the 1 for the bias term):
$$\mathbf{x} = \begin{bmatrix} 1 \\ x_1 \\ x_2 \\ \vdots \\ x_p \end{bmatrix}$$

Now our prediction is simply a **dot product**:

$$\boxed{\hat{y} = \mathbf{w}^T \mathbf{x}}$$

### The Design Matrix

For $n$ training examples, we organize all features into a matrix **$\mathbf{X}$** (called the **<span style="color: #2E86AB; font-weight: bold;">design matrix</span>**):

$$\mathbf{X} = \begin{bmatrix} 
1 & x_{11} & x_{12} & \cdots & x_{1p} \\
1 & x_{21} & x_{22} & \cdots & x_{2p} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & x_{n1} & x_{n2} & \cdots & x_{np}
\end{bmatrix}$$

Each row is one training example (with a 1 prepended for the bias).

The target values for all examples go into a vector:

$$\mathbf{y} = \begin{bmatrix} y_1 \\ y_2 \\ \vdots \\ y_n \end{bmatrix}$$

### Deriving the Normal Equation

This is the mathematical heart of linear regression. Let's work through it step by step.

#### Step 1: Express the Loss in Matrix Form

The loss function for all examples can be written as:

$$J(\mathbf{w}) = (\mathbf{y} - \mathbf{Xw})^T (\mathbf{y} - \mathbf{Xw})$$

This is equivalent to summing the squared errors for all $n$ data points.

#### Step 2: Expand the Expression

Using the rules of matrix algebra:

$$J(\mathbf{w}) = \mathbf{y}^T\mathbf{y} - 2\mathbf{y}^T\mathbf{Xw} + \mathbf{w}^T\mathbf{X}^T\mathbf{Xw}$$

**Derivation details:**
- $(\mathbf{y} - \mathbf{Xw})^T (\mathbf{y} - \mathbf{Xw})$
- $= \mathbf{y}^T\mathbf{y} - \mathbf{y}^T\mathbf{Xw} - (\mathbf{Xw})^T\mathbf{y} + (\mathbf{Xw})^T\mathbf{Xw}$
- Since $\mathbf{y}^T\mathbf{Xw}$ is a scalar, it equals its transpose: $\mathbf{y}^T\mathbf{Xw} = (\mathbf{Xw})^T\mathbf{y}$
- Therefore: $\mathbf{y}^T\mathbf{y} - 2\mathbf{y}^T\mathbf{Xw} + \mathbf{w}^T\mathbf{X}^T\mathbf{Xw}$

#### Step 3: Take the Gradient

The gradient of $J$ with respect to the vector $\mathbf{w}$ is:

$$\nabla_{\mathbf{w}} J(\mathbf{w}) = -2\mathbf{X}^T\mathbf{y} + 2\mathbf{X}^T\mathbf{Xw}$$

**Using matrix calculus rules:**
- $\frac{\partial}{\partial \mathbf{w}}(\mathbf{y}^T\mathbf{y}) = \mathbf{0}$ (no dependence on $\mathbf{w}$)
- $\frac{\partial}{\partial \mathbf{w}}(-2\mathbf{y}^T\mathbf{Xw}) = -2\mathbf{X}^T\mathbf{y}$
- $\frac{\partial}{\partial \mathbf{w}}(\mathbf{w}^T\mathbf{X}^T\mathbf{Xw}) = 2\mathbf{X}^T\mathbf{Xw}$

#### Step 4: Set Gradient to Zero and Solve

For the minimum, we set the gradient to zero:

$$-2\mathbf{X}^T\mathbf{y} + 2\mathbf{X}^T\mathbf{Xw} = \mathbf{0}$$

$$\mathbf{X}^T\mathbf{Xw} = \mathbf{X}^T\mathbf{y}$$

Multiplying both sides by $(\mathbf{X}^T\mathbf{X})^{-1}$:

$$\boxed{\mathbf{\hat{w}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}}$$

### 🎉 The Normal Equation

This is one of the most important equations in machine learning! It gives us the **closed-form solution** for the optimal weights.

**Key properties:**
- ✅ Works for **any number of features**
- ✅ Gives the **exact optimal solution** (not an approximation)
- ✅ Single computation (no iterative process needed)
- ⚠️ Requires matrix inversion (can be slow for very large datasets)
- ⚠️ Requires $\mathbf{X}^T\mathbf{X}$ to be invertible

### Example with Multiple Features

Suppose we have data on houses:

$$\mathbf{X} = \begin{bmatrix} 
1 & 1200 & 2 \\
1 & 1800 & 3 \\
1 & 2400 & 4 \\
1 & 950 & 1
\end{bmatrix}, \quad
\mathbf{y} = \begin{bmatrix} 
185000 \\
265000 \\
320000 \\
145000
\end{bmatrix}$$

Computing (conceptually):

1. $\mathbf{X}^T\mathbf{X}$ gives a $(3 \times 3)$ matrix
2. $\mathbf{X}^T\mathbf{y}$ gives a $(3 \times 1)$ vector
3. Invert $\mathbf{X}^T\mathbf{X}$ and multiply to get $\mathbf{\hat{w}}$

The result might be:
$$\mathbf{\hat{w}} = \begin{bmatrix} 45000 \\ 110 \\ 8000 \end{bmatrix}$$

Interpretation:
- Base price: £45,000
- Each sq ft adds: £110
- Each bedroom adds: £8,000

---

## 7. Implementation in Python

### Using scikit-learn

The beauty of modern machine learning libraries is that they implement the Normal Equation (and more efficient alternatives) for you. Here's how simple it is:

```python
from sklearn.linear_model import LinearRegression
import numpy as np

# Our data
X = np.array([
    [1200, 2],   # Size, Bedrooms
    [1800, 3],
    [2400, 4],
    [950, 1]
])

y = np.array([185000, 265000, 320000, 145000])

# Create and train the model
model = LinearRegression()
model.fit(X, y)  # This computes the Normal Equation internally

# Get the learned parameters
print(f"Weights (w1, w2): {model.coef_}")
print(f"Bias (w0): {model.intercept_}")

# Make a prediction for a new house: 1500 sq ft, 2 bedrooms
new_house = np.array([[1500, 2]])
predicted_price = model.predict(new_house)
print(f"Predicted price: £{predicted_price[0]:,.0f}")
```

**Output might look like:**
```
Weights (w1, w2): [110.5  8234.2]
Bias (w0): 42156.8
Predicted price: £216,009
```

### Understanding the Workflow

Every supervised machine learning project follows this pattern:

```
1. PREPARE DATA
   ├─ Load dataset
   ├─ Split into features (X) and target (y)
   └─ Split into training and test sets
   
2. CREATE MODEL
   └─ Instantiate the algorithm (e.g., LinearRegression())
   
3. TRAIN MODEL
   └─ Call .fit(X_train, y_train)
   
4. EVALUATE MODEL
   ├─ Make predictions on test set
   └─ Compute metrics (e.g., MSE, R²)
   
5. USE MODEL
   └─ Make predictions on new, real data
```

### Training vs. Testing

> **<span style="color: #E63946; font-weight: bold;">Critical Concept:</span>** We **train** on one set of data and **test** on a different set.

**Why?** To check if our model **generalizes** to new data, or if it has simply **memorized** the training data (**<span style="color: #F77F00; font-weight: bold;">overfitting</span>**).

```python
from sklearn.model_selection import train_test_split

# Split data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train on training set only
model.fit(X_train, y_train)

# Evaluate on both sets
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"Training R² score: {train_score:.3f}")
print(f"Testing R² score: {test_score:.3f}")
```

If training score is much higher than testing score → **overfitting**! (We'll address this next week with regularization)

#### Other Evaluation Metrics

There are other evaluation metrics that are more robust to outliers and can be used to evaluate the performance of the model.

- Mean Absolute Error (MAE): $$MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$
- Mean Squared Error (MSE): $$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$
- Root Mean Squared Error (RMSE): $$RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$
- R² score: $$R² = 1 - \frac{SSR}{SST}$$ where $SSR=\sum_{i=1}^{n} (y_i - \hat{y}_i)^2$ is the sum of squared residuals and $SST=\sum_{i=1}^{n} (y_i - \bar{y})^2$ is the total sum of squares, $\bar{y}$ is the mean of the target values. 
R² score is a good measure because it is a normalized measure that is always between 0 and 1. 0 means the model is no better than the mean of the target values, 1 means the model is perfect. 

Different metrics are more appropriate for different types of data and models. In General, MSE is the most common and most interpretable metric.

---

## 8. Summary and Next Steps

### What We Learned Today

1. **<span style="color: #2E86AB; font-weight: bold;">Regression</span>** is the task of predicting continuous numerical values

2. **<span style="color: #06A77D; font-weight: bold;">Linear models</span>** represent the relationship as: $\hat{y} = w_1 x_1 + w_2 x_2 + \cdots + w_p x_p + w_0$

3. **<span style="color: #E63946; font-weight: bold;">Mean Squared Error (MSE)</span>** measures how well our model fits the data

4. **<span style="color: #F77F00; font-weight: bold;">Training</span>** finds the optimal parameters by minimizing the loss function

5. The **<span style="color: #2E86AB; font-weight: bold;">Normal Equation</span>** $\mathbf{\hat{w}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$ gives us the closed-form solution

6. We must evaluate on a **test set** to check **generalization**

### The Universal ML Framework

Today's lecture revealed a pattern you'll see repeatedly:

```
MODEL → LOSS → OPTIMIZATION
```

1. **Model:** Choose an architecture (e.g., linear function)
2. **Loss:** Define how to measure error (e.g., MSE)
3. **Optimization:** Find parameters that minimize loss (e.g., Normal Equation)

This framework applies to almost every supervised learning algorithm!

### Looking Ahead: Linear Regression Plus

Our linear model is powerful but makes strong assumptions:
- ❓ What if relationships aren't actually linear?
- ❓ What if we have too many features?
- ❓ How do we prevent overfitting?

**Next lecture (Lesson 3):** We'll extend our model with:
- **Feature engineering** to capture non-linear relationships
- **Regularization** (Ridge & Lasso) to prevent overfitting
- **Cross-validation** to tune hyperparameters

### Key Takeaways

> 💡 **Linear regression is simple but powerful**  
> It's often the first model to try on a new problem

> 💡 **The Normal Equation is elegant**  
> It shows that linear regression has a beautiful mathematical foundation

> 💡 **Always test on unseen data**  
> Training performance alone doesn't tell you if your model will work in the real world

> 💡 **This framework generalizes**  
> The pattern of Model → Loss → Optimization applies far beyond linear regression

---

## Additional Resources

**For deeper understanding:**
- 📘 Pattern Recognition and Machine Learning (Bishop) - Chapter 3
- 📘 The Elements of Statistical Learning - Chapter 3
- 🎥 StatQuest: Linear Regression (YouTube)

**For practice:**
- Try implementing the Normal Equation from scratch in NumPy
- Experiment with the `sklearn` Boston Housing or California Housing datasets
- Compare predictions with different numbers of features

**For next week:**
- Review polynomial functions (we'll use them for feature engineering)
- Think about: when might a straight line be a poor fit for data?

---

## Practice Problems

1. **Conceptual:** Why do we square the errors in MSE instead of just taking absolute values?

2. **Mathematical:** Given data points $(1, 2)$, $(2, 4)$, $(3, 5)$, use calculus to find optimal $w_0$ and $w_1$ for the model $\hat{y} = w_1x + w_0$

3. **Computational:** Load a dataset from sklearn (e.g., diabetes dataset), fit a linear regression model, and compute the MSE on both training and test sets

4. **Interpretation:** If you have a model $\hat{y} = 50x_1 + 20x_2 + 100$ for predicting exam scores, where $x_1$ is hours studied and $x_2$ is previous test score, interpret what each coefficient means

---

<!-- add a quote from George Box here -->
<div class="p-4 bg-gray-50 rounded-lg">

"All models are wrong, but some are useful."

<div class="text-lg mt-4 text-right">
— George Box, Statistician
</div>

</div>


**Remember:** Linear regression is the foundation. Master it, and everything else becomes easier!



## Extra for MPS439 students

### Alternative Optimization: Gradient Descent

#### When the Normal Equation Isn't Ideal

The Normal Equation $\mathbf{\hat{w}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$ is elegant, but has limitations:

- **Computational cost:** Matrix inversion is $O(p^3)$ where $p$ is the number of features
- **Memory requirements:** Must compute and store $\mathbf{X}^T\mathbf{X}$ (a $p \times p$ matrix)
- **Numerical stability:** $\mathbf{X}^T\mathbf{X}$ might not be invertible

For large datasets (millions of examples or thousands of features), we need an alternative: **<span style="color: #F77F00; font-weight: bold;">Gradient Descent</span>**.

#### The Gradient Descent Idea

Imagine you're hiking down a mountain in thick fog. You can't see the bottom, but you can feel which direction slopes downward. **Gradient descent** uses the same strategy:

1. Start with random parameter values
2. Compute the gradient (direction of steepest ascent)
3. Take a small step in the **opposite** direction (downhill)
4. Repeat until we reach the minimum

```
Loss J(w)
   │     
   │  ●  Start here (random w)
   │   ╲
   │    ●  Step 1
   │     ╲
   │      ●  Step 2
   │       ╲
   │        ● Step 3
   │         ╲___●  Converge to minimum
   └─────────────────────── w
```

#### The Algorithm

**Initialize:** Start with random weights $\mathbf{w}^{(0)}$

**Repeat until convergence:**

$$\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} - \alpha \nabla_{\mathbf{w}} J(\mathbf{w}^{(t)})$$

Where:
- $\alpha$ is the **<span style="color: #2E86AB; font-weight: bold;">learning rate</span>** (step size, typically 0.001 to 0.1)
- $\nabla_{\mathbf{w}} J$ is the gradient we derived earlier: $-2\mathbf{X}^T\mathbf{y} + 2\mathbf{X}^T\mathbf{Xw}$

**Update rule in detail:**

$$\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} - \alpha \left( -2\mathbf{X}^T\mathbf{y} + 2\mathbf{X}^T\mathbf{X}\mathbf{w}^{(t)} \right)$$

Simplified (absorbing the 2 into $\alpha$):

$$\boxed{\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} + \alpha \mathbf{X}^T(\mathbf{y} - \mathbf{X}\mathbf{w}^{(t)})}$$

#### Choosing the Learning Rate

The learning rate $\alpha$ is crucial:

- **Too large:** We might overshoot the minimum and diverge
- **Too small:** Convergence will be very slow

```
Too large α:              Just right α:           Too small α:
Loss                      Loss                    Loss
 │  ●                      │  ●                     │  ●
 │   ╲╱●                   │   ╲                    │   ╲
 │   ╱ ╲                   │    ╲●                  │    ●
 │  ●   ●                  │     ●                  │     ●
 │   ╲ ╱                   │      ●                 │      ●
 │    ●  Oscillating!      │       ●  Converges     │       ●  Very slow
 └───────── w              └────────── w            └─────────── w
```

#### Normal Equation vs. Gradient Descent

| Aspect | Normal Equation | Gradient Descent |
|--------|----------------|------------------|
| **Speed** | Fast for small $p$ (<10,000) | Fast for large $p$ |
| **Complexity** | $O(p^3)$ | $O(kp^2)$ where $k$ = iterations |
| **Memory** | $O(p^2)$ | $O(p)$ |
| **Invertibility** | Requires $\mathbf{X}^T\mathbf{X}$ invertible | Always works |
| **Implementation** | One-step computation | Iterative process |
| **Hyperparameters** | None | Learning rate $\alpha$ |

> **<span style="color: #06A77D; font-weight: bold;">Practical Tip:</span>** For most standard regression problems with $p < 10,000$ features, use the Normal Equation (what sklearn does by default). For very large-scale problems, gradient descent is essential.

#### Quick Python Example

```python
import numpy as np

# Simple gradient descent implementation
def gradient_descent(X, y, alpha=0.01, iterations=1000):
    n, p = X.shape
    w = np.zeros(p)  # Initialize weights to zero
    
    for i in range(iterations):
        # Compute predictions
        y_pred = X @ w
        
        # Compute gradient
        gradient = -2/n * X.T @ (y - y_pred)
        
        # Update weights
        w = w - alpha * gradient
        
        # Optionally: compute and print loss every 100 iterations
        if i % 100 == 0:
            loss = np.mean((y - y_pred)**2)
            print(f"Iteration {i}: Loss = {loss:.2f}")
    
    return w

# Use it
w_optimal = gradient_descent(X, y, alpha=0.01, iterations=1000)
```

**Note:** In practice, you'd use optimized implementations from libraries like sklearn (which can automatically choose between methods) or use `SGDRegressor` for explicit gradient-based optimization.

---

*End of Lesson 2 Lecture Notes*  
*MPS311/439 Machine Learning · Dr. Wei Xing · University of Sheffield · 2026–27*
