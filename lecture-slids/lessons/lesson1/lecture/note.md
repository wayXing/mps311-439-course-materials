# Lesson 1: Welcome to Machine Learning
## Course Roadmap, Tools, and Your First ML Workflow

**MPS311/439 Machine Learning**  
**Dr. Wei Xing**  
**University of Sheffield**  
**Academic year 2026–27**

---

## Learning Outcomes

By the end of this lecture, you will be able to:

### Core Learning Outcomes (MPS311 - Required for all students)

- explain the difference between traditional programming and machine learning;
- identify features, targets, training data, and test data;
- describe the four-stage machine-learning workflow;
- access the course tools and prepare for the first lab;
- use an AI assistant to support learning while checking its suggestions.

### Advanced Learning Outcomes (MPS439 - Additional depth)

- connect the course workflow to a complete model-development cycle;
- distinguish model fitting from model evaluation;
- explain why generalisation to unseen data is the central goal of machine learning.

---

## Table of Contents

1. [What Machine Learning Is](#1-what-machine-learning-is)
2. [The Machine-Learning Workflow](#2-the-machine-learning-workflow)
3. [How the Course Is Organised](#3-how-the-course-is-organised)
4. [Tools and Working Practices](#4-tools-and-working-practices)
5. [Using AI Responsibly](#5-using-ai-responsibly)
6. [Preparing for the First Lab](#6-preparing-for-the-first-lab)
7. [Summary and Next Steps](#7-summary-and-next-steps)

---

## 1. What Machine Learning Is

Machine learning is already part of everyday life. Recommendation systems suggest films and music, email services filter spam, navigation systems estimate travel time, and online shops predict which products may be relevant.

Traditional programming starts with explicit rules:

$$\text{input} + \text{rules} \rightarrow \text{output}$$

Machine learning changes the role of the rules. We provide examples containing inputs and known outputs, and the algorithm learns a model from those examples:

$$\text{input} + \text{known output} \rightarrow \text{learned model}$$

The learned model can then make predictions for new inputs.

### Features and Targets

- **Features** are the input measurements used to make a prediction.
- **Target** is the value or category that the model should predict.
- **Model** is the learned relationship between features and the target.
- **Prediction** is the model's output for a new example.

For a house-price problem, size, location, and number of bedrooms may be features; the sale price is the target.

---

## 2. The Machine-Learning Workflow

Most work in this course follows four stages.

### 2.1 Define

State the question clearly. Decide what should be predicted and what evidence is available.

### 2.2 Collect and Understand

Load the data, inspect its variables, identify missing or implausible values, and explore important patterns.

### 2.3 Model and Train

Choose an appropriate algorithm and fit it using the training data.

### 2.4 Validate

Evaluate the fitted model on unseen data. A model is useful only when it generalises beyond the examples used for training.

This cycle is iterative. Results from validation often reveal that we need to revise the features, data preparation, or model choice.

---

## 3. How the Course Is Organised

The course progresses from fundamental predictive models to methods that learn more complex structure:

- **Lessons 1–3:** foundations, linear regression, feature engineering, and regularisation;
- **Lessons 4–6:** classification methods and decision trees;
- **Lessons 7–8:** unsupervised learning with PCA and clustering;
- **Lessons 9–10:** neural networks and convolutional neural networks.

Each lesson combines a lecture with a practical lab. The lecture develops the core ideas; the lab turns those ideas into a reproducible workflow using Python.

MPS311 students should focus first on the core learning outcomes. MPS439 students complete the same core work and then explore the additional mathematical or implementation detail identified in each week's materials.

---

## 4. Tools and Working Practices

The course uses:

- **Google Colab or Jupyter Notebook** for running Python;
- **Python** for data analysis and modelling;
- **pandas** for structured data;
- **matplotlib** for visualisation;
- **scikit-learn** for standard machine-learning workflows;
- **Keras/TensorFlow** for neural-network topics later in the course.

Good working practice matters as much as individual commands:

1. keep the data and notebook together;
2. run cells in order and check their output;
3. use meaningful variable names;
4. record observations as well as code;
5. save a working copy before making major changes;
6. verify results rather than assuming that code is correct because it runs.

---

## 5. Using AI Responsibly

AI assistants are welcome as learning tools. They are useful for explaining error messages, suggesting debugging steps, comparing alternative implementations, and asking follow-up questions about concepts.

A productive workflow is:

1. describe the task and include the exact error or relevant code;
2. ask for an explanation, not only a replacement answer;
3. test the suggestion in the notebook;
4. inspect whether the result is plausible;
5. rewrite the explanation in your own words.

AI output may be incomplete or incorrect. You remain responsible for the code you submit and the conclusions you draw.

---

## 6. Preparing for the First Lab

Before the lab:

- confirm that you can open Google Colab or Jupyter Notebook;
- download the Lesson 1 worksheet and its data file;
- practise running and re-running a code cell;
- bring a laptop or ensure that you can access a suitable computer;
- note any setup problems so they can be resolved at the start of the session.

The first lab introduces exploratory data analysis. You will load a dataset, inspect its structure and quality, create basic visualisations, and communicate an evidence-based observation.

---

## 7. Summary and Next Steps

The central goal of machine learning is not to memorise the training data. It is to learn a useful relationship that works on new data.

Remember the course workflow:

$$\text{Define} \rightarrow \text{Collect} \rightarrow \text{Model} \rightarrow \text{Validate}$$

Next week introduces **linear regression**, our first predictive model for continuous outcomes.

---

*End of Lesson 1 Lecture Notes*  
*MPS311/439 Machine Learning · Dr. Wei Xing · University of Sheffield · 2026–27*
