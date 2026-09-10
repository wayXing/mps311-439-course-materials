# Appendix: Interactive Code Demonstrations for Lesson 4 Lecture
## Logistic Regression - Live Demo Descriptions

This document provides detailed descriptions (NOT code) for each interactive demonstration that will be performed in Google Colab during the Lesson 4 lecture. These descriptions are designed to guide the creation of executable code by another AI agent or instructor.

---

## Demo 1: Load and Explore Dataset

**Purpose**: Introduce the breast cancer dataset and familiarize students with its structure

**Dataset**: Breast Cancer Wisconsin (Diagnostic) dataset from sklearn.datasets

**Detailed Description**:
1. Import necessary libraries:
   - numpy for numerical operations
   - pandas for data manipulation (optional, for better display)
   - sklearn.datasets for loading the breast cancer dataset
   - matplotlib.pyplot for visualizations (for later demos)

2. Load the breast cancer dataset using sklearn's built-in function

3. Display basic dataset information:
   - Print the total number of samples (should be 569)
   - Print the total number of features (should be 30)
   - Show the first 5 feature names from the feature list (e.g., "mean radius", "mean texture", etc.)

4. Show class distribution:
   - Count how many samples belong to each class (malignant = 0, benign = 1)
   - Display as: "Malignant: X samples, Benign: Y samples"
   - Optionally show the percentage split

5. Display a few sample rows of the data:
   - Show first 3-5 rows with their features and corresponding labels
   - Format nicely so students can see what the data looks like

**Learning Objective**: Students understand they're working with real medical data, see the data structure (samples × features), and recognize this is a binary classification task.

**Expected Output Example**:
```
Dataset loaded successfully!
Total samples: 569
Total features: 30
First 5 features: ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness']

Class distribution:
  Malignant (0): 212 samples (37.3%)
  Benign (1): 357 samples (62.7%)
```

**Time**: ~2 minutes

---

## Demo 2: Train-Test Split

**Purpose**: Demonstrate proper data splitting methodology for unbiased model evaluation

**Detailed Description**:
1. Import train_test_split function from sklearn.model_selection

2. Split the dataset into training and testing sets:
   - Use 70% of data for training, 30% for testing
   - Set random_state=42 for reproducibility (so results are consistent across runs)
   - Use stratify parameter set to the target labels to maintain class balance in both sets

3. Display the sizes of resulting sets:
   - Print X_train shape (number of samples, number of features)
   - Print X_test shape
   - Print y_train shape (number of samples)
   - Print y_test shape

4. Verify class balance was maintained:
   - Show class distribution in training set
   - Show class distribution in test set
   - Both should have approximately the same percentage split as original data

5. Explain to students (via comment or print statement):
   - Why we need separate test data: to evaluate how well the model generalizes to unseen data
   - Why we use stratify: to ensure both sets have similar class proportions
   - Why we set random_state: to make results reproducible

**Learning Objective**: Students understand the fundamental principle of holding out test data and why proper splitting matters for honest evaluation.

**Expected Output Example**:
```
Data split complete!
Training set: 398 samples (70%)
  Malignant: 148, Benign: 250
Testing set: 171 samples (30%)
  Malignant: 64, Benign: 107

Class balance maintained in both sets ✓
```

**Time**: ~1 minute

---

## Demo 3: Train Logistic Regression (The "Magic" Moment)

**Purpose**: Show students how incredibly simple it is to train a classifier with sklearn

**Detailed Description**:
1. Import LogisticRegression from sklearn.linear_model

2. Create a LogisticRegression object:
   - Use default parameters (no need to specify anything)
   - Optionally set max_iter=10000 to ensure convergence (prevent warnings)
   - Use random_state=42 for reproducibility

3. Train the model:
   - Call the .fit() method with X_train and y_train
   - Emphasize: THIS SINGLE LINE does all the complex optimization!
   - Mention that sklearn automatically:
     - Initializes weights
     - Uses cross-entropy loss
     - Runs optimization (LBFGS or similar solver)
     - Finds the best weights

4. Show that training is complete:
   - Print a success message
   - Optionally display the training accuracy using .score(X_train, y_train)

5. Emphasize the simplicity:
   - "Just 3 lines of code: import, create, fit!"
   - "sklearn handles all the math and optimization for us"

**Learning Objective**: Students realize that ML is accessible - they don't need to understand all the math to use powerful tools. Build confidence!

**Expected Output Example**:
```
Creating logistic regression model...
Training the model...
✓ Training complete!

Training accuracy: 95.7%

That's it! Just 3 lines of code:
  1. Import LogisticRegression
  2. Create the model object
  3. Call .fit() with training data
```

**Time**: ~1.5 minutes

---

## Demo 4: Making Predictions

**Purpose**: Demonstrate the difference between hard class predictions and probability predictions

**Detailed Description**:
1. Make hard predictions (class labels):
   - Use model.predict(X_test) to get binary predictions (0 or 1)
   - Store result in y_pred

2. Make probability predictions:
   - Use model.predict_proba(X_test) to get probability estimates
   - This returns a 2D array where each row has [P(class 0), P(class 1)]
   - Extract just P(class 1) by taking the second column: [:, 1]
   - Store result in y_proba

3. Display first 10 test samples in a formatted table showing:
   - Sample index (0-9)
   - True label (actual class from y_test)
   - Predicted class (from y_pred)
   - P(class 0) - probability of malignant
   - P(class 1) - probability of benign
   - Confidence level (maximum of the two probabilities)

4. Highlight interesting cases:
   - Point out samples where model is very confident (prob > 0.9)
   - Point out samples where model is uncertain (prob close to 0.5)
   - Show if any predictions are wrong and note their confidence

5. Explain the connection:
   - When P(class 1) > 0.5, the model predicts class 1
   - When P(class 1) ≤ 0.5, the model predicts class 0
   - The threshold is 0.5 by default, but we can change it (will show later)

**Learning Objective**: Students understand that logistic regression outputs probabilities, and these are converted to hard classifications using a threshold.

**Expected Output Example**:
```
Predictions on first 10 test samples:

Sample | True | Pred | P(Malign) | P(Benign) | Confidence
-------|------|------|-----------|-----------|------------
   0   |  1   |  1   |   0.088   |   0.912   |   91.2%  ← Very confident
   1   |  0   |  0   |   0.857   |   0.143   |   85.7%
   2   |  1   |  1   |   0.124   |   0.876   |   87.6%
   3   |  0   |  0   |   0.911   |   0.089   |   91.1%
   4   |  1   |  0   |   0.513   |   0.487   |   51.3%  ← Uncertain!
   5   |  1   |  1   |   0.032   |   0.968   |   96.8%  ← Very confident
   ...

Notice: Sample 4 has low confidence (close to 50/50 split)
```

**Time**: ~2 minutes

---

## Demo 5: Display Confusion Matrix

**Purpose**: Visualize classification results in a confusion matrix to understand all types of errors

**Detailed Description**:
1. Import confusion_matrix from sklearn.metrics
2. Optionally import seaborn or matplotlib for visualization

3. Compute confusion matrix:
   - Call confusion_matrix(y_test, y_pred)
   - This returns a 2×2 numpy array

4. Display as a heatmap (visual approach):
   - Use seaborn.heatmap() or matplotlib's imshow()
   - Add annotations showing the actual counts in each cell
   - Use a colormap (e.g., Blues or RdYlGn)
   - Label axes clearly: "Predicted Label" (x-axis), "True Label" (y-axis)
   - Label ticks: [0: Malignant, 1: Benign]

5. OR display as a text table (simpler approach):
   - Print the matrix in a formatted way
   - Clearly label which cell is TP, TN, FP, FN

6. Extract and display the four key numbers:
   - True Negatives (TN): top-left cell
   - False Positives (FP): top-right cell
   - False Negatives (FN): bottom-left cell
   - True Positives (TP): bottom-right cell

7. Add interpretation:
   - "TN: Correctly identified malignant cases"
   - "TP: Correctly identified benign cases"
   - "FP: Falsely identified as benign (missed malignant)"
   - "FN: Falsely identified as malignant (false alarm)"

8. Calculate and display overall accuracy:
   - Accuracy = (TP + TN) / Total
   - Show the calculation explicitly

**Learning Objective**: Students learn to read and interpret confusion matrices, understanding what each cell represents.

**Expected Output Example**:
```
Confusion Matrix:

                  Predicted
                Malign  Benign
Actual  Malign    59      5      ← TN=59, FP=5
        Benign     3     104     ← FN=3,  TP=104

Interpretation:
  True Negatives (TN): 59 - Correctly identified malignant
  False Positives (FP): 5 - Incorrectly predicted as benign
  False Negatives (FN): 3 - Incorrectly predicted as malignant
  True Positives (TP): 104 - Correctly identified benign

Overall Accuracy: (59 + 104) / 171 = 95.3%
```

**Time**: ~2 minutes

---

## Demo 6: Calculate Metrics

**Purpose**: Compute and display precision, recall, F1-score, and understand what each measures

**Detailed Description**:
1. Import classification_report from sklearn.metrics
2. Optionally import individual metric functions (precision_score, recall_score, f1_score)

3. Display full classification report:
   - Call classification_report(y_test, y_pred)
   - This automatically shows precision, recall, F1 for both classes
   - Also shows support (number of samples per class)
   - Display with proper formatting

4. Manually calculate and display key metrics for the positive class (benign = 1):
   - Calculate precision: TP / (TP + FP)
   - Calculate recall: TP / (TP + FN)
   - Calculate F1-score: 2 × (precision × recall) / (precision + recall)
   - Show the formulas alongside the calculated values

5. Calculate overall accuracy:
   - (TP + TN) / Total samples
   - Compare with the accuracy from the classification report

6. Provide interpretation in context of breast cancer diagnosis:
   - "Precision = 95.4% means: When we predict benign, we're right 95.4% of the time"
   - "Recall = 97.2% means: We correctly identify 97.2% of all benign cases"
   - "F1 = 96.3% balances both metrics"
   - Discuss: In this medical context, which metric matters more?

7. Emphasize the tradeoffs:
   - High precision: Few false alarms (don't scare healthy patients)
   - High recall: Don't miss any malignant cases (critical!)
   - For cancer screening, recall is typically more important

**Learning Objective**: Students understand what each metric measures and how to interpret them in real-world context.

**Expected Output Example**:
```
Classification Report:

              precision    recall  f1-score   support
   Malignant      0.951     0.922     0.936        64
      Benign      0.954     0.972     0.963       107
    accuracy                          0.953       171

Manual calculation for Benign class:
  Precision = TP/(TP+FP) = 104/(104+5) = 0.954 = 95.4%
  Recall = TP/(TP+FN) = 104/(104+3) = 0.972 = 97.2%
  F1-Score = 2×(P×R)/(P+R) = 0.963 = 96.3%

Interpretation for medical diagnosis:
  → We correctly identify 97.2% of benign cases (high recall ✓)
  → When we say "benign", we're right 95.4% of the time (high precision ✓)
```

**Time**: ~2 minutes

---

## Demo 7: Threshold Tuning (The Precision-Recall Tradeoff)

**Purpose**: Demonstrate how changing the decision threshold affects precision and recall, showing the fundamental tradeoff

**Detailed Description**:
1. Get probability predictions:
   - Use model.predict_proba(X_test)[:, 1] to get P(benign) for all test samples
   - Store in y_proba variable

2. **Scenario 1: Default threshold (0.5)**
   - Use the existing y_pred (which uses 0.5 by default)
   - Calculate precision and recall using sklearn functions
   - Display as "Baseline: threshold = 0.5"

3. **Scenario 2: Lower threshold (0.3) - "Aggressive Screening"**
   - Create new predictions: y_pred_aggressive = (y_proba > 0.3).astype(int)
   - Calculate confusion matrix for these predictions
   - Calculate precision and recall
   - Display metrics
   - Explain: "Lower threshold means we predict 'benign' more often"
   - Show that recall increased (catch more benign cases)
   - Show that precision decreased (more false positives)

4. **Scenario 3: Higher threshold (0.7) - "Conservative Diagnosis"**
   - Create new predictions: y_pred_conservative = (y_proba > 0.7).astype(int)
   - Calculate confusion matrix for these predictions
   - Calculate precision and recall
   - Display metrics
   - Explain: "Higher threshold means we're more cautious about predicting 'benign'"
   - Show that precision increased (fewer false positives)
   - Show that recall decreased (miss some benign cases)

5. Create comparison table:
   - Show all three scenarios side-by-side
   - Columns: Threshold, Precision, Recall, F1-Score
   - Highlight the tradeoff pattern

6. Discuss the implications:
   - "For cancer screening, which threshold would you choose?"
   - "Low threshold (0.3): Don't miss any cancer, but many false alarms"
   - "High threshold (0.7): Fewer false alarms, but might miss some cancer"
   - "The choice depends on the cost of each type of error"

7. Optionally show how many predictions changed:
   - Count how many samples changed from class 0 to 1 or vice versa
   - Highlight specific examples where threshold change mattered

**Learning Objective**: Students understand the precision-recall tradeoff and learn that they can tune the threshold based on problem requirements.

**Expected Output Example**:
```
Threshold Tuning: Precision-Recall Tradeoff

Scenario 1: Default Threshold (0.5)
  Precision: 95.4%  |  Recall: 97.2%  |  F1: 96.3%
  
Scenario 2: Aggressive Screening (threshold = 0.3)
  Precision: 91.2%  |  Recall: 99.1%  |  F1: 94.9%
  → More benign predictions → Higher recall, lower precision
  → Catches almost all benign cases, but more false alarms
  
Scenario 3: Conservative Diagnosis (threshold = 0.7)
  Precision: 98.1%  |  Recall: 93.5%  |  F1: 95.7%
  → Fewer benign predictions → Higher precision, lower recall
  → Very confident when saying "benign", but misses some cases

Comparison Table:
┌───────────┬───────────┬─────────┬─────────┐
│ Threshold │ Precision │  Recall │ F1-Score│
├───────────┼───────────┼─────────┼─────────┤
│   0.3     │   91.2%   │  99.1%  │  94.9%  │ ← Aggressive
│   0.5     │   95.4%   │  97.2%  │  96.3%  │ ← Balanced
│   0.7     │   98.1%   │  93.5%  │  95.7%  │ ← Conservative
└───────────┴───────────┴─────────┴─────────┘

⚖️ The Tradeoff:
  Lower threshold → Higher Recall, Lower Precision
  Higher threshold → Higher Precision, Lower Recall
  
💡 For cancer screening: Usually choose lower threshold (don't miss cases!)
```

**Time**: ~3 minutes

---

## Demo 8: Visualize Decision Boundary (Optional/Time Permitting)

**Purpose**: Provide visual intuition for what a linear decision boundary looks like in 2D feature space

**Detailed Description**:
1. Select only 2 features from the dataset for visualization:
   - Choose features that show good separation (e.g., "mean radius" and "mean texture")
   - Extract X_2d = X[:, [0, 1]] (or whatever indices chosen)
   - Keep corresponding y labels

2. Re-train logistic regression on 2D data:
   - Split the 2D data into train/test
   - Create and train a new LogisticRegression model on X_2d_train
   - This gives us a 2D decision boundary we can visualize

3. Create a mesh grid:
   - Determine the range of both features
   - Create a grid of points covering the entire feature space
   - Use np.meshgrid() with fine resolution (e.g., 200×200 points)

4. Predict probabilities for all grid points:
   - Use the trained model to predict probability for every point in the grid
   - Reshape to match the grid dimensions

5. Create the visualization:
   - Plot filled contour showing probability regions (use plt.contourf)
   - Use colormap (e.g., RdBu_r) where red = high P(malignant), blue = high P(benign)
   - Add a colorbar showing probability scale
   - Overlay contour lines at key probabilities (0.1, 0.3, 0.5, 0.7, 0.9)
   - Make the 0.5 contour line thick and black (this is the decision boundary)

6. Overlay actual data points:
   - Scatter plot of training data
   - Use different colors and markers for two classes:
     - Malignant (0): red circles
     - Benign (1): blue triangles
   - Add transparency so we can see overlapping points

7. Label and annotate:
   - X-axis: first feature name
   - Y-axis: second feature name
   - Title: "Logistic Regression Decision Boundary (2D Projection)"
   - Legend showing classes and decision boundary
   - Add text annotation pointing to decision boundary: "Decision boundary (P=0.5)"

8. Explain to students:
   - "The black line is where P(benign) = 0.5"
   - "It's a straight line - that's what 'linear' means!"
   - "Points far from the line have confident predictions"
   - "Points near the line are uncertain"
   - "In 30D (full feature space), it's a 29D hyperplane, but the idea is the same"

**Learning Objective**: Students get visual intuition for linear decision boundaries and understand what "linear classification" means geometrically.

**Expected Output**: A matplotlib figure showing:
- Colored background (probability contours)
- Black decision boundary line
- Scattered data points (two classes with different colors)
- Clear labels and legend

**Time**: ~2 minutes (if time permits)

---

## Summary of Demo Flow

**Total estimated time: ~15 minutes for all demos**

1. **Demo 1-2** (3 min): Load data and split → Students see the data
2. **Demo 3** (1.5 min): Train model → "Magic" moment showing simplicity
3. **Demo 4** (2 min): Make predictions → Understanding probabilities vs classes
4. **Demo 5-6** (4 min): Evaluation metrics → Learning to assess performance
5. **Demo 7** (3 min): Threshold tuning → Understanding the tradeoff
6. **Demo 8** (2 min): Visualization → Optional if time permits

**Teaching Strategy**:
- Keep code on screen briefly, don't dwell on syntax
- Focus on outputs and interpretation
- Ask students questions: "What do you notice?" "Which threshold would you choose?"
- Connect back to slides: "Remember the confusion matrix we saw?"
- Emphasize the simplicity: "This is why sklearn is powerful - it handles the complexity!"

**Technical Setup**:
- Have Google Colab notebook pre-prepared with sections
- Test all code before class to ensure it runs smoothly
- Have backup static outputs in case of connectivity issues
- Consider having code cells pre-written but not executed (execute live)
- Use large font size in Colab for readability

---

**End of Appendix**
