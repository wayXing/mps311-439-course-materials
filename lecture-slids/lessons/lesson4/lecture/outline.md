# Lesson 4 Lecture Slides: Detailed Outline

## Slide-by-Slide Breakdown

| Slide # | Slide Title | Main Content Description |
|---------|-------------|--------------------------|
| 1 | Title Slide | Lesson 4: Linear Classification with Logistic Regression |
| 2 | Quick Recap | Review Weeks 1-3: Linear regression for continuous outputs, feature engineering, regularization. Today's new challenge: predicting categories instead of numbers. |
| 3 | The Classification Challenge | Real-world examples: spam detection, disease diagnosis, credit approval. Key question: Can we use what we learned for regression? |
| 4 | Why Linear Regression Fails | Visual demonstration using `fig_regression_classification_failure.png`. Show predictions outside [0,1] range and sensitivity to outliers - regression doesn't respect probability constraints. |
| 5 | What We Need Instead | Define requirements: outputs between 0 and 1, interpretable as probabilities, smooth function. Preview: Logistic regression gives us exactly this! |
| 6 | The Sigmoid Function | Introduce $\sigma(z) = \frac{1}{1+e^{-z}}$ with visual `fig_sigmoid_function.png`. Key insight: squashes any number into (0,1) range - perfect for probabilities! |
| 7 | Logistic Regression Model | Two-step process: (1) Linear combination $z = w^T x$, (2) Apply sigmoid to get $P(y=1\|x)$. Show how positive/negative weights affect predictions with simple example. |
| 8 | Decision Boundaries | Visualize with `fig_decision_boundary_2d.png` showing linear boundary and probability contours. Decision rule: predict class 1 if $P(y=1) > 0.5$, equivalent to $z > 0$. |
| 9 | Training: Loss Function | Brief mention that we use cross-entropy loss (not squared error). Show intuition: heavily penalizes confident wrong predictions, encourages calibrated probabilities. |
| 10 | Using Sklearn | Show the simplicity: 3 lines of code - create model, fit, predict. Mention `.predict()` gives classes, `.predict_proba()` gives probabilities - we'll see this live! |
| 11 | Evaluation: The Problem with Accuracy | Motivating scenario: 95% emails are not spam, model that always predicts "not spam" = 95% accuracy but useless! Need better metrics for imbalanced data. |
| 12 | Confusion Matrix | Visual `fig_confusion_matrix.png` with TP, TN, FP, FN clearly labeled. Foundation for all other metrics - tells us exactly where our model succeeds and fails. |
| 13 | Precision vs Recall | Define both metrics with formulas and medical diagnosis example. Show the tradeoff: conservative (high precision) vs aggressive (high recall) - no free lunch! |
| 14 | Live Demo Preview | What we'll build together: breast cancer classification in <10 lines of code. Steps: load data, train model, visualize boundary, examine metrics, adjust threshold. |
| 15 | **[LIVE DEMO]** | Execute all code demonstrations interactively in Google Colab (see Appendix for details). Students follow along, ask questions, experiment with threshold values. |
| 16 | When Does It Work? | Success: linearly separable data (`fig_linear_vs_nonlinear.png` left panel). Failure: complex non-linear patterns (right panel) - need different methods (preview Lesson 5: LDA/QDA). |
| 17 | Key Takeaways | Three main points: (1) Sigmoid transforms linear regression for classification, (2) Multiple metrics needed for proper evaluation, (3) Linear boundaries are limitation. Next week: curved boundaries with QDA! |
| 18 | Questions & Next Steps | Open floor for questions. Remind about lab session this week to practice. Preview: Lesson 5 covers discriminant analysis for non-linear decision boundaries. |

**Total Content Slides: 16** (excluding title slide)
**Lecture Structure:**
- Slides 1-5: Setup & Motivation (8 min)
- Slides 6-10: Core Method (15 min)
- Slides 11-14: Evaluation (12 min)
- Slide 15: Live Demo (10 min)
- Slides 16-18: Wrap-up (5 min)

---

## Detailed Slide Content Specifications

### Slide 1: Title Slide
- **Title:** Lesson 4: Linear Classification with Logistic Regression
- **Subtitle:** From Predicting Numbers to Predicting Categories
- **Footer:** MPS311/439 Machine Learning - Dr. Wei Xing

### Slide 2: Quick Recap
- **Layout:** Two-column
- **Left column:** 
  - Bullet points summarizing Weeks 1-3
  - Lesson 1: Python/Colab basics
  - Lesson 2: Linear regression (predict house prices, temperatures)
  - Lesson 3: Feature engineering & regularization (polynomial features, Ridge/Lasso)
- **Right column:**
  - Visual icon or simple diagram showing continuous output
  - **Today's transition:** What if output is discrete?
- **Speaking notes:** "We've been predicting continuous values. But many real problems need discrete predictions..."

### Slide 3: The Classification Challenge
- **Layout:** Grid of 4 example boxes
- **Examples with icons:**
  1. Email: Spam 📧 or Not Spam ✓
  2. Medical: Disease 🏥 or Healthy ❤️
  3. Finance: Approve 💳 or Reject ❌
  4. Image: Cat 🐱 or Dog 🐕
- **Bottom text (emphasized):** "Binary Classification: Predicting one of two categories"
- **Speaking notes:** "These are everywhere in ML applications. Can we adapt our linear regression?"

### Slide 4: Why Linear Regression Fails
- **Layout:** Large figure with annotations
- **Main content:** Display `fig_regression_classification_failure.png`
- **Key annotations (overlaid on figure):**
  - Red arrow pointing to prediction < 0: "Negative probability? Impossible!"
  - Red arrow pointing to prediction > 1: "Probability > 100%? Impossible!"
  - Highlight outlier point: "One outlier ruins everything"
- **Bottom text:** "Linear regression doesn't respect probability constraints [0,1]"
- **Speaking notes:** "Let's see what happens when we try... predictions go crazy!"

### Slide 5: What We Need Instead
- **Layout:** Checklist with ✓ marks
- **Requirements:**
  - ✓ Always output between 0 and 1
  - ✓ Interpretable as probabilities
  - ✓ Smooth and differentiable (for training)
  - ✓ Can still use linear combinations (w^T x)
- **Bottom (bold, colored):** "Solution: Logistic Regression"
- **Speaking notes:** "Here's our wishlist. Spoiler: logistic regression checks all boxes!"

### Slide 6: The Sigmoid Function
- **Layout:** Left figure, right explanation
- **Left:** Display `fig_sigmoid_function.png`
- **Right (annotated):**
  - Formula: $\sigma(z) = \frac{1}{1+e^{-z}}$
  - Key properties bullets:
    - Input: any number $(-\infty, +\infty)$
    - Output: always between (0, 1)
    - $\sigma(0) = 0.5$ (decision point)
  - Visual note: "S-shaped curve"
- **Bottom:** "The 'squashing' function that makes everything work!"
- **Speaking notes:** "This beautiful function is our key tool. No matter what z is, output is probability!"

### Slide 7: Logistic Regression Model
- **Layout:** Step-by-step visual flow
- **Step 1 (left):** 
  - Box: "Linear Combination"
  - $z = w_0 + w_1 x_1 + w_2 x_2 + \cdots$
  - "Same as linear regression!"
- **Arrow:** →
- **Step 2 (middle):**
  - Box: "Apply Sigmoid"
  - $P(y=1|x) = \sigma(z)$
  - "Squash to [0,1]"
- **Arrow:** →
- **Step 3 (right):**
  - Box: "Make Decision"
  - If $P > 0.5$: Class 1
  - If $P \leq 0.5$: Class 0
- **Example box below:** 
  - "Age=45, Cholesterol=200 → z=1.2 → P=0.77 → Disease"
- **Speaking notes:** "Just two steps: linear combination (we know this!) + sigmoid (new!)"

### Slide 8: Decision Boundaries
- **Layout:** Large centered figure with annotations
- **Main content:** Display `fig_decision_boundary_2d.png`
- **Annotations:**
  - Arrow to black line: "Decision Boundary (P=0.5)"
  - Arrow to blue region: "Predicts Class 0"
  - Arrow to red region: "Predicts Class 1"
  - Note: "Still a straight line (linear)!"
- **Bottom insight:** "Points far from boundary = high confidence, points near = uncertain"
- **Speaking notes:** "Notice it's still a linear boundary. That's both strength and limitation!"

### Slide 9: Training: Loss Function
- **Layout:** Simple visual comparison
- **Left box (crossed out):**
  - "❌ Squared Error"
  - $(y - \hat{y})^2$
  - Problem: Non-convex with sigmoid
- **Right box (checkmark):**
  - "✓ Cross-Entropy Loss"
  - $-[y \log(\hat{y}) + (1-y)\log(1-\hat{y})]$
  - Benefit: Convex, penalizes wrong confidence
- **Simple intuition box:**
  - "Wrong + Confident = Big Penalty"
  - "Wrong + Uncertain = Small Penalty"
- **Speaking notes:** "We need a different loss. Cross-entropy penalizes confident mistakes heavily."

### Slide 10: Using Sklearn
- **Layout:** Code snippet with annotations
- **Code box (syntax highlighted):**
```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)      # Train
y_pred = model.predict(X_test)   # Get classes (0 or 1)
y_prob = model.predict_proba(X_test)  # Get probabilities
```
- **Annotations:**
  - Arrow to line 3: "That's it - sklearn does all the math!"
  - Arrow to line 4: "Hard predictions"
  - Arrow to line 5: "Probability estimates"
- **Bottom (emphasized):** "Just like linear regression - sklearn makes it easy!"
- **Speaking notes:** "Look how simple! sklearn handles cross-entropy, optimization, everything!"

### Slide 11: Evaluation: The Problem with Accuracy
- **Layout:** Side-by-side comparison
- **Scenario box (top):**
  - "Email Dataset: 95% Not Spam, 5% Spam"
- **Left (Bad model):**
  - Model: "Always predict 'Not Spam'"
  - Accuracy: 95% ✓
  - Spam caught: 0% ❌
  - **Verdict:** "Useless!"
- **Right (Good model):**
  - Model: "Actually learns patterns"
  - Accuracy: 92%
  - Spam caught: 80% ✓
  - **Verdict:** "Useful!"
- **Bottom (bold):** "Accuracy alone is misleading with imbalanced classes!"
- **Speaking notes:** "This is why accuracy isn't enough. We need to see the full picture!"

### Slide 12: Confusion Matrix
- **Layout:** Large centered figure with clear labels
- **Main content:** Display `fig_confusion_matrix.png`
- **Additional text boxes around figure:**
  - Top-left (TP): "Caught the disease ✓"
  - Top-right (FP): "False alarm - worried patient unnecessarily"
  - Bottom-left (FN): "Missed the disease - dangerous!"
  - Bottom-right (TN): "Correctly said healthy ✓"
- **Bottom:** "Shows all four types of outcomes - foundation for all metrics"
- **Speaking notes:** "This 2x2 table tells us everything. Different cells matter for different applications!"

### Slide 13: Precision vs Recall
- **Layout:** Two-column comparison with medical example
- **Left column (Precision):**
  - Formula: $\frac{TP}{TP + FP}$
  - Meaning: "When I say disease, how often am I right?"
  - Medical analogy: "Conservative doctor"
  - Use case: When false alarms are costly
- **Right column (Recall):**
  - Formula: $\frac{TP}{TP + FN}$
  - Meaning: "Of all diseases, how many did I catch?"
  - Medical analogy: "Aggressive screening"
  - Use case: When missing cases is dangerous
- **Middle (emphasized):** "⚖️ The Tradeoff: Can't maximize both!"
- **Bottom:** "Choose based on your problem's costs"
- **Speaking notes:** "There's no free lunch. High precision means low recall and vice versa!"

### Slide 14: Live Demo Preview
- **Layout:** Numbered steps with icons
- **Title:** "What We'll Build Together (in <10 lines!)"
- **Steps:**
  1. 📊 Load breast cancer dataset (sklearn built-in)
  2. 🎯 Train logistic regression (1 line!)
  3. 📈 Visualize decision boundary (for 2 features)
  4. 🔍 Examine confusion matrix
  5. ⚖️ Adjust threshold to see precision/recall tradeoff
- **Bottom (excited tone):** "Follow along in your Colab - let's see the magic! 🚀"
- **Speaking notes:** "Open your Colab notebooks. We'll code this together in real-time!"

### Slide 15: [LIVE DEMO]
- **Layout:** Minimal text, maximize screen share
- **Slide content:**
  - Large text: "🖥️ LIVE CODING IN GOOGLE COLAB"
  - Subtitle: "Follow along: [Share Colab link]"
- **Speaking notes:** This slide stays on screen while you share your Colab window
- **Execution:** See Appendix for detailed demo script

### Slide 16: When Does It Work?
- **Layout:** Side-by-side comparison
- **Main content:** Display `fig_linear_vs_nonlinear.png`
- **Left panel annotation:**
  - "✓ Success: Linearly Separable"
  - "Straight line separates classes nicely"
- **Right panel annotation:**
  - "❌ Failure: Complex Patterns"
  - "Straight line can't capture XOR-like patterns"
- **Bottom boxes:**
  - Solutions: Feature engineering (like Lesson 3) OR non-linear models
  - Preview: "Next week - QDA gives us curved boundaries!"
- **Speaking notes:** "This is the fundamental limitation. But we have solutions!"

### Slide 17: Key Takeaways
- **Layout:** Three numbered boxes
- **1. The Method:**
  - Sigmoid transforms linear regression for classification
  - Two steps: linear combo + squashing function
  - Still produces linear decision boundaries
- **2. Evaluation:**
  - Accuracy isn't enough (especially imbalanced data)
  - Confusion matrix shows all outcomes
  - Precision vs Recall tradeoff - choose based on problem
- **3. Limitations & Next Steps:**
  - Only works for linearly separable data
  - Lesson 5: LDA/QDA for curved boundaries
  - Lesson 6: Decision trees for truly non-linear patterns
- **Speaking notes:** "These three points are what you should remember from today!"

### Slide 18: Questions & Next Steps
- **Layout:** Simple, open design
- **Content:**
  - Large text: "Questions? 🤔"
  - Subheadings:
    - **This Week:** Lab session - practice with real datasets
    - **Next Week:** Discriminant Analysis (LDA/QDA) - curved boundaries
    - **Assessment 1:** Released - due Lesson 6
- **Bottom:** Contact info and office hours
- **Speaking notes:** "Time for your questions. Also, don't forget the lab to practice!"

---

## Appendix: Interactive Code Demonstrations

### Demo 1: Basic Logistic Regression Training
**Purpose:** Show students how simple it is to train a logistic regression classifier using sklearn.

**Dataset:** Use sklearn's built-in breast cancer dataset (binary classification, well-behaved, meaningful features).

**Steps:**
1. Import necessary libraries: sklearn's LogisticRegression, train_test_split, and load_breast_cancer dataset
2. Load the breast cancer dataset and examine its shape (569 samples, 30 features)
3. Display first few feature names (e.g., mean radius, mean texture) so students understand it's real medical data
4. Split data into training (70%) and testing (30%) sets using train_test_split with random_state=42 for reproducibility
5. Create a LogisticRegression model with default parameters
6. Fit the model on training data (emphasize this is one line of code)
7. Make predictions on test set using .predict() method
8. Print training and testing accuracy scores
9. Show that the model achieves ~96% accuracy

**Key Teaching Points:**
- Emphasize the simplicity: just 3 lines of code to train and predict
- Mention that sklearn handles all the complex optimization (cross-entropy loss, gradient descent)
- Point out that high accuracy (96%) suggests linearly separable data
- Briefly explain train/test split concept if students seem confused

**Estimated Time:** 3 minutes

---

### Demo 2: Probability Predictions vs Class Predictions
**Purpose:** Help students understand the difference between hard class predictions and probability estimates.

**Steps:**
1. Use the same trained model from Demo 1
2. Get probability predictions using .predict_proba() method on first 10 test samples
3. Display side-by-side comparison showing:
   - True label (0 or 1)
   - Predicted class (0 or 1)
   - Probability of class 0
   - Probability of class 1
4. Point out specific examples:
   - High confidence correct: True=1, Pred=1, P(1)=0.98
   - High confidence incorrect (if any): True=0, Pred=1, P(1)=0.92
   - Low confidence: True=1, Pred=1, P(1)=0.52 (barely above threshold)
5. Explain that probabilities sum to 1.0 for each sample
6. Emphasize that .predict() internally uses 0.5 threshold on these probabilities

**Key Teaching Points:**
- Probabilities give us confidence levels, not just yes/no answers
- Low probability predictions (near 0.5) indicate uncertainty
- This probability information is useful for ranking predictions or custom thresholds
- Connect to real-world: "Would you trust a medical diagnosis with 0.51 probability?"

**Estimated Time:** 2 minutes

---

### Demo 3: Visualizing Decision Boundary (2D)
**Purpose:** Give students visual intuition for what logistic regression learns - a linear boundary in feature space.

**Steps:**
1. Select only two features from breast cancer dataset (e.g., "mean radius" and "mean texture") for 2D visualization
2. Create a new model and train on just these 2 features
3. Create a mesh grid covering the feature space (using np.meshgrid)
4. Compute probability predictions for every point in the mesh grid
5. Create a contour plot showing:
   - Background colored by probability (blue for P(cancer)~0, red for P(cancer)~1)
   - Decision boundary as a bold black line where P=0.5
   - Scatter plot of actual data points overlaid (different colors/markers for each class)
6. Zoom in to show points near the boundary (uncertain predictions)
7. Point to misclassified points if any exist

**Key Teaching Points:**
- "This is what the model 'sees' - it draws a straight line through feature space"
- Points far from boundary = high confidence, points near = uncertain
- The boundary is perfectly straight (linear) - this is the limitation
- Colors show probability gradient (not just binary decision)
- In higher dimensions (30 features), it's a hyperplane, but concept is the same

**Visualization Libraries:** Use matplotlib with contourf for filled contours and scatter for data points

**Estimated Time:** 2 minutes

---

### Demo 4: Confusion Matrix Analysis
**Purpose:** Show students how to compute and interpret the confusion matrix to understand model errors.

**Steps:**
1. Use predictions from Demo 1 (full 30-feature model on test set)
2. Import confusion_matrix and ConfusionMatrixDisplay from sklearn.metrics
3. Compute confusion matrix comparing y_test and y_pred
4. Display the matrix as a heatmap using ConfusionMatrixDisplay.from_predictions()
5. Point to each cell and explain:
   - Top-left (TN): Correctly predicted benign
   - Top-right (FP): Incorrectly predicted malignant (false alarm)
   - Bottom-left (FN): Incorrectly predicted benign (missed cancer - dangerous!)
   - Bottom-right (TP): Correctly predicted malignant
6. Calculate and display metrics manually from the matrix:
   - Accuracy = (TP + TN) / Total
   - Precision = TP / (TP + FP)
   - Recall = TP / (TP + FN)
7. Show sklearn's classification_report() for automatic metric calculation

**Key Teaching Points:**
- "This matrix tells us exactly where the model succeeds and fails"
- FN (missed cancer) is more dangerous than FP (false alarm) in medical context
- Different applications care about different cells
- All other metrics (precision, recall, F1) are derived from this 2x2 table

**Estimated Time:** 2 minutes

---

### Demo 5: Threshold Adjustment and Precision-Recall Tradeoff
**Purpose:** Demonstrate that we can adjust the decision threshold to trade precision for recall, reinforcing the "no free lunch" concept.

**Steps:**
1. Get probability predictions from Demo 1: y_prob = model.predict_proba(X_test)[:, 1]
2. Show default behavior: threshold = 0.5, compute precision and recall
3. **Scenario 1 - Conservative (High Precision):**
   - Set threshold = 0.7 (only predict cancer if P > 0.7)
   - Apply custom threshold: y_pred_conservative = (y_prob > 0.7).astype(int)
   - Compute new confusion matrix
   - Calculate precision (should increase) and recall (should decrease)
   - Interpretation: "Fewer false alarms, but we miss some cancers"
4. **Scenario 2 - Aggressive (High Recall):**
   - Set threshold = 0.3 (predict cancer if P > 0.3)
   - Apply custom threshold: y_pred_aggressive = (y_prob > 0.3).astype(int)
   - Compute confusion matrix
   - Calculate precision (should decrease) and recall (should increase)
   - Interpretation: "Catch more cancers, but more false alarms"
5. Create a simple plot showing precision, recall, and F1-score for thresholds from 0.1 to 0.9
6. Ask students: "Which threshold would you choose for cancer screening? Why?"

**Key Teaching Points:**
- Default 0.5 threshold isn't always optimal
- Can adjust threshold based on problem costs
- Medical screening: prefer high recall (catch all cancers, accept false alarms)
- Spam filter: prefer high precision (real emails must not go to spam)
- There's no universal "best" - depends on your application
- This demonstrates the fundamental tradeoff in classification

**Interactive Element:** Ask students to suggest thresholds and run them live, showing the metrics change

**Estimated Time:** 3 minutes (including student interaction)

---

### Demo 6: When Logistic Regression Fails (Optional, Time Permitting)
**Purpose:** Build intuition for when linear boundaries are insufficient.

**Steps:**
1. Generate synthetic XOR-pattern data using sklearn.datasets.make_classification with specific parameters to create non-linear separation
2. Visualize the data: two classes arranged in XOR pattern (class 1 in opposite corners, class 0 in other corners)
3. Train logistic regression on this data
4. Visualize the decision boundary (will be a diagonal line)
5. Show the poor accuracy (~50%, barely better than random guessing)
6. Explain: "A straight line cannot separate these classes - we need curved boundaries"
7. Preview: "Feature engineering (polynomial features) can help, OR use non-linear models like next week's QDA"

**Key Teaching Points:**
- Logistic regression has fundamental limitations
- Not all classification problems are linearly separable
- Visual failure is more convincing than just stating the limitation
- Motivates upcoming weeks (QDA for curved boundaries, trees for complex patterns)

**Estimated Time:** 2 minutes (if time permits)

---

## Demo Execution Notes

**Technical Setup:**
- All demos should be in a single Google Colab notebook with clear markdown section headers
- Use `random_state=42` everywhere for reproducibility
- Pre-load all libraries at the top to avoid delays during demo
- Test the entire notebook before class to ensure no errors
- Have a backup static version in case of connectivity issues

**Pedagogical Approach:**
- Type slowly or use pre-written cells that students can follow
- After each demo, pause for questions: "Does this make sense? Any questions?"
- Encourage students to modify values (e.g., try different thresholds)
- If a student asks "what if...", try it live if time permits
- Connect each demo back to the slides: "Remember when we saw the sigmoid function? This is it in action!"

**Time Management:**
- Allocate 10 minutes total for all demos
- If running behind, skip Demo 6 (optional)
- Can combine Demos 1 and 2 for efficiency
- Keep Demo 5 (threshold adjustment) as it's the most pedagogically valuable

**Common Student Questions to Anticipate:**
1. "Why is the decision boundary straight?" → Emphasize linear combination, preview non-linear methods
2. "How does sklearn optimize?" → Mention gradient descent briefly, say "beyond scope, covered in advanced ML"
3. "What threshold should we use?" → "Depends on your problem costs! No universal answer"
4. "Why not just use accuracy?" → Point back to the spam example on Slide 11
5. "Can we use this for 3+ classes?" → Yes! sklearn handles it automatically (mention softmax briefly)

---

## Additional Materials for Instructor

**Pre-Class Setup Checklist:**
- [ ] Create Google Colab notebook with all demos
- [ ] Test entire notebook (runtime: restart and run all)
- [ ] Share Colab link with students before class
- [ ] Have backup: downloaded .ipynb file and PDF export
- [ ] Ensure projector/screen share works
- [ ] Have figures (fig_*.png) accessible in Colab or local folder

**Student Engagement Strategies:**
- Start Demo 5 by asking: "You're designing a cancer screening test. High precision or high recall?"
- Let students vote on threshold choice
- If very engaged class, let a student suggest a threshold to try
- Encourage experimentation: "Try changing this value in your notebook during the break!"

**Links to Share:**
- Colab notebook with all demos (share at start of class)
- Link to lecture notes (for deeper reading)
- Sklearn documentation for LogisticRegression (for reference)

This detailed outline and demo script should provide everything needed for an effective 50-minute lecture! Let me know if you'd like me to proceed with generating the actual Slidev markdown.