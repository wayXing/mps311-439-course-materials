# Figure Quick Reference Guide
## Lesson 6: Decision Trees - MPS311/439

---

## Figure 1: `fig_xor_linear_fail.png`
**Size**: 8" × 6" (245 KB)  
**Purpose**: Motivation - Show limitations of linear classifiers  
**Section**: Introduction & Motivation (1.2)  
**Description**: XOR problem with 4 clusters showing where linear decision boundaries fail. Includes misclassified points marked with X.  
**Key Elements**:
- 200 data points (4 clusters of 50)
- Linear logistic regression boundary (green dashed line)
- Error markers showing misclassifications
- Clear demonstration of non-linear problem

**Teaching Point**: "Linear models can't solve XOR - we need something more flexible!"

---

## Figure 2: `fig_nested_boundaries.png`
**Size**: 8" × 6" (287 KB)  
**Purpose**: Show complex boundaries that challenge linear methods  
**Section**: Introduction & Motivation (1.2)  
**Description**: Nested circular classes (inner circle vs outer ring) with QDA decision boundary struggling to separate them.  
**Key Elements**:
- 200 data points (100 inner, 100 outer)
- Class A: Inner circle (radius 0-2)
- Class B: Outer ring (radius 2-4)
- QDA boundary (purple dashed line)

**Teaching Point**: "Even quadratic boundaries struggle with nested/circular patterns!"

---

## Figure 3: `fig_simple_tree_example.png`
**Size**: 12" × 8" (363 KB)  
**Purpose**: Visual introduction to decision tree structure  
**Section**: What is a Decision Tree? (2.2)  
**Description**: Complete decision tree (depth=3) trained on Iris dataset showing all components: root, internal nodes, leaves.  
**Key Elements**:
- Root node at top
- Decision nodes with feature ≤ threshold tests
- Leaf nodes color-coded by majority class
- Shows: samples, Gini, value arrays
- Uses petal length and petal width features

**Teaching Point**: "This is what a decision tree looks like - a flowchart of yes/no questions!"

---

## Figure 4: `fig_gini_entropy_comparison.png`
**Size**: 10" × 5" (203 KB)  
**Purpose**: Compare two splitting criteria mathematically  
**Section**: Building a Tree (3.5)  
**Description**: Line plots comparing Gini impurity and Entropy as functions of class probability.  
**Key Elements**:
- X-axis: Probability of Class 1 (0 to 1)
- Y-axis: Impurity measure (0 to 1)
- Blue solid line: Gini = 2p(1-p)
- Red dashed line: Entropy = -p log₂(p) - (1-p) log₂(1-p)
- Vertical line at p=0.5 (maximum impurity)

**Teaching Point**: "Both metrics are similar - Gini is slightly faster to compute!"

---

## Figure 5: `fig_tree_decision_boundary.png`
**Size**: 12" × 5" (319 KB)  
**Purpose**: Compare shallow vs deep tree decision boundaries  
**Section**: Implementation in Python (4.3)  
**Description**: Side-by-side comparison of decision boundaries on moons dataset.  
**Key Elements**:
- Left panel: max_depth=2 (shallow, smooth boundary)
- Right panel: max_depth=8 (deep, complex boundary)
- Same dataset (200 samples)
- Color-coded decision regions (semi-transparent)
- Scatter points overlay

**Teaching Point**: "Deeper trees → more complex boundaries (but risk overfitting!)"

---

## Figure 6: `fig_overfitting_comparison.png`
**Size**: 12" × 10" (468 KB)  
**Purpose**: Comprehensive demonstration of overfitting problem  
**Section**: The Overfitting Problem (5.3)  
**Description**: 2×2 grid showing multiple aspects of overfitting.  

**Panel Breakdown**:
- **Top-left**: Shallow tree (depth=2) - underfitting
  - Simple boundary, misses some patterns
  
- **Top-right**: Deep tree (depth=15) - overfitting
  - Complex, jagged boundary following noise
  
- **Bottom-left**: Training vs Test Accuracy
  - Training accuracy increases monotonically
  - Test accuracy peaks then declines
  - Optimal depth marked (~6)
  
- **Bottom-right**: Training Set Gini Impurity
  - Decreases with depth
  - Approaches zero for deep trees

**Teaching Point**: "Perfect training accuracy ≠ good model! Watch the test accuracy!"

---

## Figure 7: `fig_tree_instability.png`
**Size**: 15" × 6" (472 KB)  
**Purpose**: Demonstrate high variance of single trees  
**Section**: [OPTIONAL - MPS439] Beyond Single Trees (9.1.1)  
**Description**: 5 different decision trees trained on bootstrap samples from same data.  
**Key Elements**:
- 5 panels showing different bootstrap samples
- Same underlying data (200 samples, moons dataset)
- Each tree has max_depth=5
- Different colored boundaries for each tree
- Shows significant variation in decision boundaries

**Teaching Point**: "Small data changes → completely different trees! This is why we need ensembles."

---

## Figure 8: `fig_ensemble_comparison.png`
**Size**: 10" × 8" (195 KB)  
**Purpose**: Compare single tree vs ensemble methods  
**Section**: [OPTIONAL - MPS439] Beyond Single Trees (9.5.2)  
**Description**: Performance comparison on Wine dataset.  

**Panel Breakdown**:
- **Top panel**: Cross-validation accuracy (5-fold)
  - Decision Tree: baseline
  - Random Forest: improvement
  - XGBoost: best performance
  - Error bars show standard deviation
  
- **Bottom panel**: Training time comparison
  - Shows computation cost trade-off
  - XGBoost faster than Random Forest

**Teaching Point**: "XGBoost wins on accuracy! This is what professionals use."

---

## Usage in Lecture Notes

### Markdown Syntax
```markdown
![XOR Problem](./figures/fig_xor_linear_fail.png)
*Figure 1: The XOR problem demonstrates where linear classifiers fail.*
```

### HTML Syntax
```html
<img src="./figures/fig_xor_linear_fail.png" alt="XOR Problem" width="600">
<p><em>Figure 1: The XOR problem demonstrates where linear classifiers fail.</em></p>
```

### LaTeX Syntax
```latex
\begin{figure}[h]
  \centering
  \includegraphics[width=0.8\textwidth]{figures/fig_xor_linear_fail.png}
  \caption{The XOR problem demonstrates where linear classifiers fail.}
  \label{fig:xor_fail}
\end{figure}
```

---

## Figure Organization by Section

### Section 1: Introduction & Motivation
- Figure 1: `fig_xor_linear_fail.png`
- Figure 2: `fig_nested_boundaries.png`

### Section 2: What is a Decision Tree?
- Figure 3: `fig_simple_tree_example.png`

### Section 3: Building a Tree
- Figure 4: `fig_gini_entropy_comparison.png`

### Section 4: Implementation in Python
- Figure 5: `fig_tree_decision_boundary.png`

### Section 5: The Overfitting Problem
- Figure 6: `fig_overfitting_comparison.png`

### Section 9: [OPTIONAL - MPS439] Beyond Single Trees
- Figure 7: `fig_tree_instability.png`
- Figure 8: `fig_ensemble_comparison.png`

---

## Design Philosophy

### Visual Hierarchy
1. **Clear titles**: Bold, large font, descriptive
2. **Axis labels**: Always present, bold, units when relevant
3. **Legends**: Positioned to not obscure data
4. **Grid lines**: Subtle (alpha=0.3) for readability
5. **Colors**: Accessible, distinguishable

### Color Scheme
- **Blue**: Often Class A or positive class
- **Red**: Often Class B or negative class
- **Green**: Optimal points, decision boundaries
- **Purple/Orange**: Secondary elements, errors
- **Semi-transparent fills**: Decision regions (alpha=0.3)

### Accessibility Considerations
- High contrast between elements
- Multiple visual cues (color + shape + line style)
- Large enough text (min 10pt)
- Clear line widths (min 1.5pt for main elements)

---

## Quality Checklist

Before using figures in lecture:
- [ ] All text is readable at projected size
- [ ] Colors are distinguishable for colorblind viewers
- [ ] Axis labels are descriptive (not just "x" and "y")
- [ ] Legends explain all visual elements
- [ ] File size is reasonable (<500 KB per figure)
- [ ] Resolution is adequate (300 DPI)
- [ ] Figure aligns with lecture narrative

---

## Regenerating Figures

If you need to regenerate any figure:

1. Open `figure_gen.py`
2. Modify the specific function (e.g., `generate_fig_xor_linear_fail()`)
3. Run the script: `python figure_gen.py`
4. All figures regenerate with consistent random seeds

**Tip**: Comment out other functions in `main()` to regenerate only one figure quickly.

---

*Quick Reference Guide v1.0*  
*MPS311/439 - Machine Learning Course*  
*Lesson 6: Decision Trees*
