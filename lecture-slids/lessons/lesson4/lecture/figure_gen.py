"""
Figure Generation Script for Lesson 4 Lecture Notes
Generates all figures for Linear Classification with Logistic Regression
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns
import os

# Create figure directory if it doesn't exist
os.makedirs('./figure', exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

# Set style
plt.style.use('default')
sns.set_palette("husl")

print("Generating figures for Lesson 4 lecture notes...")

# ============================================================================
# Figure 1: fig_regression_classification_failure
# ============================================================================
print("Generating Figure 1: Regression Classification Failure...")

fig, ax = plt.subplots(figsize=(10, 6))

# Generate 1D binary classification data
X_hours = np.array([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 
                     5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10,
                     0.8, 1.2, 2.2, 3.8, 4.2, 5.8, 6.2, 7.2, 8.2, 9.2,
                     1.5, 2.8, 4.8, 6.8, 7.8, 8.8, 9.8, 3.2, 5.2, 7.5])
y_pass = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
                    0, 0, 1, 1, 1, 1, 1, 0, 1, 1])

# Add an outlier
X_hours = np.append(X_hours, [1.0])
y_pass = np.append(y_pass, [1])

# Fit linear regression
X_hours_reshaped = X_hours.reshape(-1, 1)
lin_reg = LinearRegression()
lin_reg.fit(X_hours_reshaped, y_pass)

# Generate prediction line
X_line = np.linspace(-0.5, 11, 300).reshape(-1, 1)
y_pred_line = lin_reg.predict(X_line)

# Plot data points
class_0 = y_pass == 0
class_1 = y_pass == 1
ax.scatter(X_hours[class_0], y_pass[class_0], c='blue', marker='o', s=100, 
           label='Class 0 (Fail)', alpha=0.7, edgecolors='black')
ax.scatter(X_hours[class_1], y_pass[class_1], c='red', marker='^', s=120, 
           label='Class 1 (Pass)', alpha=0.7, edgecolors='black')

# Plot regression line
ax.plot(X_line, y_pred_line, 'g-', linewidth=2.5, label='Linear Regression', alpha=0.8)

# Add reference lines
ax.axhline(y=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
ax.axhline(y=1, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
ax.axhline(y=0.5, color='purple', linestyle=':', linewidth=2, alpha=0.6, label='Decision Threshold (0.5)')

# Add annotations for problematic predictions
ax.annotate('Prediction < 0\n(Impossible!)', xy=(0.5, -0.15), xytext=(1.5, -0.35),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=11, color='red', weight='bold')
ax.annotate('Prediction > 1\n(Impossible!)', xy=(9.5, 1.15), xytext=(8, 1.35),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=11, color='red', weight='bold')

# Highlight outlier
ax.scatter([1.0], [1], c='orange', marker='*', s=400, 
           edgecolors='black', linewidths=2, label='Outlier', zorder=5)

ax.set_xlabel('Hours Studied', fontsize=13, weight='bold')
ax.set_ylabel('Pass/Fail', fontsize=13, weight='bold')
ax.set_title('Why Linear Regression Fails for Classification', fontsize=15, weight='bold', pad=15)
ax.set_ylim(-0.5, 1.6)
ax.set_xlim(-0.5, 11)
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('./figures/fig_regression_classification_failure.png', dpi=300, bbox_inches='tight')
plt.close()

print("  ✓ Figure 1 saved")

# ============================================================================
# Figure 2: fig_sigmoid_function
# ============================================================================
print("Generating Figure 2: Sigmoid Function...")

fig, ax = plt.subplots(figsize=(10, 6))

# Generate sigmoid function
z = np.linspace(-7, 7, 500)
sigmoid = 1 / (1 + np.exp(-z))

# Plot sigmoid
ax.plot(z, sigmoid, 'b-', linewidth=3, label='σ(z) = 1/(1+e⁻ᶻ)')

# Mark key points
ax.scatter([0], [0.5], c='red', s=200, zorder=5, edgecolors='black', linewidths=2)
ax.annotate('(0, 0.5)\nDecision Point', xy=(0, 0.5), xytext=(1.5, 0.3),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=11, weight='bold')

# Add reference lines
ax.axhline(y=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
ax.axhline(y=1, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
ax.axvline(x=0, color='red', linestyle='--', linewidth=1.5, alpha=0.5)

# Add region annotations
ax.fill_between(z[z < 0], 0, 1, alpha=0.1, color='blue')
ax.fill_between(z[z > 0], 0, 1, alpha=0.1, color='red')

ax.text(-4, 0.7, 'z < 0 region:\nP(y=1) < 0.5\n→ Predict Class 0', 
        fontsize=11, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
        ha='center', weight='bold')
ax.text(4, 0.3, 'z > 0 region:\nP(y=1) > 0.5\n→ Predict Class 1', 
        fontsize=11, bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8),
        ha='center', weight='bold')

# Add asymptotic behavior labels
ax.text(-6, 0.05, 'σ(-∞) → 0', fontsize=12, weight='bold', style='italic')
ax.text(5, 0.95, 'σ(+∞) → 1', fontsize=12, weight='bold', style='italic')

ax.set_xlabel('z (Linear Combination)', fontsize=13, weight='bold')
ax.set_ylabel('σ(z) - Probability', fontsize=13, weight='bold')
ax.set_title('The Sigmoid Function: Mapping Real Numbers to Probabilities', 
             fontsize=15, weight='bold', pad=15)
ax.set_ylim(-0.1, 1.1)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=12, loc='upper left')

plt.tight_layout()
plt.savefig('./figures/fig_sigmoid_function.png', dpi=300, bbox_inches='tight')
plt.close()

print("  ✓ Figure 2 saved")

# ============================================================================
# Figure 3: fig_decision_boundary_2d
# ============================================================================
print("Generating Figure 3: Decision Boundary 2D...")

fig, ax = plt.subplots(figsize=(10, 8))

# Generate 2D binary classification dataset
np.random.seed(42)
n_samples = 200

# Class 0
X0 = np.random.randn(n_samples//2, 2) + np.array([-1.5, -1.5])
y0 = np.zeros(n_samples//2)

# Class 1
X1 = np.random.randn(n_samples//2, 2) + np.array([1.5, 1.5])
y1 = np.ones(n_samples//2)

# Combine
X = np.vstack([X0, X1])
y = np.hstack([y0, y1])

# Fit logistic regression
log_reg = LogisticRegression()
log_reg.fit(X, y)

# Create mesh for probability contours
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))

# Get probability predictions
Z = log_reg.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
Z = Z.reshape(xx.shape)

# Plot probability contours (background)
contour_filled = ax.contourf(xx, yy, Z, levels=20, cmap='RdBu_r', alpha=0.6)
plt.colorbar(contour_filled, ax=ax, label='P(y=1)')

# Plot contour lines
contour_lines = ax.contour(xx, yy, Z, levels=[0.1, 0.3, 0.5, 0.7, 0.9], 
                            colors='black', linewidths=1, alpha=0.4)
ax.clabel(contour_lines, inline=True, fontsize=9)

# Plot decision boundary (P=0.5)
ax.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=3)

# Plot data points
ax.scatter(X[y==0, 0], X[y==0, 1], c='blue', marker='o', s=80, 
           label='Class 0', alpha=0.8, edgecolors='black', linewidths=1)
ax.scatter(X[y==1, 0], X[y==1, 1], c='red', marker='^', s=100, 
           label='Class 1', alpha=0.8, edgecolors='black', linewidths=1)

ax.set_xlabel('Feature 1', fontsize=13, weight='bold')
ax.set_ylabel('Feature 2', fontsize=13, weight='bold')
ax.set_title('Logistic Regression Decision Boundary and Probability Regions', 
             fontsize=15, weight='bold', pad=15)
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('./figures/fig_decision_boundary_2d.png', dpi=300, bbox_inches='tight')
plt.close()

print("  ✓ Figure 3 saved")

# ============================================================================
# Figure 4: fig_confusion_matrix
# ============================================================================
print("Generating Figure 4: Confusion Matrix...")

fig, ax = plt.subplots(figsize=(10, 8))

# Example confusion matrix values
cm = np.array([[85, 10],
               [5, 100]])

# Calculate metrics
TN, FP, FN, TP = cm.ravel()
precision = TP / (TP + FP)
recall = TP / (TP + FN)
accuracy = (TP + TN) / cm.sum()

# Create custom colormap (green for correct, red for incorrect)
colors = np.array([[0.6, 0.9, 0.6], [0.9, 0.5, 0.5],
                   [0.9, 0.5, 0.5], [0.5, 0.8, 0.5]])
colors_reshaped = colors.reshape(2, 2, 3)

# Plot confusion matrix
im = ax.imshow(cm, cmap='RdYlGn', alpha=0.7, vmin=0, vmax=100)

# Add text annotations
labels = [['True Negative\n(Correct Rejection)', 'False Positive\n(Type I Error)'],
          ['False Negative\n(Type II Error)', 'True Positive\n(Correct Detection)']]

for i in range(2):
    for j in range(2):
        # Large count number
        text = ax.text(j, i, f'{cm[i, j]}',
                      ha="center", va="center", color="black",
                      fontsize=40, weight='bold')
        # Label text
        text = ax.text(j, i+0.35, labels[i][j],
                      ha="center", va="center", color="black",
                      fontsize=11, weight='bold')

# Set ticks and labels
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['Predicted\nClass 0', 'Predicted\nClass 1'], fontsize=12, weight='bold')
ax.set_yticklabels(['Actual\nClass 0', 'Actual\nClass 1'], fontsize=12, weight='bold')

# Add title
ax.set_title('Confusion Matrix: Understanding Classification Errors', 
             fontsize=16, weight='bold', pad=20)

# Add metrics below
metrics_text = f'Accuracy = (TP+TN)/Total = ({TP}+{TN})/{cm.sum()} = {accuracy:.3f}\n'
metrics_text += f'Precision = TP/(TP+FP) = {TP}/({TP}+{FP}) = {precision:.3f}\n'
metrics_text += f'Recall = TP/(TP+FN) = {TP}/({TP}+{FN}) = {recall:.3f}'

ax.text(0.5, -0.25, metrics_text, transform=ax.transAxes,
        fontsize=13, weight='bold', ha='center',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('./figures/fig_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

print("  ✓ Figure 4 saved")

# ============================================================================
# Figure 5: fig_metrics_comparison
# ============================================================================
print("Generating Figure 5: Metrics Comparison...")

fig, ax = plt.subplots(figsize=(12, 7))

# Classifier data
classifiers = ['Balanced', 'Conservative', 'Aggressive', 'Random']
precision = [0.80, 0.95, 0.65, 0.50]
recall = [0.82, 0.60, 0.95, 0.52]
f1 = [0.81, 0.74, 0.77, 0.51]

# Set up bar positions
x = np.arange(len(classifiers))
width = 0.25

# Create bars
bars1 = ax.bar(x - width, precision, width, label='Precision', color='steelblue', 
               edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x, recall, width, label='Recall', color='coral', 
               edgecolor='black', linewidth=1.5)
bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='mediumseagreen', 
               edgecolor='black', linewidth=1.5)

# Add value labels on bars
def add_value_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=11, weight='bold')

add_value_labels(bars1)
add_value_labels(bars2)
add_value_labels(bars3)

# Add reference line at 0.5
ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=2, alpha=0.5, label='Random baseline')

# Customize
ax.set_xlabel('Classifier Type', fontsize=13, weight='bold')
ax.set_ylabel('Metric Value', fontsize=13, weight='bold')
ax.set_title('Comparing Classification Metrics Across Models', fontsize=15, weight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(classifiers, fontsize=12, weight='bold')
ax.set_ylim(0, 1.1)
ax.legend(fontsize=12, loc='upper right')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('./figures/fig_metrics_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print("  ✓ Figure 5 saved")

# ============================================================================
# Figure 6: fig_roc_curve
# ============================================================================
print("Generating Figure 6: ROC Curve...")

fig, ax = plt.subplots(figsize=(9, 9))

# Generate example data
np.random.seed(42)
n = 300
y_true = np.random.randint(0, 2, n)
y_scores = np.random.rand(n)
# Make scores somewhat correlated with true labels
y_scores[y_true == 1] += 0.4
y_scores = np.clip(y_scores, 0, 1)

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
ax.plot(fpr, tpr, color='blue', linewidth=3, label=f'ROC Curve (AUC = {roc_auc:.2f})')

# Fill area under curve
ax.fill_between(fpr, tpr, alpha=0.3, color='lightblue')

# Plot diagonal (random classifier)
ax.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=2, 
        label='Random Guess (AUC = 0.5)')

# Mark some key points
sample_indices = [len(fpr)//4, len(fpr)//2, 3*len(fpr)//4]
for idx in sample_indices:
    ax.scatter(fpr[idx], tpr[idx], s=100, c='red', zorder=5, edgecolors='black', linewidths=2)

# Add annotations
ax.annotate('Better\n(towards top-left)', xy=(0.1, 0.9), xytext=(0.3, 0.75),
            arrowprops=dict(arrowstyle='->', color='green', lw=2.5),
            fontsize=12, weight='bold', color='green')
ax.annotate('Worse\n(towards bottom-right)', xy=(0.9, 0.1), xytext=(0.6, 0.3),
            arrowprops=dict(arrowstyle='->', color='red', lw=2.5),
            fontsize=12, weight='bold', color='red')

# Add AUC text box
ax.text(0.6, 0.2, f'AUC = {roc_auc:.3f}', fontsize=18, weight='bold',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

ax.set_xlabel('False Positive Rate', fontsize=13, weight='bold')
ax.set_ylabel('True Positive Rate (Recall)', fontsize=13, weight='bold')
ax.set_title('ROC Curve: Tradeoff Between True and False Positives', 
             fontsize=15, weight='bold', pad=15)
ax.legend(fontsize=12, loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig('./figures/fig_roc_curve.png', dpi=300, bbox_inches='tight')
plt.close()

print("  ✓ Figure 6 saved")

# ============================================================================
# Figure 7: fig_linear_vs_nonlinear
# ============================================================================
print("Generating Figure 7: Linear vs Nonlinear...")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# LEFT PANEL: Linearly Separable Data
np.random.seed(42)
n = 150

# Generate linearly separable data
X0_linear = np.random.randn(n//2, 2) * 0.8 + np.array([-2, -2])
X1_linear = np.random.randn(n//2, 2) * 0.8 + np.array([2, 2])
X_linear = np.vstack([X0_linear, X1_linear])
y_linear = np.hstack([np.zeros(n//2), np.ones(n//2)])

# Fit logistic regression
log_reg_linear = LogisticRegression()
log_reg_linear.fit(X_linear, y_linear)

# Create mesh
x_min, x_max = X_linear[:, 0].min() - 1, X_linear[:, 0].max() + 1
y_min, y_max = X_linear[:, 1].min() - 1, X_linear[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

Z_linear = log_reg_linear.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
Z_linear = Z_linear.reshape(xx.shape)

# Plot
axes[0].contourf(xx, yy, Z_linear, levels=20, cmap='RdBu_r', alpha=0.5)
axes[0].contour(xx, yy, Z_linear, levels=[0.5], colors='black', linewidths=3)
axes[0].scatter(X_linear[y_linear==0, 0], X_linear[y_linear==0, 1], 
                c='blue', marker='o', s=80, label='Class 0', 
                alpha=0.8, edgecolors='black', linewidths=1)
axes[0].scatter(X_linear[y_linear==1, 0], X_linear[y_linear==1, 1], 
                c='red', marker='^', s=100, label='Class 1', 
                alpha=0.8, edgecolors='black', linewidths=1)

axes[0].set_title('Linearly Separable Data\nLogistic Regression Works Well ✓', 
                  fontsize=14, weight='bold', color='green')
axes[0].set_xlabel('Feature 1', fontsize=12, weight='bold')
axes[0].set_ylabel('Feature 2', fontsize=12, weight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# RIGHT PANEL: Non-linearly Separable Data (XOR-like)
np.random.seed(42)

# Generate XOR pattern - ensure consistent sizes
samples_per_cluster = n // 4
X0_nonlinear_1 = np.random.randn(samples_per_cluster, 2) * 0.5 + np.array([-2, -2])
X0_nonlinear_2 = np.random.randn(samples_per_cluster, 2) * 0.5 + np.array([2, 2])
X0_nonlinear = np.vstack([X0_nonlinear_1, X0_nonlinear_2])

X1_nonlinear_1 = np.random.randn(samples_per_cluster, 2) * 0.5 + np.array([-2, 2])
X1_nonlinear_2 = np.random.randn(samples_per_cluster, 2) * 0.5 + np.array([2, -2])
X1_nonlinear = np.vstack([X1_nonlinear_1, X1_nonlinear_2])

X_nonlinear = np.vstack([X0_nonlinear, X1_nonlinear])
y_nonlinear = np.hstack([np.zeros(2 * samples_per_cluster), np.ones(2 * samples_per_cluster)])

# Fit logistic regression (will fail to capture pattern)
log_reg_nonlinear = LogisticRegression()
log_reg_nonlinear.fit(X_nonlinear, y_nonlinear)

# Create mesh
x_min, x_max = X_nonlinear[:, 0].min() - 1, X_nonlinear[:, 0].max() + 1
y_min, y_max = X_nonlinear[:, 1].min() - 1, X_nonlinear[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

Z_nonlinear = log_reg_nonlinear.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
Z_nonlinear = Z_nonlinear.reshape(xx.shape)

# Plot
axes[1].contourf(xx, yy, Z_nonlinear, levels=20, cmap='RdBu_r', alpha=0.5)
axes[1].contour(xx, yy, Z_nonlinear, levels=[0.5], colors='black', linewidths=3)
axes[1].scatter(X_nonlinear[y_nonlinear==0, 0], X_nonlinear[y_nonlinear==0, 1], 
                c='blue', marker='o', s=80, label='Class 0', 
                alpha=0.8, edgecolors='black', linewidths=1)
axes[1].scatter(X_nonlinear[y_nonlinear==1, 0], X_nonlinear[y_nonlinear==1, 1], 
                c='red', marker='^', s=100, label='Class 1', 
                alpha=0.8, edgecolors='black', linewidths=1)

# Add arrows pointing to misclassified regions
axes[1].annotate('Misclassified\nRegion', xy=(-2, -2), xytext=(-3.5, -0.5),
                arrowprops=dict(arrowstyle='->', color='darkred', lw=3),
                fontsize=11, weight='bold', color='darkred')
axes[1].annotate('Misclassified\nRegion', xy=(2, 2), xytext=(3.5, 0.5),
                arrowprops=dict(arrowstyle='->', color='darkred', lw=3),
                fontsize=11, weight='bold', color='darkred')

axes[1].set_title('Non-Linearly Separable Data\nLogistic Regression Fails ✗', 
                  fontsize=14, weight='bold', color='darkred')
axes[1].set_xlabel('Feature 1', fontsize=12, weight='bold')
axes[1].set_ylabel('Feature 2', fontsize=12, weight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

# Overall title
fig.suptitle('When Linear Decision Boundaries Are (In)sufficient', 
             fontsize=16, weight='bold', y=1.02)

plt.tight_layout()
plt.savefig('./figures/fig_linear_vs_nonlinear.png', dpi=300, bbox_inches='tight')
plt.close()

print("  ✓ Figure 7 saved")

# ============================================================================
# Figure 8: fig_gradient_descent_convergence
# ============================================================================
print("Generating Figure 8: Gradient Descent Convergence...")

fig, ax = plt.subplots(figsize=(12, 7))

# Generate binary classification data
np.random.seed(42)
n = 100
X_train = np.random.randn(n, 2)
y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)

# Add intercept term
X_train_with_intercept = np.c_[np.ones(n), X_train]

# Define sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

# Define cross-entropy loss
def compute_loss(X, y, w):
    predictions = sigmoid(X @ w)
    predictions = np.clip(predictions, 1e-7, 1 - 1e-7)  # Avoid log(0)
    return -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))

# Gradient descent with different learning rates
learning_rates = [0.01, 0.1, 1.0]
colors = ['green', 'blue', 'red']
labels = ['Too Low (α=0.01)', 'Good (α=0.1)', 'Too High (α=1.0)']
line_styles = [':', '-', '--']
max_iters = 200

for lr, color, label, ls in zip(learning_rates, colors, labels, line_styles):
    # Initialize weights
    w = np.random.randn(3) * 0.01
    losses = []
    
    for iteration in range(max_iters):
        # Compute predictions
        predictions = sigmoid(X_train_with_intercept @ w)
        
        # Compute gradient
        gradient = X_train_with_intercept.T @ (predictions - y_train) / n
        
        # Update weights
        w = w - lr * gradient
        
        # Store loss
        loss = compute_loss(X_train_with_intercept, y_train, w)
        losses.append(loss)
    
    # Plot
    ax.plot(range(max_iters), losses, color=color, linewidth=2.5, 
            label=label, linestyle=ls, alpha=0.9)

# Add annotations
ax.annotate('Rapid Initial Descent', xy=(10, 0.55), xytext=(40, 0.65),
            arrowprops=dict(arrowstyle='->', color='black', lw=2),
            fontsize=12, weight='bold')
ax.annotate('Convergence Region\n(Global Minimum)', xy=(150, 0.25), xytext=(100, 0.15),
            arrowprops=dict(arrowstyle='->', color='black', lw=2),
            fontsize=12, weight='bold')

# Highlight the convergence
ax.axhline(y=0.23, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
ax.text(180, 0.21, 'Global Minimum', fontsize=11, style='italic')

ax.set_xlabel('Iteration / Epoch', fontsize=13, weight='bold')
ax.set_ylabel('Cross-Entropy Loss', fontsize=13, weight='bold')
ax.set_title('Gradient Descent Convergence for Logistic Regression', 
             fontsize=15, weight='bold', pad=15)
ax.legend(fontsize=12, loc='upper right', title='Learning Rate', title_fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 0.75)

plt.tight_layout()
plt.savefig('./figures/fig_gradient_descent_convergence.png', dpi=300, bbox_inches='tight')
plt.close()

print("  ✓ Figure 8 saved")

print("\n" + "="*60)
print("All figures generated successfully!")
print("Figures saved to: ./figures/")
print("="*60)
print("\nGenerated figures:")
print("  1. fig_regression_classification_failure.png")
print("  2. fig_sigmoid_function.png")
print("  3. fig_decision_boundary_2d.png")
print("  4. fig_confusion_matrix.png")
print("  5. fig_metrics_comparison.png")
print("  6. fig_roc_curve.png")
print("  7. fig_linear_vs_nonlinear.png")
print("  8. fig_gradient_descent_convergence.png")
