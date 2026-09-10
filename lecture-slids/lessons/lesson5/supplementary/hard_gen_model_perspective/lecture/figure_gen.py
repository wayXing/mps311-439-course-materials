"""
Figure Generation Script for Lesson 5: Linear Discriminant Analysis
Generates all figures needed for the lecture notes
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, make_classification
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from scipy.stats import multivariate_normal
import os

# Create figure directory if it doesn't exist
os.makedirs('./figure', exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

# Set style for all plots
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.dpi'] = 100

print("Generating figures for Lesson 5 lecture notes...")

# =============================================================================
# Figure 1: fig1_iris_species.png (800x400)
# Two iris species showing natural separation
# =============================================================================
print("Generating Figure 1: Iris species scatter plot...")

iris = load_iris()
# Use only first 2 features and first 2 classes
X_fig1 = iris.data[:100, :2]  # First 100 samples (classes 0 and 1)
y_fig1 = iris.target[:100]

fig1, ax1 = plt.subplots(figsize=(8, 4))

# Plot class 0 (Setosa)
mask0 = y_fig1 == 0
ax1.scatter(X_fig1[mask0, 0], X_fig1[mask0, 1], 
           c='#1f77b4', marker='o', s=60, alpha=0.6, 
           label='Setosa', edgecolors='k', linewidth=0.5)

# Plot class 1 (Versicolor)
mask1 = y_fig1 == 1
ax1.scatter(X_fig1[mask1, 0], X_fig1[mask1, 1], 
           c='#ff7f0e', marker='^', s=60, alpha=0.6, 
           label='Versicolor', edgecolors='k', linewidth=0.5)

ax1.set_xlabel('Sepal length (cm)')
ax1.set_ylabel('Sepal width (cm)')
ax1.set_title('Two Iris Species in Feature Space')
ax1.legend()
ax1.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('./figure/fig1_iris_species.png', dpi=100, bbox_inches='tight')
plt.close()

print("  ✓ Figure 1 saved")

# =============================================================================
# Figure 2: fig2_gaussian_classes.png (800x500)
# Two Gaussian distributions with equal covariance
# =============================================================================
print("Generating Figure 2: Gaussian classes with equal covariance...")

fig2, ax2 = plt.subplots(figsize=(8, 5))

# Define two Gaussian distributions with equal covariance
mean1 = np.array([2, 2])
mean2 = np.array([5, 5])
cov = np.array([[1, 0.5], [0.5, 1]])

# Create grid
x = np.linspace(0, 7, 200)
y = np.linspace(0, 7, 200)
X_grid, Y_grid = np.meshgrid(x, y)
pos = np.dstack((X_grid, Y_grid))

# Calculate PDFs
rv1 = multivariate_normal(mean1, cov)
rv2 = multivariate_normal(mean2, cov)
Z1 = rv1.pdf(pos)
Z2 = rv2.pdf(pos)

# Plot contours for both distributions
contour1 = ax2.contour(X_grid, Y_grid, Z1, levels=4, colors='#1f77b4', alpha=0.6, linewidths=2)
contour2 = ax2.contour(X_grid, Y_grid, Z2, levels=4, colors='#d62728', alpha=0.6, linewidths=2)

# Plot means
ax2.plot(mean1[0], mean1[1], 'o', color='#1f77b4', markersize=12, 
        markeredgecolor='k', markeredgewidth=1.5, label='μ₁ (Class 1)')
ax2.plot(mean2[0], mean2[1], 'o', color='#d62728', markersize=12, 
        markeredgecolor='k', markeredgewidth=1.5, label='μ₂ (Class 2)')

# Draw covariance ellipses (at 2 standard deviations)
from matplotlib.patches import Ellipse
def plot_cov_ellipse(mean, cov, ax, n_std=2.0, **kwargs):
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width, height = 2 * n_std * np.sqrt(eigenvalues)
    ellipse = Ellipse(mean, width, height, angle=angle, **kwargs)
    ax.add_patch(ellipse)

plot_cov_ellipse(mean1, cov, ax2, n_std=2, facecolor='none', 
                edgecolor='#1f77b4', linewidth=2, linestyle='--')
plot_cov_ellipse(mean2, cov, ax2, n_std=2, facecolor='none', 
                edgecolor='#d62728', linewidth=2, linestyle='--')

# Add decision boundary (perpendicular bisector of line between means)
midpoint = (mean1 + mean2) / 2
direction = mean2 - mean1
perpendicular = np.array([-direction[1], direction[0]])
perpendicular = perpendicular / np.linalg.norm(perpendicular)
t = np.linspace(-2, 2, 100)
boundary = midpoint[:, np.newaxis] + perpendicular[:, np.newaxis] * t
ax2.plot(boundary[0], boundary[1], 'k--', linewidth=1.5, label='Decision boundary')

ax2.set_xlim(0, 7)
ax2.set_ylim(0, 7)
ax2.set_xlabel('Feature 1')
ax2.set_ylabel('Feature 2')
ax2.set_title('LDA Assumption: Equal Covariance, Different Means')
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('./figure/fig2_gaussian_classes.png', dpi=100, bbox_inches='tight')
plt.close()

print("  ✓ Figure 2 saved")

# =============================================================================
# Figure 3: fig3_lda_projection.png (800x400)
# LDA projection showing dimensionality reduction and separation
# =============================================================================
print("Generating Figure 3: LDA projection...")

# Generate 2D data
n_samples = 150
X_c1 = np.random.multivariate_normal([1, 3], [[0.8, 0.4], [0.4, 0.8]], n_samples)
X_c2 = np.random.multivariate_normal([4, 1], [[0.8, 0.4], [0.4, 0.8]], n_samples)
X_proj = np.vstack([X_c1, X_c2])
y_proj = np.hstack([np.zeros(n_samples), np.ones(n_samples)])

# Fit LDA to get projection direction
lda_proj = LinearDiscriminantAnalysis(n_components=1)
X_transformed = lda_proj.fit_transform(X_proj, y_proj)

# Get projection direction
direction = lda_proj.coef_[0]
direction = direction / np.linalg.norm(direction)

fig3, (ax3a, ax3b) = plt.subplots(2, 1, figsize=(8, 4), 
                                   gridspec_kw={'height_ratios': [2, 1]})

# Top panel: 2D scatter with projection direction
ax3a.scatter(X_c1[:, 0], X_c1[:, 1], c='#1f77b4', alpha=0.6, s=30, label='Class 1')
ax3a.scatter(X_c2[:, 0], X_c2[:, 1], c='#ff7f0e', alpha=0.6, s=30, label='Class 2')

# Draw projection direction arrow
mean_point = X_proj.mean(axis=0)
arrow_length = 2
ax3a.arrow(mean_point[0] - direction[0]*arrow_length/2, 
          mean_point[1] - direction[1]*arrow_length/2,
          direction[0]*arrow_length, direction[1]*arrow_length,
          head_width=0.2, head_length=0.2, fc='black', ec='black', linewidth=2)
ax3a.text(mean_point[0] + direction[0]*arrow_length/2 + 0.3, 
         mean_point[1] + direction[1]*arrow_length/2 + 0.3,
         'LDA projection', fontsize=11, fontweight='bold')

ax3a.set_xlabel('Feature 1')
ax3a.set_ylabel('Feature 2')
ax3a.set_title('LDA Projects Data to Maximize Class Separation')
ax3a.legend()
ax3a.grid(True, alpha=0.3)

# Bottom panel: 1D histogram of projections
ax3b.hist(X_transformed[y_proj==0], bins=30, alpha=0.6, color='#1f77b4', 
         label='Class 1', edgecolor='black', linewidth=0.5)
ax3b.hist(X_transformed[y_proj==1], bins=30, alpha=0.6, color='#ff7f0e', 
         label='Class 2', edgecolor='black', linewidth=0.5)

# Add decision threshold
threshold = (X_transformed[y_proj==0].mean() + X_transformed[y_proj==1].mean()) / 2
ax3b.axvline(threshold, color='black', linestyle='--', linewidth=2, label='Decision threshold')

ax3b.set_xlabel('Projected values')
ax3b.set_ylabel('Frequency')
ax3b.legend()
ax3b.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('./figure/fig3_lda_projection.png', dpi=100, bbox_inches='tight')
plt.close()

print("  ✓ Figure 3 saved")

# =============================================================================
# Figure 4: fig4_lda_decision_boundary.png (800x600)
# LDA decision boundaries on Iris dataset (3 classes)
# =============================================================================
print("Generating Figure 4: LDA decision boundaries on Iris...")

iris = load_iris()
X_iris = iris.data[:, :2]  # First 2 features for visualization
y_iris = iris.target

# Train LDA
lda_iris = LinearDiscriminantAnalysis()
lda_iris.fit(X_iris, y_iris)

# Create mesh for decision boundaries
h = 0.02
x_min, x_max = X_iris[:, 0].min() - 0.5, X_iris[:, 0].max() + 0.5
y_min, y_max = X_iris[:, 1].min() - 0.5, X_iris[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Predict on mesh
Z = lda_iris.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

fig4, ax4 = plt.subplots(figsize=(8, 6))

# Plot decision regions
colors = ['#aec7e8', '#ffbb78', '#98df8a']
ax4.contourf(xx, yy, Z, alpha=0.3, colors=colors, levels=[-0.5, 0.5, 1.5, 2.5])

# Plot decision boundaries
ax4.contour(xx, yy, Z, colors='black', linewidths=1.5, levels=[0.5, 1.5])

# Plot training points
markers = ['o', 's', '^']
colors_dark = ['#1f77b4', '#ff7f0e', '#2ca02c']
class_names = ['Setosa', 'Versicolor', 'Virginica']

for i in range(3):
    mask = y_iris == i
    ax4.scatter(X_iris[mask, 0], X_iris[mask, 1], 
               c=colors_dark[i], marker=markers[i], s=60, alpha=0.8,
               edgecolors='black', linewidth=1, label=class_names[i])

ax4.set_xlabel('Sepal length (cm)')
ax4.set_ylabel('Sepal width (cm)')
ax4.set_title('LDA Decision Boundaries on Iris Dataset')
ax4.legend()
ax4.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('./figure/fig4_lda_decision_boundary.png', dpi=100, bbox_inches='tight')
plt.close()

print("  ✓ Figure 4 saved")

# =============================================================================
# Figure 5: fig5_lda_vs_qda_boundaries.png (1000x500)
# Side-by-side comparison of LDA vs QDA
# =============================================================================
print("Generating Figure 5: LDA vs QDA comparison...")

# Generate synthetic data with different covariances per class
np.random.seed(42)
n_samples = 100
# Class 1: wide in x, narrow in y
X_c1_qda = np.random.multivariate_normal([2, 2], [[2, 0], [0, 0.5]], n_samples)
# Class 2: more circular
X_c2_qda = np.random.multivariate_normal([5, 5], [[1, 0.3], [0.3, 1]], n_samples)

X_qda = np.vstack([X_c1_qda, X_c2_qda])
y_qda = np.hstack([np.zeros(n_samples), np.ones(n_samples)])

# Train both models
lda_model = LinearDiscriminantAnalysis()
qda_model = QuadraticDiscriminantAnalysis()
lda_model.fit(X_qda, y_qda)
qda_model.fit(X_qda, y_qda)

# Create mesh
h = 0.1
x_min, x_max = X_qda[:, 0].min() - 1, X_qda[:, 0].max() + 1
y_min, y_max = X_qda[:, 1].min() - 1, X_qda[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

fig5, (ax5a, ax5b) = plt.subplots(1, 2, figsize=(10, 5))

# Left panel: LDA
Z_lda = lda_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z_lda = Z_lda.reshape(xx.shape)
ax5a.contourf(xx, yy, Z_lda, alpha=0.3, colors=['#aec7e8', '#ffbb78'], levels=[-0.5, 0.5, 1.5])
ax5a.contour(xx, yy, Z_lda, colors='black', linewidths=2, levels=[0.5])
ax5a.scatter(X_c1_qda[:, 0], X_c1_qda[:, 1], c='#1f77b4', s=30, alpha=0.7, edgecolors='k', linewidth=0.5)
ax5a.scatter(X_c2_qda[:, 0], X_c2_qda[:, 1], c='#ff7f0e', s=30, alpha=0.7, edgecolors='k', linewidth=0.5)
ax5a.set_xlabel('Feature 1')
ax5a.set_ylabel('Feature 2')
ax5a.set_title('LDA: Linear Boundary')
ax5a.grid(True, alpha=0.3)

# Right panel: QDA
Z_qda = qda_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z_qda = Z_qda.reshape(xx.shape)
ax5b.contourf(xx, yy, Z_qda, alpha=0.3, colors=['#aec7e8', '#ffbb78'], levels=[-0.5, 0.5, 1.5])
ax5b.contour(xx, yy, Z_qda, colors='black', linewidths=2, levels=[0.5])
ax5b.scatter(X_c1_qda[:, 0], X_c1_qda[:, 1], c='#1f77b4', s=30, alpha=0.7, 
            edgecolors='k', linewidth=0.5, label='Class 1')
ax5b.scatter(X_c2_qda[:, 0], X_c2_qda[:, 1], c='#ff7f0e', s=30, alpha=0.7, 
            edgecolors='k', linewidth=0.5, label='Class 2')
ax5b.set_xlabel('Feature 1')
ax5b.set_ylabel('Feature 2')
ax5b.set_title('QDA: Quadratic Boundary')
ax5b.legend()
ax5b.grid(True, alpha=0.3)

fig5.suptitle('LDA vs QDA Decision Boundaries', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('./figure/fig5_lda_vs_qda_boundaries.png', dpi=100, bbox_inches='tight')
plt.close()

print("  ✓ Figure 5 saved")

# =============================================================================
# Figure 6: fig6_lda_vs_logreg.png (800x500)
# Performance comparison: LDA vs Logistic Regression
# =============================================================================
print("Generating Figure 6: LDA vs Logistic Regression comparison...")

# Generate synthetic dataset for binary classification
X_comp, y_comp = make_classification(n_samples=300, n_features=2, n_redundant=0, 
                                     n_informative=2, n_clusters_per_class=1,
                                     random_state=42, flip_y=0.1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_comp, y_comp, 
                                                      test_size=0.3, random_state=42)

# Train both models
lda_comp = LinearDiscriminantAnalysis()
logreg_comp = LogisticRegression(random_state=42)

lda_comp.fit(X_train, y_train)
logreg_comp.fit(X_train, y_train)

# Calculate accuracies
lda_train_acc = lda_comp.score(X_train, y_train)
lda_test_acc = lda_comp.score(X_test, y_test)
logreg_train_acc = logreg_comp.score(X_train, y_train)
logreg_test_acc = logreg_comp.score(X_test, y_test)

fig6, (ax6a, ax6b) = plt.subplots(1, 2, figsize=(10, 5))

# Left panel: Bar chart of accuracies
x_pos = np.arange(2)
width = 0.35

lda_scores = [lda_train_acc, lda_test_acc]
logreg_scores = [logreg_train_acc, logreg_test_acc]

bars1 = ax6a.bar(x_pos - width/2, lda_scores, width, label='LDA', color='#1f77b4', alpha=0.8)
bars2 = ax6a.bar(x_pos + width/2, logreg_scores, width, label='Logistic Regression', 
                color='#ff7f0e', alpha=0.8)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax6a.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)

ax6a.set_ylabel('Accuracy')
ax6a.set_xticks(x_pos)
ax6a.set_xticklabels(['Training Set', 'Test Set'])
ax6a.set_ylim(0, 1.1)
ax6a.legend()
ax6a.grid(True, alpha=0.3, axis='y')

# Right panel: Decision boundaries
h = 0.02
x_min, x_max = X_comp[:, 0].min() - 0.5, X_comp[:, 0].max() + 0.5
y_min, y_max = X_comp[:, 1].min() - 0.5, X_comp[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Plot training data
ax6b.scatter(X_train[y_train==0, 0], X_train[y_train==0, 1], 
            c='#1f77b4', s=30, alpha=0.6, edgecolors='k', linewidth=0.5, label='Class 0')
ax6b.scatter(X_train[y_train==1, 0], X_train[y_train==1, 1], 
            c='#ff7f0e', s=30, alpha=0.6, edgecolors='k', linewidth=0.5, label='Class 1')

# Plot LDA decision boundary
Z_lda_comp = lda_comp.predict(np.c_[xx.ravel(), yy.ravel()])
Z_lda_comp = Z_lda_comp.reshape(xx.shape)
ax6b.contour(xx, yy, Z_lda_comp, colors='#1f77b4', linewidths=2.5, 
            levels=[0.5], linestyles='solid', label='LDA boundary')

# Plot Logistic Regression decision boundary
Z_logreg_comp = logreg_comp.predict(np.c_[xx.ravel(), yy.ravel()])
Z_logreg_comp = Z_logreg_comp.reshape(xx.shape)
ax6b.contour(xx, yy, Z_logreg_comp, colors='#ff7f0e', linewidths=2.5, 
            levels=[0.5], linestyles='dashed', label='Logistic boundary')

ax6b.set_xlabel('Feature 1')
ax6b.set_ylabel('Feature 2')
ax6b.legend()
ax6b.grid(True, alpha=0.3)

fig6.suptitle('LDA vs Logistic Regression Comparison', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('./figure/fig6_lda_vs_logreg.png', dpi=100, bbox_inches='tight')
plt.close()

print("  ✓ Figure 6 saved")

print("\n" + "="*60)
print("All figures generated successfully!")
print("Figures saved in ./figure/ directory:")
print("  • fig1_iris_species.png")
print("  • fig2_gaussian_classes.png")
print("  • fig3_lda_projection.png")
print("  • fig4_lda_decision_boundary.png")
print("  • fig5_lda_vs_qda_boundaries.png")
print("  • fig6_lda_vs_logreg.png")
print("="*60)
