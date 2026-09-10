"""
Figure Generation Script for Lesson 5: Linear Discriminant Analysis
Generates all 7 figures referenced in the lecture notes
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from matplotlib.patches import Ellipse
import os

# Create figure directory if it doesn't exist
os.makedirs('./figure', exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

# Set style for better-looking plots
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 10


def generate_2d_data(n_samples=100, mean0=(2, 3), mean1=(5, 6), cov0=None, cov1=None):
    """Generate 2D binary classification data"""
    if cov0 is None:
        cov0 = [[1.0, 0.5], [0.5, 1.0]]
    if cov1 is None:
        cov1 = [[1.0, 0.5], [0.5, 1.0]]
    
    X0 = np.random.multivariate_normal(mean0, cov0, n_samples)
    X1 = np.random.multivariate_normal(mean1, cov1, n_samples)
    
    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(n_samples), np.ones(n_samples)])
    
    return X, y


def plot_decision_boundary(clf, X, y, ax, title='', show_regions=True):
    """Plot decision boundary for a 2D classifier"""
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    if show_regions:
        ax.contourf(xx, yy, Z, alpha=0.3, levels=[0, 0.5, 1], colors=['blue', 'red'])
    ax.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)


# ============================================================================
# Figure 1: Projection Motivation
# ============================================================================
def generate_fig1():
    """Generate fig1_projection_motivation.png"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    # Generate data with some overlap
    X, y = generate_2d_data(n_samples=100, mean0=(2, 3), mean1=(5, 6))
    
    # Fit LDA to get optimal projection direction
    lda = LinearDiscriminantAnalysis()
    lda.fit(X, y)
    
    # Left plot: 2D scatter with projection line
    ax1.scatter(X[y==0, 0], X[y==0, 1], c='blue', alpha=0.6, s=40, label='Class 0')
    ax1.scatter(X[y==1, 0], X[y==1, 1], c='red', alpha=0.6, s=40, label='Class 1')
    
    # Draw projection direction
    w = lda.coef_[0]
    w = w / np.linalg.norm(w)
    
    # Draw the projection line
    center = X.mean(axis=0)
    scale = 4
    x_line = [center[0] - scale*w[0], center[0] + scale*w[0]]
    y_line = [center[1] - scale*w[1], center[1] + scale*w[1]]
    ax1.plot(x_line, y_line, 'g-', linewidth=3, label='Projection direction', alpha=0.7)
    ax1.arrow(center[0], center[1], 2*w[0], 2*w[1], 
              head_width=0.3, head_length=0.2, fc='green', ec='green', linewidth=2)
    
    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')
    ax1.set_title('Original 2D Data')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Right plot: 1D projection showing separation
    X_proj = X @ w
    
    # Create histogram-like visualization
    bins = np.linspace(X_proj.min(), X_proj.max(), 30)
    ax2.hist(X_proj[y==0], bins=bins, alpha=0.6, color='blue', label='Class 0', edgecolor='black')
    ax2.hist(X_proj[y==1], bins=bins, alpha=0.6, color='red', label='Class 1', edgecolor='black')
    
    ax2.set_xlabel('Projected value')
    ax2.set_ylabel('Frequency')
    ax2.set_title('After Projection (1D)')
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.axvline(X_proj[y==0].mean(), color='blue', linestyle='--', linewidth=2, alpha=0.7)
    ax2.axvline(X_proj[y==1].mean(), color='red', linestyle='--', linewidth=2, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('./figures/fig1_projection_motivation.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("✓ Generated fig1_projection_motivation.png")


# ============================================================================
# Figure 2: Good vs Bad Projections
# ============================================================================
def generate_fig2():
    """Generate fig2_good_bad_projections.png"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    # Generate data
    X, y = generate_2d_data(n_samples=80, mean0=(2, 3), mean1=(5, 6))
    
    # Bad projection: nearly horizontal
    w_bad = np.array([1.0, 0.1])
    w_bad = w_bad / np.linalg.norm(w_bad)
    
    # Good projection: diagonal (LDA optimal)
    lda = LinearDiscriminantAnalysis()
    lda.fit(X, y)
    w_good = lda.coef_[0]
    w_good = w_good / np.linalg.norm(w_good)
    
    # Left plot: Bad projection
    ax1.scatter(X[y==0, 0], X[y==0, 1], c='blue', alpha=0.6, s=40, label='Class 0')
    ax1.scatter(X[y==1, 0], X[y==1, 1], c='red', alpha=0.6, s=40, label='Class 1')
    
    center = X.mean(axis=0)
    scale = 3
    ax1.plot([center[0] - scale*w_bad[0], center[0] + scale*w_bad[0]],
             [center[1] - scale*w_bad[1], center[1] + scale*w_bad[1]],
             'purple', linewidth=3, alpha=0.7, label='Bad projection')
    ax1.arrow(center[0], center[1], 1.5*w_bad[0], 1.5*w_bad[1],
              head_width=0.3, head_length=0.2, fc='purple', ec='purple', linewidth=2)
    
    # Show projected points below
    X_proj_bad = X @ w_bad
    y_offset = X[:, 1].min() - 1.5
    for i in range(len(X)):
        color = 'blue' if y[i] == 0 else 'red'
        proj_point = center + (X_proj_bad[i] - X_proj_bad.mean()) * w_bad
        ax1.scatter(proj_point[0], y_offset, c=color, s=30, alpha=0.4)
    
    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')
    ax1.set_title('Bad Projection\n(Much overlap)')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Right plot: Good projection
    ax2.scatter(X[y==0, 0], X[y==0, 1], c='blue', alpha=0.6, s=40, label='Class 0')
    ax2.scatter(X[y==1, 0], X[y==1, 1], c='red', alpha=0.6, s=40, label='Class 1')
    
    ax2.plot([center[0] - scale*w_good[0], center[0] + scale*w_good[0]],
             [center[1] - scale*w_good[1], center[1] + scale*w_good[1]],
             'green', linewidth=3, alpha=0.7, label='Good projection')
    ax2.arrow(center[0], center[1], 1.5*w_good[0], 1.5*w_good[1],
              head_width=0.3, head_length=0.2, fc='green', ec='green', linewidth=2)
    
    # Show projected points below
    X_proj_good = X @ w_good
    for i in range(len(X)):
        color = 'blue' if y[i] == 0 else 'red'
        proj_point = center + (X_proj_good[i] - X_proj_good.mean()) * w_good
        ax2.scatter(proj_point[0], y_offset, c=color, s=30, alpha=0.4)
    
    ax2.set_xlabel('Feature 1')
    ax2.set_ylabel('Feature 2')
    ax2.set_title('Good Projection\n(Clear separation)')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./figures/fig2_good_bad_projections.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("✓ Generated fig2_good_bad_projections.png")


# ============================================================================
# Figure 3: Between-Class and Within-Class Variance
# ============================================================================
def generate_fig3():
    """Generate fig3_variance_components.png"""
    fig, ax = plt.subplots(figsize=(10, 6.25))
    
    # Create x-axis for projected space
    x = np.linspace(-4, 8, 1000)
    
    # Class 0: mean at 0, variance 1
    mu0 = 0.0
    sigma0 = 1.0
    y0 = (1/(sigma0 * np.sqrt(2*np.pi))) * np.exp(-0.5*((x - mu0)/sigma0)**2)
    
    # Class 1: mean at 4, variance 1
    mu1 = 4.0
    sigma1 = 1.0
    y1 = (1/(sigma1 * np.sqrt(2*np.pi))) * np.exp(-0.5*((x - mu1)/sigma1)**2)
    
    # Plot distributions
    ax.plot(x, y0, 'b-', linewidth=3, label='Class 0')
    ax.fill_between(x, 0, y0, alpha=0.3, color='blue')
    ax.plot(x, y1, 'r-', linewidth=3, label='Class 1')
    ax.fill_between(x, 0, y1, alpha=0.3, color='red')
    
    # Mark means
    ax.axvline(mu0, color='blue', linestyle='--', linewidth=2, alpha=0.7)
    ax.axvline(mu1, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.text(mu0, 0.45, r'$\tilde{\mu}_0$', fontsize=14, ha='center', 
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    ax.text(mu1, 0.45, r'$\tilde{\mu}_1$', fontsize=14, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    # Between-class variance arrow
    ax.annotate('', xy=(mu1, 0.35), xytext=(mu0, 0.35),
                arrowprops=dict(arrowstyle='<->', color='black', lw=3))
    ax.text((mu0 + mu1)/2, 0.38, 'Between-class\nvariance\n(want LARGE)', 
            fontsize=11, ha='center', va='bottom',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # Within-class variance annotations
    # For class 0
    ax.annotate('', xy=(mu0 - sigma0, 0.15), xytext=(mu0 + sigma0, 0.15),
                arrowprops=dict(arrowstyle='<->', color='blue', lw=2))
    ax.text(mu0, 0.17, 'Within-class\nvariance\n(want small)', 
            fontsize=9, ha='center', va='bottom', color='blue',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.6))
    
    # For class 1
    ax.annotate('', xy=(mu1 - sigma1, 0.15), xytext=(mu1 + sigma1, 0.15),
                arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax.text(mu1, 0.17, 'Within-class\nvariance\n(want small)', 
            fontsize=9, ha='center', va='bottom', color='red',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.6))
    
    ax.set_xlabel('Projected Space (z)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title("Components of Fisher's Criterion", fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_ylim([0, 0.5])
    
    plt.tight_layout()
    plt.savefig('./figures/fig3_variance_components.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("✓ Generated fig3_variance_components.png")


# ============================================================================
# Figure 4: LDA Decision Boundary and Projection Direction
# ============================================================================
def generate_fig4():
    """Generate fig4_lda_decision_boundary.png"""
    fig, ax = plt.subplots(figsize=(8.75, 7.5))
    
    # Generate data
    X, y = generate_2d_data(n_samples=100, mean0=(2, 3), mean1=(5, 6))
    
    # Fit LDA
    lda = LinearDiscriminantAnalysis()
    lda.fit(X, y)
    
    # Plot decision boundary
    plot_decision_boundary(lda, X, y, ax, show_regions=True)
    
    # Plot data points
    ax.scatter(X[y==0, 0], X[y==0, 1], c='blue', alpha=0.7, s=60, 
               edgecolors='black', linewidths=0.5, label='Class 0')
    ax.scatter(X[y==1, 0], X[y==1, 1], c='red', alpha=0.7, s=60, 
               edgecolors='black', linewidths=0.5, label='Class 1')
    
    # Plot class means
    mu0 = X[y==0].mean(axis=0)
    mu1 = X[y==1].mean(axis=0)
    ax.scatter(*mu0, c='blue', marker='*', s=400, edgecolors='black', 
               linewidths=2, label=r'$\mu_0$', zorder=5)
    ax.scatter(*mu1, c='red', marker='*', s=400, edgecolors='black', 
               linewidths=2, label=r'$\mu_1$', zorder=5)
    
    # Draw projection direction (perpendicular to decision boundary)
    w = lda.coef_[0]
    w = w / np.linalg.norm(w)
    
    center = X.mean(axis=0)
    scale = 3
    ax.arrow(center[0] - scale*w[0], center[1] - scale*w[1], 
             2*scale*w[0], 2*scale*w[1],
             head_width=0.3, head_length=0.3, fc='green', ec='green', 
             linewidth=3, alpha=0.8, label='Projection direction w')
    
    # Draw some projection lines (dotted)
    for i in range(0, len(X), 15):
        proj = (X[i] @ w) * w
        ax.plot([X[i, 0], center[0] + (proj[0] - center[0])],
                [X[i, 1], center[1] + (proj[1] - center[1])],
                'gray', linestyle=':', linewidth=0.5, alpha=0.5)
    
    ax.set_xlabel('Feature 1', fontsize=12)
    ax.set_ylabel('Feature 2', fontsize=12)
    ax.set_title('LDA Decision Boundary and Projection Direction', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./figures/fig4_lda_decision_boundary.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("✓ Generated fig4_lda_decision_boundary.png")


# ============================================================================
# Figure 5: sklearn LDA Example (Iris Dataset)
# ============================================================================
def generate_fig5():
    """Generate fig5_sklearn_lda_example.png"""
    fig, ax = plt.subplots(figsize=(8.75, 7.5))
    
    # Load iris dataset (first 2 classes, first 2 features)
    iris = load_iris()
    X = iris.data[:100, :2]  # sepal length and width
    y = iris.target[:100]
    
    # Split into train and test (simple split)
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    train_idx = indices[:70]
    test_idx = indices[70:]
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    # Fit LDA
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)
    
    # Calculate accuracy
    accuracy = lda.score(X_test, y_test)
    
    # Plot decision boundary
    plot_decision_boundary(lda, X, y, ax, show_regions=True)
    
    # Plot training data
    ax.scatter(X_train[y_train==0, 0], X_train[y_train==0, 1], 
               c='blue', marker='o', s=80, alpha=0.7, edgecolors='black',
               linewidths=1, label='Train Class 0 (Setosa)')
    ax.scatter(X_train[y_train==1, 0], X_train[y_train==1, 1], 
               c='red', marker='o', s=80, alpha=0.7, edgecolors='black',
               linewidths=1, label='Train Class 1 (Versicolor)')
    
    # Plot test data with different markers
    ax.scatter(X_test[y_test==0, 0], X_test[y_test==0, 1], 
               c='blue', marker='^', s=100, alpha=0.9, edgecolors='black',
               linewidths=1.5, label='Test Class 0')
    ax.scatter(X_test[y_test==1, 0], X_test[y_test==1, 1], 
               c='red', marker='^', s=100, alpha=0.9, edgecolors='black',
               linewidths=1.5, label='Test Class 1')
    
    # Mark any misclassified points
    y_pred_test = lda.predict(X_test)
    misclassified = y_pred_test != y_test
    if misclassified.any():
        ax.scatter(X_test[misclassified, 0], X_test[misclassified, 1],
                   c='none', marker='X', s=200, edgecolors='black',
                   linewidths=3, label='Misclassified')
    
    ax.set_xlabel('Sepal length (cm)', fontsize=12)
    ax.set_ylabel('Sepal width (cm)', fontsize=12)
    ax.set_title(f'LDA Classification on Iris Dataset\nAccuracy: {accuracy:.1%}', 
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./figures/fig5_sklearn_lda_example.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("✓ Generated fig5_sklearn_lda_example.png")


# ============================================================================
# Figure 6: LDA vs QDA Decision Boundaries
# ============================================================================
def generate_fig6():
    """Generate fig6_lda_vs_qda_boundaries.png"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    # Generate data with different covariances
    # Class 0: circular (isotropic)
    cov0 = [[1.0, 0.0], [0.0, 1.0]]
    # Class 1: elliptical (stretched)
    cov1 = [[2.5, 1.5], [1.5, 2.0]]
    
    X, y = generate_2d_data(n_samples=150, mean0=(2, 3), mean1=(6, 6), 
                            cov0=cov0, cov1=cov1)
    
    # Fit LDA
    lda = LinearDiscriminantAnalysis()
    lda.fit(X, y)
    
    # Fit QDA
    qda = QuadraticDiscriminantAnalysis()
    qda.fit(X, y)
    
    # Left plot: LDA
    plot_decision_boundary(lda, X, y, ax1, show_regions=True)
    ax1.scatter(X[y==0, 0], X[y==0, 1], c='blue', alpha=0.6, s=50, 
                edgecolors='black', linewidths=0.5)
    ax1.scatter(X[y==1, 0], X[y==1, 1], c='red', alpha=0.6, s=50, 
                edgecolors='black', linewidths=0.5)
    ax1.set_xlabel('Feature 1', fontsize=11)
    ax1.set_ylabel('Feature 2', fontsize=11)
    ax1.set_title('LDA (Linear Boundary)\nAssumes equal covariance', fontsize=12, fontweight='bold')
    ax1.grid(alpha=0.3)
    
    # Right plot: QDA
    plot_decision_boundary(qda, X, y, ax2, show_regions=True)
    ax2.scatter(X[y==0, 0], X[y==0, 1], c='blue', alpha=0.6, s=50, 
                edgecolors='black', linewidths=0.5, label='Class 0 (circular)')
    ax2.scatter(X[y==1, 0], X[y==1, 1], c='red', alpha=0.6, s=50, 
                edgecolors='black', linewidths=0.5, label='Class 1 (elliptical)')
    ax2.set_xlabel('Feature 1', fontsize=11)
    ax2.set_ylabel('Feature 2', fontsize=11)
    ax2.set_title('QDA (Quadratic Boundary)\nAllows different covariances', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./figures/fig6_lda_vs_qda_boundaries.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("✓ Generated fig6_lda_vs_qda_boundaries.png")


# ============================================================================
# Figure 7: LDA vs Logistic Regression Comparison
# ============================================================================
def generate_fig7():
    """Generate fig7_lda_vs_logreg.png"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    # Generate data
    X, y = generate_2d_data(n_samples=100, mean0=(2, 3), mean1=(5, 6))
    
    # Fit LDA
    lda = LinearDiscriminantAnalysis()
    lda.fit(X, y)
    lda_acc = lda.score(X, y)
    
    # Fit Logistic Regression
    logreg = LogisticRegression()
    logreg.fit(X, y)
    logreg_acc = logreg.score(X, y)
    
    # Left plot: LDA
    plot_decision_boundary(lda, X, y, ax1, show_regions=True)
    ax1.scatter(X[y==0, 0], X[y==0, 1], c='blue', alpha=0.7, s=60, 
                edgecolors='black', linewidths=0.5, label='Class 0')
    ax1.scatter(X[y==1, 0], X[y==1, 1], c='red', alpha=0.7, s=60, 
                edgecolors='black', linewidths=0.5, label='Class 1')
    ax1.set_xlabel('Feature 1', fontsize=11)
    ax1.set_ylabel('Feature 2', fontsize=11)
    ax1.set_title(f'LDA\n(Accuracy: {lda_acc:.1%})', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    
    # Right plot: Logistic Regression
    plot_decision_boundary(logreg, X, y, ax2, show_regions=True)
    ax2.scatter(X[y==0, 0], X[y==0, 1], c='blue', alpha=0.7, s=60, 
                edgecolors='black', linewidths=0.5, label='Class 0')
    ax2.scatter(X[y==1, 0], X[y==1, 1], c='red', alpha=0.7, s=60, 
                edgecolors='black', linewidths=0.5, label='Class 1')
    ax2.set_xlabel('Feature 1', fontsize=11)
    ax2.set_ylabel('Feature 2', fontsize=11)
    ax2.set_title(f'Logistic Regression\n(Accuracy: {logreg_acc:.1%})', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    
    # Add note at bottom
    fig.text(0.5, 0.02, 'Both produce linear boundaries, but from different approaches', 
             ha='center', fontsize=11, style='italic',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig('./figures/fig7_lda_vs_logreg.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("✓ Generated fig7_lda_vs_logreg.png")


# ============================================================================
# Main execution
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Generating figures for Lesson 5: Linear Discriminant Analysis")
    print("=" * 60)
    print()
    
    generate_fig1()
    generate_fig2()
    generate_fig3()
    generate_fig4()
    generate_fig5()
    generate_fig6()
    generate_fig7()
    
    print()
    print("=" * 60)
    print("✅ All figures generated successfully!")
    print("=" * 60)
    print(f"Figures saved to: ./figures/")
    print()
    print("Generated files:")
    print("  - fig1_projection_motivation.png")
    print("  - fig2_good_bad_projections.png")
    print("  - fig3_variance_components.png")
    print("  - fig4_lda_decision_boundary.png")
    print("  - fig5_sklearn_lda_example.png")
    print("  - fig6_lda_vs_qda_boundaries.png")
    print("  - fig7_lda_vs_logreg.png")
