"""
Figure Generation Script for Lesson 6: Decision Trees Lecture Notes
MPS311/439 - Machine Learning Course

This script generates all figures needed for the lecture notes.
Figures are saved to ./figures/ directory with unique names.

Author: Dr. Wei Xing
Date: November 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_moons, make_circles, load_wine, load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import xgboost as xgb
import os
import time
import warnings
warnings.filterwarnings('ignore')

# Create figures directory if it doesn't exist
os.makedirs('./figures', exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

# Set matplotlib style for better-looking plots
plt.style.use('default')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


def generate_fig_xor_linear_fail():
    """
    Figure 1: XOR Problem showing where linear classifiers fail
    """
    print("Generating Figure 1: XOR Linear Fail...")
    
    # Generate XOR dataset
    n_samples = 50
    noise = 0.15
    
    # Four clusters at corners
    cluster1 = np.random.randn(n_samples, 2) * noise + [1, 1]  # bottom-left, Class A
    cluster2 = np.random.randn(n_samples, 2) * noise + [4, 4]  # top-right, Class A
    cluster3 = np.random.randn(n_samples, 2) * noise + [1, 4]  # top-left, Class B
    cluster4 = np.random.randn(n_samples, 2) * noise + [4, 1]  # bottom-right, Class B
    
    X = np.vstack([cluster1, cluster2, cluster3, cluster4])
    y = np.array([0]*n_samples + [0]*n_samples + [1]*n_samples + [1]*n_samples)
    
    # Train linear classifier
    log_reg = LogisticRegression()
    log_reg.fit(X, y)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot data points
    scatter_a = ax.scatter(X[y==0, 0], X[y==0, 1], c='blue', s=50, alpha=0.7, 
                          edgecolors='black', linewidth=0.5, label='Class A')
    scatter_b = ax.scatter(X[y==1, 0], X[y==1, 1], c='red', s=50, alpha=0.7, 
                          edgecolors='black', linewidth=0.5, label='Class B')
    
    # Plot decision boundary
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    Z = log_reg.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot boundary
    ax.contour(xx, yy, Z, levels=[0.5], colors='green', linewidths=2.5, 
               linestyles='--', label='Linear Boundary')
    
    # Mark misclassified points
    y_pred = log_reg.predict(X)
    errors = y_pred != y
    if errors.sum() > 0:
        ax.scatter(X[errors, 0], X[errors, 1], s=200, facecolors='none', 
                  edgecolors='orange', linewidth=3, marker='x', 
                  label='Errors', zorder=5)
    
    ax.set_xlabel('Feature 1', fontsize=12, fontweight='bold')
    ax.set_ylabel('Feature 2', fontsize=12, fontweight='bold')
    ax.set_title('XOR Problem: Where Linear Classifiers Fail', 
                fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    plt.savefig('./figures/fig_xor_linear_fail.png', bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: fig_xor_linear_fail.png")


def generate_fig_nested_boundaries():
    """
    Figure 2: Nested/circular class boundaries that challenge linear methods
    """
    print("Generating Figure 2: Nested Boundaries...")
    
    # Generate nested circular dataset
    n_samples = 100
    
    # Inner circle - Class A
    theta_inner = np.random.uniform(0, 2*np.pi, n_samples)
    r_inner = np.random.uniform(0, 2, n_samples)
    X_inner = np.column_stack([r_inner * np.cos(theta_inner), 
                               r_inner * np.sin(theta_inner)])
    y_inner = np.zeros(n_samples)
    
    # Outer ring - Class B
    theta_outer = np.random.uniform(0, 2*np.pi, n_samples)
    r_outer = np.random.uniform(2, 4, n_samples)
    X_outer = np.column_stack([r_outer * np.cos(theta_outer), 
                               r_outer * np.sin(theta_outer)])
    y_outer = np.ones(n_samples)
    
    X = np.vstack([X_inner, X_outer])
    y = np.hstack([y_inner, y_outer])
    
    # Train QDA
    qda = QuadraticDiscriminantAnalysis()
    qda.fit(X, y)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot decision boundary
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    Z = qda.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot boundary as contour
    ax.contour(xx, yy, Z, levels=[0.5], colors='purple', linewidths=2.5, 
               linestyles='--', label='QDA Boundary')
    
    # Plot data points
    ax.scatter(X[y==0, 0], X[y==0, 1], c='blue', s=50, alpha=0.7, 
              edgecolors='black', linewidth=0.5, label='Class A (inner)')
    ax.scatter(X[y==1, 0], X[y==1, 1], c='red', s=50, alpha=0.7, 
              edgecolors='black', linewidth=0.5, label='Class B (outer)')
    
    ax.set_xlabel('Feature 1', fontsize=12, fontweight='bold')
    ax.set_ylabel('Feature 2', fontsize=12, fontweight='bold')
    ax.set_title('Nested Class Boundaries Challenge Linear Methods', 
                fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('./figures/fig_nested_boundaries.png', bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: fig_nested_boundaries.png")


def generate_fig_simple_tree_example():
    """
    Figure 3: Simple decision tree example on Iris dataset
    """
    print("Generating Figure 3: Simple Tree Example...")
    
    # Load Iris dataset
    iris = load_iris()
    X, y = iris.data[:, [2, 3]], iris.target  # Use petal length and width
    
    # Train decision tree with depth=3
    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(X, y)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot tree
    plot_tree(clf, filled=True, rounded=True, 
             feature_names=['petal length', 'petal width'],
             class_names=iris.target_names,
             fontsize=10, ax=ax)
    
    ax.set_title('Decision Tree for Iris Classification (depth=3)', 
                fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('./figures/fig_simple_tree_example.png', bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: fig_simple_tree_example.png")


def generate_fig_gini_entropy_comparison():
    """
    Figure 4: Comparison of Gini impurity and Entropy
    """
    print("Generating Figure 4: Gini vs Entropy Comparison...")
    
    # Generate probability values
    p = np.linspace(0.001, 0.999, 100)
    
    # Calculate Gini impurity (for binary classification)
    gini = 2 * p * (1 - p)
    
    # Calculate Entropy
    entropy = -(p * np.log2(p) + (1-p) * np.log2(1-p))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Plot both curves
    ax.plot(p, gini, 'b-', linewidth=2.5, label='Gini Impurity = 2p(1-p)')
    ax.plot(p, entropy, 'r--', linewidth=2.5, label='Entropy = -p log₂(p) - (1-p) log₂(1-p)')
    
    # Add vertical line at p=0.5
    ax.axvline(x=0.5, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.text(0.5, 0.95, 'Maximum impurity\nat p=0.5', 
           ha='center', va='bottom', fontsize=10, 
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel('Probability of Class 1 (p)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Impurity Value', fontsize=12, fontweight='bold')
    ax.set_title('Gini Impurity vs Entropy for Binary Classification', 
                fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='upper center', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig('./figures/fig_gini_entropy_comparison.png', bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: fig_gini_entropy_comparison.png")


def generate_fig_tree_decision_boundary():
    """
    Figure 5: Decision boundaries for shallow vs deep trees
    """
    print("Generating Figure 5: Tree Decision Boundaries...")
    
    # Generate moons dataset
    X, y = make_moons(n_samples=200, noise=0.25, random_state=42)
    
    # Train two trees with different depths
    clf_shallow = DecisionTreeClassifier(max_depth=2, random_state=42)
    clf_deep = DecisionTreeClassifier(max_depth=8, random_state=42)
    
    clf_shallow.fit(X, y)
    clf_deep.fit(X, y)
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot decision boundaries
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    
    for idx, (clf, title, ax) in enumerate([
        (clf_shallow, 'Shallow Tree (depth=2)', axes[0]),
        (clf_deep, 'Deep Tree (depth=8)', axes[1])
    ]):
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Plot decision regions
        ax.contourf(xx, yy, Z, alpha=0.3, levels=1, colors=['blue', 'red'])
        
        # Plot data points
        ax.scatter(X[y==0, 0], X[y==0, 1], c='blue', s=40, alpha=0.8, 
                  edgecolors='black', linewidth=0.5, label='Class 0')
        ax.scatter(X[y==1, 0], X[y==1, 1], c='red', s=40, alpha=0.8, 
                  edgecolors='black', linewidth=0.5, label='Class 1')
        
        ax.set_xlabel('Feature 1', fontsize=11, fontweight='bold')
        ax.set_ylabel('Feature 2', fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
    
    fig.suptitle('Decision Boundaries: Shallow vs Deep Trees', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('./figures/fig_tree_decision_boundary.png', bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: fig_tree_decision_boundary.png")


def generate_fig_overfitting_comparison():
    """
    Figure 6: Comprehensive overfitting comparison (2x2 grid)
    """
    print("Generating Figure 6: Overfitting Comparison...")
    
    # Generate classification dataset
    X, y = make_moons(n_samples=300, noise=0.25, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # --- Top-left: Shallow tree decision boundary ---
    ax = axes[0, 0]
    clf_shallow = DecisionTreeClassifier(max_depth=2, random_state=42)
    clf_shallow.fit(X_train, y_train)
    
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    
    Z = clf_shallow.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.3, levels=1, colors=['blue', 'red'])
    ax.scatter(X_train[y_train==0, 0], X_train[y_train==0, 1], c='blue', s=30, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax.scatter(X_train[y_train==1, 0], X_train[y_train==1, 1], c='red', s=30, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax.set_xlabel('Feature 1', fontsize=10, fontweight='bold')
    ax.set_ylabel('Feature 2', fontsize=10, fontweight='bold')
    ax.set_title('Shallow Tree (depth=2): Underfitting', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # --- Top-right: Deep tree decision boundary ---
    ax = axes[0, 1]
    clf_deep = DecisionTreeClassifier(max_depth=15, random_state=42)
    clf_deep.fit(X_train, y_train)
    
    Z = clf_deep.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.3, levels=1, colors=['blue', 'red'])
    ax.scatter(X_train[y_train==0, 0], X_train[y_train==0, 1], c='blue', s=30, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax.scatter(X_train[y_train==1, 0], X_train[y_train==1, 1], c='red', s=30, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax.set_xlabel('Feature 1', fontsize=10, fontweight='bold')
    ax.set_ylabel('Feature 2', fontsize=10, fontweight='bold')
    ax.set_title('Deep Tree (depth=15): Overfitting', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # --- Bottom-left: Training vs Test accuracy ---
    ax = axes[1, 0]
    depths = range(1, 21)
    train_accs = []
    test_accs = []
    
    for depth in depths:
        clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
        clf.fit(X_train, y_train)
        train_accs.append(clf.score(X_train, y_train))
        test_accs.append(clf.score(X_test, y_test))
    
    ax.plot(depths, train_accs, 'b-', linewidth=2.5, label='Training Accuracy', marker='o', markersize=4)
    ax.plot(depths, test_accs, 'r--', linewidth=2.5, label='Test Accuracy', marker='s', markersize=4)
    
    # Mark optimal depth
    optimal_depth = np.argmax(test_accs) + 1
    ax.axvline(x=optimal_depth, color='green', linestyle=':', linewidth=2, alpha=0.7)
    ax.text(optimal_depth, 0.5, f'Optimal\ndepth={optimal_depth}', 
           ha='center', fontsize=9, 
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    ax.set_xlabel('Tree Depth', fontsize=10, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=10, fontweight='bold')
    ax.set_title('Training vs Test Accuracy', fontsize=11, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, 1.05)
    
    # --- Bottom-right: Training set Gini impurity ---
    ax = axes[1, 1]
    gini_scores = []
    
    for depth in depths:
        clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
        clf.fit(X_train, y_train)
        # Get average Gini impurity of leaves (weighted by samples)
        tree = clf.tree_
        n_nodes = tree.node_count
        gini_sum = 0
        total_samples = 0
        for node_id in range(n_nodes):
            if tree.children_left[node_id] == tree.children_right[node_id]:  # leaf node
                gini_sum += tree.impurity[node_id] * tree.n_node_samples[node_id]
                total_samples += tree.n_node_samples[node_id]
        avg_gini = gini_sum / total_samples if total_samples > 0 else 0
        gini_scores.append(avg_gini)
    
    ax.plot(depths, gini_scores, 'purple', linewidth=2.5, marker='D', markersize=5)
    ax.set_xlabel('Tree Depth', fontsize=10, fontweight='bold')
    ax.set_ylabel('Average Gini Impurity', fontsize=10, fontweight='bold')
    ax.set_title('Training Set Gini Impurity', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    fig.suptitle('The Overfitting Problem in Decision Trees', 
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('./figures/fig_overfitting_comparison.png', bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: fig_overfitting_comparison.png")


def generate_fig_tree_instability():
    """
    Figure 7: Tree instability with bootstrap samples
    """
    print("Generating Figure 7: Tree Instability...")
    
    # Generate base dataset
    X, y = make_moons(n_samples=200, noise=0.25, random_state=42)
    
    # Create figure with 2x3 grid (only use 5 panels)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Create 5 bootstrap samples and train trees
    colors = ['blue', 'green', 'red', 'orange', 'purple']
    
    for i in range(5):
        ax = axes[i]
        
        # Bootstrap sample
        indices = np.random.choice(len(X), size=len(X), replace=True)
        X_boot = X[indices]
        y_boot = y[indices]
        
        # Train tree
        clf = DecisionTreeClassifier(max_depth=5, random_state=i)
        clf.fit(X_boot, y_boot)
        
        # Plot decision boundary
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                             np.linspace(y_min, y_max, 200))
        
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Plot with unique color for this bootstrap
        ax.contourf(xx, yy, Z, alpha=0.3, levels=1, cmap='RdYlBu')
        ax.contour(xx, yy, Z, levels=[0.5], colors=colors[i], linewidths=3, linestyles='-')
        
        # Plot original data points (semi-transparent)
        ax.scatter(X[y==0, 0], X[y==0, 1], c='blue', s=30, alpha=0.4, edgecolors='black', linewidth=0.3)
        ax.scatter(X[y==1, 0], X[y==1, 1], c='red', s=30, alpha=0.4, edgecolors='black', linewidth=0.3)
        
        ax.set_xlabel('Feature 1', fontsize=10, fontweight='bold')
        ax.set_ylabel('Feature 2', fontsize=10, fontweight='bold')
        ax.set_title(f'Bootstrap Sample {i+1}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
    
    # Hide the 6th panel
    axes[5].axis('off')
    
    fig.suptitle('Tree Instability: Different Trees from Bootstrapped Data', 
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('./figures/fig_tree_instability.png', bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: fig_tree_instability.png")


def generate_fig_ensemble_comparison():
    """
    Figure 8: Ensemble methods comparison (accuracy and training time)
    """
    print("Generating Figure 8: Ensemble Comparison...")
    
    # Load Wine dataset
    wine = load_wine()
    X, y = wine.data, wine.target
    
    # Define models
    models = {
        'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
        'XGBoost': xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, 
                                     random_state=42, verbosity=0)
    }
    
    # Evaluate models
    results = {}
    for name, model in models.items():
        # Cross-validation for accuracy
        start = time.time()
        cv_scores = cross_val_score(model, X, y, cv=5)
        train_time = time.time() - start
        
        results[name] = {
            'accuracy': cv_scores.mean(),
            'std': cv_scores.std(),
            'time': train_time
        }
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # --- Top panel: Accuracy comparison ---
    ax = axes[0]
    model_names = list(results.keys())
    accuracies = [results[name]['accuracy'] for name in model_names]
    stds = [results[name]['std'] for name in model_names]
    
    colors_bar = ['skyblue', 'lightgreen', 'salmon']
    bars = ax.barh(model_names, accuracies, xerr=stds, color=colors_bar, 
                   edgecolor='black', linewidth=1.5, capsize=5)
    
    # Annotate bars with values
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        ax.text(acc + stds[i] + 0.01, bar.get_y() + bar.get_height()/2, 
               f'{acc:.3f}', va='center', fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Cross-Validation Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Model Accuracy Comparison (5-Fold CV)', fontsize=13, fontweight='bold', pad=10)
    ax.set_xlim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='x')
    
    # --- Bottom panel: Training time comparison ---
    ax = axes[1]
    times = [results[name]['time'] for name in model_names]
    
    bars = ax.barh(model_names, times, color=colors_bar, 
                   edgecolor='black', linewidth=1.5)
    
    # Annotate bars with values
    for bar, t in zip(bars, times):
        ax.text(t + 0.01, bar.get_y() + bar.get_height()/2, 
               f'{t:.2f}s', va='center', fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Training Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Training Time Comparison', fontsize=13, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3, axis='x')
    
    fig.suptitle('Decision Tree vs Random Forest vs XGBoost', 
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('./figures/fig_ensemble_comparison.png', bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: fig_ensemble_comparison.png")


def main():
    """
    Main function to generate all figures
    """
    print("\n" + "="*60)
    print("FIGURE GENERATION FOR LESSON 6: DECISION TREES")
    print("MPS311/439 - Machine Learning Course")
    print("="*60 + "\n")
    
    print("Starting figure generation...\n")
    
    try:
        generate_fig_xor_linear_fail()
        generate_fig_nested_boundaries()
        generate_fig_simple_tree_example()
        generate_fig_gini_entropy_comparison()
        generate_fig_tree_decision_boundary()
        generate_fig_overfitting_comparison()
        generate_fig_tree_instability()
        generate_fig_ensemble_comparison()
        
        print("\n" + "="*60)
        print("✓ ALL FIGURES GENERATED SUCCESSFULLY!")
        print("="*60)
        print("\nFigures saved in: ./figures/")
        print("\nGenerated files:")
        print("  1. fig_xor_linear_fail.png")
        print("  2. fig_nested_boundaries.png")
        print("  3. fig_simple_tree_example.png")
        print("  4. fig_gini_entropy_comparison.png")
        print("  5. fig_tree_decision_boundary.png")
        print("  6. fig_overfitting_comparison.png")
        print("  7. fig_tree_instability.png")
        print("  8. fig_ensemble_comparison.png")
        print("\n" + "="*60 + "\n")
        
    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}")
        print("Please ensure all required packages are installed:")
        print("  pip install numpy matplotlib scikit-learn xgboost")


if __name__ == "__main__":
    main()
