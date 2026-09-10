"""
figure_gen.py
Generate all figures for Lesson 3 Lecture Notes: Linear Regression Plus
MPS311/439: Machine Learning
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, FancyArrowPatch
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import cross_val_score
import os

# Set random seed for reproducibility
np.random.seed(42)

# Create figures directory
os.makedirs('./figures', exist_ok=True)

# Define color palette
COLORS = {
    'primary_blue': '#2E86AB',
    'secondary_green': '#06A77D',
    'accent_orange': '#F77F00',
    'warning_red': '#E63946',
    'neutral_gray': '#6C757D',
    'light_blue': '#A7C7E7',
    'light_green': '#90EE90',
    'light_red': '#FFB6C1',
    'light_orange': '#FFD8A8'
}

# Set default style
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16


def generate_fig_01_netflix_improvement(save_path='./figures/fig_01_netflix_improvement.png'):
    """
    Generate figure 1: Netflix Prize improvement timeline
    """
    fig, ax = plt.subplots(figsize=(10, 4), dpi=300)
    
    # Timeline data
    years = ['2006\nBaseline', '2007\nEarly\nAttempts', '2008\nEnsemble\nMethods', '2009\nWinning\nSolution']
    improvements = [0, 3.5, 7.8, 10.06]
    colors_timeline = ['#E63946', '#F77F00', '#2E86AB', '#06A77D']
    
    # Create bars
    bars = ax.bar(years, improvements, color=colors_timeline, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, imp in zip(bars, improvements):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                f'{imp:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add annotations
    annotations = [
        (0, 1, 'Netflix Baseline\nAlgorithm'),
        (1, 4, 'Feature Engineering\nIntroduced'),
        (2, 8.2, 'Advanced\nEnsembles'),
        (3, 10.5, '$1M Prize Won!\n10.06% Improvement')
    ]
    
    for x, y, text in annotations:
        ax.annotate(text, xy=(x, y), fontsize=9, ha='center',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray', alpha=0.8))
    
    ax.set_ylabel('RMSE Improvement over Baseline (%)', fontsize=12, fontweight='bold')
    ax.set_title('The Netflix Prize Journey (2006-2009)', fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0, 12)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axhline(y=10, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Winning Threshold (10%)')
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()
    
    return fig, ax


def generate_fig_02_curved_relationship(save_path='./figures/fig_02_curved_relationship.png'):
    """
    Generate figure 2: House price vs age showing U-shaped relationship
    """
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    
    # Generate synthetic data: U-shaped relationship
    np.random.seed(42)
    age = np.random.uniform(0, 100, 200)
    # True relationship: U-shaped (quadratic)
    price_true = 400 - 5*age + 0.05*age**2
    # Add noise
    noise = np.random.normal(0, 20, 200)
    price = price_true + noise
    
    # Scatter plot
    ax.scatter(age, price, alpha=0.6, s=50, color=COLORS['primary_blue'], 
              edgecolors='white', linewidth=0.5, label='Houses')
    
    # True underlying relationship (dashed curve)
    age_smooth = np.linspace(0, 100, 300)
    price_smooth = 400 - 5*age_smooth + 0.05*age_smooth**2
    ax.plot(age_smooth, price_smooth, '--', color=COLORS['neutral_gray'], 
           linewidth=2, label='True Relationship', alpha=0.7)
    
    # Poor linear fit
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(age.reshape(-1, 1), price)
    price_linear = lr.predict(age_smooth.reshape(-1, 1))
    ax.plot(age_smooth, price_linear, color=COLORS['warning_red'], 
           linewidth=3, label='Linear Fit (Poor)', alpha=0.8)
    
    # Calculate R² for linear fit
    from sklearn.metrics import r2_score
    r2 = r2_score(price, lr.predict(age.reshape(-1, 1)))
    
    # Add R² annotation
    ax.text(0.05, 0.95, f'Linear Fit R² = {r2:.3f}\n(Poor!)', 
           transform=ax.transAxes, fontsize=11, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor=COLORS['light_red'], alpha=0.8))
    
    ax.set_xlabel('House Age (years)', fontsize=12, fontweight='bold')
    ax.set_ylabel('House Price (£1000s)', fontsize=12, fontweight='bold')
    ax.set_title('House Prices Don\'t Follow Straight Lines', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-5, 105)
    ax.set_ylim(100, 500)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()
    
    return fig, ax


def generate_fig_03_polynomial_progression(save_path='./figures/fig_03_polynomial_progression.png'):
    """
    Generate figure 3: Polynomial fits of increasing degree (2x3 grid)
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), dpi=300)
    axes = axes.flatten()
    
    # Generate same data as fig_02
    np.random.seed(42)
    age = np.random.uniform(0, 100, 200)
    price_true = 400 - 5*age + 0.05*age**2
    noise = np.random.normal(0, 20, 200)
    price = price_true + noise
    
    # Split into train/test
    n_train = 150
    indices = np.random.permutation(200)
    train_idx, test_idx = indices[:n_train], indices[n_train:]
    age_train, age_test = age[train_idx], age[test_idx]
    price_train, price_test = price[train_idx], price[test_idx]
    
    degrees = [1, 2, 3, 5, 10, 15]
    colors_deg = [COLORS['warning_red'], COLORS['secondary_green'], COLORS['primary_blue'], 
                  COLORS['accent_orange'], '#8B0000', '#8B0000']
    
    for idx, (degree, color) in enumerate(zip(degrees, colors_deg)):
        ax = axes[idx]
        
        # Fit polynomial
        poly = PolynomialFeatures(degree=degree)
        age_train_poly = poly.fit_transform(age_train.reshape(-1, 1))
        age_test_poly = poly.transform(age_test.reshape(-1, 1))
        
        lr = LinearRegression()
        lr.fit(age_train_poly, price_train)
        
        # Calculate R²
        train_r2 = lr.score(age_train_poly, price_train)
        test_r2 = lr.score(age_test_poly, price_test)
        
        # Plot
        ax.scatter(age_train, price_train, alpha=0.5, s=30, 
                  color=COLORS['primary_blue'], label='Train')
        ax.scatter(age_test, price_test, alpha=0.7, s=50, 
                  color=COLORS['accent_orange'], marker='^', label='Test')
        
        # Prediction curve
        age_smooth = np.linspace(0, 100, 300)
        age_smooth_poly = poly.transform(age_smooth.reshape(-1, 1))
        price_pred = lr.predict(age_smooth_poly)
        ax.plot(age_smooth, price_pred, color=color, linewidth=3, 
               label=f'Degree {degree}', alpha=0.8)
        
        # Title with R² scores
        title_color = COLORS['secondary_green'] if idx == 1 else ('black' if idx <= 2 else COLORS['warning_red'])
        ax.set_title(f'Degree {degree}: Train R²={train_r2:.3f}, Test R²={test_r2:.3f}',
                    fontsize=12, fontweight='bold', color=title_color)
        
        # Annotations
        if idx == 0:
            ax.text(0.5, 0.1, 'UNDERFITS', transform=ax.transAxes, 
                   fontsize=11, ha='center', color=COLORS['warning_red'], fontweight='bold')
        elif idx == 1:
            ax.text(0.5, 0.1, '✓ JUST RIGHT', transform=ax.transAxes, 
                   fontsize=11, ha='center', color=COLORS['secondary_green'], fontweight='bold')
            # Add a star
            ax.plot(50, 275, marker='*', markersize=20, color='gold')
        elif idx >= 4:
            ax.text(0.5, 0.9, 'OVERFITS!', transform=ax.transAxes, 
                   fontsize=11, ha='center', color=COLORS['warning_red'], fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        
        ax.set_xlabel('House Age (years)', fontsize=10)
        ax.set_ylabel('House Price (£1000s)', fontsize=10)
        ax.set_xlim(-5, 105)
        ax.set_ylim(100, 500)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(loc='upper right', fontsize=9)
    
    plt.suptitle('Polynomial Progression: From Underfitting to Overfitting', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()
    
    return fig, axes


def generate_fig_04_interaction_effect(save_path='./figures/fig_04_interaction_effect.png'):
    """
    Generate figure 4: 3D surface plot showing interaction effects
    """
    fig = plt.figure(figsize=(8, 8), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    
    # Create grid
    size = np.linspace(1000, 3000, 50)
    bedrooms = np.linspace(1, 5, 50)
    SIZE, BEDROOMS = np.meshgrid(size, bedrooms)
    
    # Additive model (no interaction)
    price_additive = 100 + 0.08*SIZE + 20*BEDROOMS
    
    # Model with interaction
    price_interaction = 100 + 0.08*SIZE + 20*BEDROOMS + 0.005*SIZE*BEDROOMS
    
    # Plot surfaces
    surf1 = ax.plot_surface(SIZE, BEDROOMS, price_additive, alpha=0.4, 
                           color=COLORS['primary_blue'], label='Additive')
    surf2 = ax.plot_surface(SIZE, BEDROOMS, price_interaction, alpha=0.8, 
                           color=COLORS['accent_orange'], label='With Interaction')
    
    # Add specific points
    unusual_points = [
        (2500, 1, 'Large 1-bedroom\n(unusual)'),
        (1200, 4, 'Small 4-bedroom\n(cramped)')
    ]
    
    for s, b, label in unusual_points:
        p_add = 100 + 0.08*s + 20*b
        p_int = 100 + 0.08*s + 20*b + 0.005*s*b
        ax.scatter([s], [b], [p_add], color='blue', s=100, marker='o', edgecolors='black', linewidth=2)
        ax.scatter([s], [b], [p_int], color='red', s=100, marker='o', edgecolors='black', linewidth=2)
        ax.text(s, b, p_int + 20, label, fontsize=9, ha='center')
    
    ax.set_xlabel('House Size (sq ft)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Bedrooms', fontsize=11, fontweight='bold')
    ax.set_zlabel('Price (£1000s)', fontsize=11, fontweight='bold')
    ax.set_title('Interaction Effects Capture Non-Additive Relationships', 
                fontsize=13, fontweight='bold', pad=20)
    
    # Set viewing angle
    ax.view_init(elev=30, azim=45)
    
    # Create custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['primary_blue'], alpha=0.4, label='Additive Model'),
        Patch(facecolor=COLORS['accent_orange'], alpha=0.8, label='With Interaction')
    ]
    ax.legend(handles=legend_elements, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()
    
    return fig, ax


def generate_fig_05_overfitting_visualization(save_path='./figures/fig_05_overfitting_visualization.png'):
    """
    Generate figure 5: Dramatic visualization of degree 10 overfitting
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    # Generate data
    np.random.seed(42)
    age = np.random.uniform(0, 100, 50)
    price_true = 400 - 5*age + 0.05*age**2
    noise = np.random.normal(0, 20, 50)
    price_train = price_true + noise
    
    # Test data
    age_test = np.random.uniform(0, 100, 20)
    price_test_true = 400 - 5*age_test + 0.05*age_test**2
    price_test = price_test_true + np.random.normal(0, 20, 20)
    
    # Fit degree 10 polynomial
    poly = PolynomialFeatures(degree=10)
    age_train_poly = poly.fit_transform(age.reshape(-1, 1))
    age_test_poly = poly.transform(age_test.reshape(-1, 1))
    
    lr = LinearRegression()
    lr.fit(age_train_poly, price_train)
    
    train_r2 = lr.score(age_train_poly, price_train)
    test_r2 = lr.score(age_test_poly, price_test)
    
    # Plot training points
    ax.scatter(age, price_train, s=100, color=COLORS['primary_blue'], 
              edgecolors='black', linewidth=1.5, label='Training Points', zorder=3)
    
    # Plot test points
    ax.scatter(age_test, price_test, s=100, color=COLORS['accent_orange'], 
              marker='^', edgecolors='black', linewidth=1.5, label='Test Points', zorder=3)
    
    # Prediction curve (very dense for oscillations)
    age_smooth = np.linspace(0, 100, 1000)
    age_smooth_poly = poly.transform(age_smooth.reshape(-1, 1))
    price_pred = lr.predict(age_smooth_poly)
    ax.plot(age_smooth, price_pred, color=COLORS['warning_red'], 
           linewidth=2.5, label='Degree 10 Polynomial', alpha=0.9, zorder=2)
    
    # True function (gray)
    price_true_smooth = 400 - 5*age_smooth + 0.05*age_smooth**2
    ax.plot(age_smooth, price_true_smooth, '--', color=COLORS['neutral_gray'], 
           linewidth=2, label='True Function', alpha=0.6, zorder=1)
    
    # Annotations
    ax.annotate('Passes through\ntraining points', xy=(75, 280), xytext=(85, 350),
               arrowprops=dict(arrowstyle='->', color='black', lw=2),
               fontsize=10, ha='center',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    ax.annotate('Unrealistic\noscillations', xy=(40, 450), xytext=(20, 520),
               arrowprops=dict(arrowstyle='->', color='red', lw=2),
               fontsize=10, ha='center',
               bbox=dict(boxstyle='round', facecolor=COLORS['light_red'], alpha=0.7))
    
    # Text box with R² scores
    textstr = f'Train R² = {train_r2:.3f} (Excellent!)\nTest R² = {test_r2:.3f} (Terrible!)'
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
           verticalalignment='top', fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='white', edgecolor='red', linewidth=2, alpha=0.9))
    
    ax.set_xlabel('House Age (years)', fontsize=12, fontweight='bold')
    ax.set_ylabel('House Price (£1000s)', fontsize=12, fontweight='bold')
    ax.set_title('Overfitting: Memorizing Training Data', fontsize=14, fontweight='bold', color='red')
    ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-5, 105)
    ax.set_ylim(100, 600)
    
    # Inset showing detail
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    axins = inset_axes(ax, width="30%", height="30%", loc='upper right', 
                      bbox_to_anchor=(0, 0, 1, 1), bbox_transform=ax.transAxes)
    
    # Zoom into region with oscillations
    mask = (age_smooth > 30) & (age_smooth < 50)
    axins.plot(age_smooth[mask], price_pred[mask], color=COLORS['warning_red'], linewidth=2)
    axins.plot(age_smooth[mask], price_true_smooth[mask], '--', 
              color=COLORS['neutral_gray'], linewidth=1.5)
    axins.set_title('Zoom: Oscillations', fontsize=9)
    axins.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()
    
    return fig, ax


def generate_fig_06_bias_variance_tradeoff(save_path='./figures/fig_06_bias_variance_tradeoff.png'):
    """
    Generate figure 6: Bias-variance tradeoff conceptual diagram
    """
    fig = plt.figure(figsize=(10, 5), dpi=300)
    gs = GridSpec(2, 3, height_ratios=[3, 1], hspace=0.4)
    
    # Generate data
    np.random.seed(42)
    x = np.linspace(0, 10, 50)
    y_true = 2 + 0.5*x - 0.05*x**2
    y = y_true + np.random.normal(0, 0.5, 50)
    
    x_smooth = np.linspace(0, 10, 200)
    y_true_smooth = 2 + 0.5*x_smooth - 0.05*x_smooth**2
    
    scenarios = [
        ('High Bias\n(Underfitting)', 1, COLORS['warning_red']),
        ('Balanced\n(Just Right)', 2, COLORS['secondary_green']),
        ('High Variance\n(Overfitting)', 8, COLORS['primary_blue'])
    ]
    
    for idx, (title, degree, color) in enumerate(scenarios):
        ax = fig.add_subplot(gs[0, idx])
        
        # Fit polynomial
        poly = PolynomialFeatures(degree=degree)
        x_poly = poly.fit_transform(x.reshape(-1, 1))
        lr = LinearRegression()
        lr.fit(x_poly, y)
        
        # Predict
        x_smooth_poly = poly.transform(x_smooth.reshape(-1, 1))
        y_pred = lr.predict(x_smooth_poly)
        
        # Plot
        ax.scatter(x, y, alpha=0.6, s=40, color='black', label='Data')
        ax.plot(x_smooth, y_true_smooth, '--', color='gray', linewidth=2, 
               label='True Function', alpha=0.6)
        ax.plot(x_smooth, y_pred, color=color, linewidth=3, label=f'Degree {degree}')
        
        ax.set_title(title, fontsize=12, fontweight='bold', color=color)
        ax.set_ylim(-2, 5)
        ax.grid(True, alpha=0.3)
        
        if idx == 0:
            ax.text(0.5, 0.15, 'Missing\nthe pattern', transform=ax.transAxes,
                   ha='center', fontsize=9, color='red', fontweight='bold')
            ax.set_ylabel('y', fontsize=11)
        elif idx == 1:
            ax.text(0.5, 0.15, 'Captures signal,\nignores noise', transform=ax.transAxes,
                   ha='center', fontsize=9, color='green', fontweight='bold')
            ax.legend(loc='upper left', fontsize=8)
        else:
            ax.text(0.5, 0.85, 'Chasing\nnoise', transform=ax.transAxes,
                   ha='center', fontsize=9, color='blue', fontweight='bold')
        
        ax.set_xlabel('x', fontsize=11)
    
    # Bottom: Error decomposition bar chart
    ax_bar = fig.add_subplot(gs[1, :])
    
    categories = ['High Bias', 'Balanced', 'High Variance']
    bias_vals = [5, 1, 0.5]
    variance_vals = [0.5, 1, 5]
    irreducible = [1, 1, 1]
    
    x_pos = np.arange(len(categories))
    
    p1 = ax_bar.bar(x_pos, bias_vals, color=COLORS['warning_red'], 
                   alpha=0.7, label='Bias²')
    p2 = ax_bar.bar(x_pos, variance_vals, bottom=bias_vals, 
                   color=COLORS['primary_blue'], alpha=0.7, label='Variance')
    p3 = ax_bar.bar(x_pos, irreducible, 
                   bottom=[b+v for b,v in zip(bias_vals, variance_vals)],
                   color=COLORS['neutral_gray'], alpha=0.7, label='Irreducible')
    
    ax_bar.set_ylabel('Error Contribution', fontsize=11, fontweight='bold')
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(categories)
    ax_bar.set_title('Error Decomposition: Total Error = Bias² + Variance + Irreducible', 
                    fontsize=11, fontweight='bold')
    ax_bar.legend(loc='upper right', ncol=3)
    ax_bar.grid(axis='y', alpha=0.3)
    
    plt.suptitle('The Bias-Variance Tradeoff', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()
    
    return fig


def generate_fig_07_train_test_error_vs_complexity(save_path='./figures/fig_07_train_test_error_vs_complexity.png'):
    """
    Generate figure 7: Classic U-curve showing train and test error vs complexity
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    # Simulate errors vs complexity
    complexity = np.arange(1, 16)
    
    # Training error: monotonically decreasing
    train_error = 50 * np.exp(-0.3 * complexity) + 2
    
    # Test error: U-shaped
    test_error = 50 * np.exp(-0.3 * complexity) + 0.5 * (complexity - 3)**2 + 5
    
    # Plot curves
    ax.plot(complexity, train_error, 'o-', color=COLORS['primary_blue'], 
           linewidth=3, markersize=8, label='Training Error', alpha=0.8)
    ax.plot(complexity, test_error, 's-', color=COLORS['accent_orange'], 
           linewidth=3, markersize=8, label='Test Error', alpha=0.8)
    
    # Mark optimal complexity
    optimal_idx = np.argmin(test_error)
    optimal_complexity = complexity[optimal_idx]
    ax.axvline(x=optimal_complexity, color=COLORS['secondary_green'], 
              linestyle='--', linewidth=2.5, label='Optimal Complexity', alpha=0.8)
    
    # Mark the optimal point
    ax.plot(optimal_complexity, test_error[optimal_idx], '*', 
           markersize=20, color='gold', markeredgecolor='black', 
           markeredgewidth=2, zorder=10, label='Minimum Test Error')
    
    # Shade regions
    ax.axvspan(1, optimal_complexity, alpha=0.15, color=COLORS['warning_red'], 
              label='Underfitting Zone')
    ax.axvspan(optimal_complexity, 15, alpha=0.15, color=COLORS['warning_red'])
    ax.axvspan(optimal_complexity-0.5, optimal_complexity+0.5, alpha=0.2, 
              color=COLORS['secondary_green'])
    
    # Annotations
    ax.annotate('Training error\nalways decreases', 
               xy=(12, train_error[-3]), xytext=(13, 15),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'),
               fontsize=10, ha='center',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.annotate('Test error increases\nafter optimal point\n(OVERFITTING)', 
               xy=(12, test_error[-3]), xytext=(11, 35),
               arrowprops=dict(arrowstyle='->', lw=2, color='red'),
               fontsize=10, ha='center', color='red',
               bbox=dict(boxstyle='round', facecolor=COLORS['light_red'], alpha=0.8))
    
    ax.text(1.5, 45, 'Underfitting\n(High Bias)', fontsize=11, 
           ha='center', fontweight='bold', color=COLORS['warning_red'])
    ax.text(optimal_complexity, 48, 'Sweet\nSpot', fontsize=11, 
           ha='center', fontweight='bold', color=COLORS['secondary_green'])
    ax.text(13.5, 45, 'Overfitting\n(High Variance)', fontsize=11, 
           ha='center', fontweight='bold', color=COLORS['warning_red'])
    
    ax.set_xlabel('Model Complexity (e.g., Polynomial Degree)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Squared Error', fontsize=12, fontweight='bold')
    ax.set_title('The U-Curve: Training vs. Test Error', fontsize=14, fontweight='bold')
    ax.set_xlim(0.5, 15.5)
    ax.set_ylim(0, 50)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()
    
    return fig, ax


def generate_fig_08_overfitting_indicators(save_path='./figures/fig_08_overfitting_indicators.png'):
    """
    Generate figure 8: Multiple panels showing overfitting warning signs
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=300)
    
    # Panel 1: Train/Test R² comparison
    ax = axes[0, 0]
    models = ['Good\nModel', 'Overfit\nModel']
    train_r2 = [0.82, 0.98]
    test_r2 = [0.78, 0.45]
    
    x_pos = np.arange(len(models))
    width = 0.35
    
    bars1 = ax.bar(x_pos - width/2, train_r2, width, label='Train R²', 
                  color=COLORS['primary_blue'], alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x_pos + width/2, test_r2, width, label='Test R²', 
                  color=COLORS['accent_orange'], alpha=0.8, edgecolor='black')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{height:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Add gap indicators
    for i, (t, te) in enumerate(zip(train_r2, test_r2)):
        gap = t - te
        ax.plot([i, i], [te, t], 'k-', linewidth=2, alpha=0.5)
        color = 'green' if gap < 0.1 else 'red'
        ax.text(i + 0.3, (t + te)/2, f'Gap:\n{gap:.2f}', 
               fontsize=9, ha='left', color=color, fontweight='bold')
    
    ax.set_ylabel('R² Score', fontsize=11, fontweight='bold')
    ax.set_title('Warning Sign 1: Large Train/Test Gap', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    
    # Panel 2: Coefficient magnitudes
    ax = axes[0, 1]
    features = [f'w{i}' for i in range(1, 9)]
    good_coefs = [12.5, -10.3, 8.7, -6.2, 5.1, -4.3, 3.8, -2.9]
    overfit_coefs = [1253, -1187, 1045, -998, 923, -845, 789, -721]
    
    x_pos = np.arange(len(features))
    width = 0.35
    
    ax.bar(x_pos - width/2, good_coefs, width, label='Good Model', 
          color=COLORS['secondary_green'], alpha=0.8, edgecolor='black')
    ax.bar(x_pos + width/2, overfit_coefs, width, label='Overfit Model', 
          color=COLORS['warning_red'], alpha=0.8, edgecolor='black')
    
    ax.set_ylabel('Coefficient Value', fontsize=11, fontweight='bold')
    ax.set_title('Warning Sign 2: Very Large Coefficients', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(features)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='black', linewidth=0.5)
    
    # Panel 3: Sensitivity to data changes
    ax = axes[1, 0]
    x = np.linspace(0, 10, 100)
    
    # Good model: consistent predictions
    for i in range(5):
        np.random.seed(i)
        offset = np.random.normal(0, 0.5)
        y = 2 + 0.5*x + offset
        ax.plot(x, y, color=COLORS['secondary_green'], alpha=0.6, linewidth=2)
    
    # Overfit model: wildly different predictions
    for i in range(5):
        np.random.seed(i+10)
        y = 2 + 0.5*x + 2*np.sin(x + i) + np.random.normal(0, 1, 100)
        ax.plot(x, y, color=COLORS['warning_red'], alpha=0.6, linewidth=2, linestyle='--')
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=COLORS['secondary_green'], linewidth=2, label='Good Model (stable)'),
        Line2D([0], [0], color=COLORS['warning_red'], linewidth=2, linestyle='--', label='Overfit Model (unstable)')
    ]
    ax.legend(handles=legend_elements, loc='upper left')
    
    ax.set_xlabel('Input', fontsize=11, fontweight='bold')
    ax.set_ylabel('Prediction', fontsize=11, fontweight='bold')
    ax.set_title('Warning Sign 3: High Sensitivity', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Panel 4: Prediction stability
    ax = axes[1, 1]
    x = np.linspace(0, 10, 200)
    
    # Good model: smooth
    y_good = 2 + 0.5*x - 0.05*x**2
    ax.plot(x, y_good, color=COLORS['secondary_green'], linewidth=3, 
           label='Good Model (smooth)', alpha=0.8)
    
    # Overfit model: erratic
    y_overfit = 2 + 0.5*x - 0.05*x**2 + 2*np.sin(5*x) + 1.5*np.cos(7*x)
    ax.plot(x, y_overfit, color=COLORS['warning_red'], linewidth=3, 
           label='Overfit Model (erratic)', alpha=0.8)
    
    ax.set_xlabel('Input', fontsize=11, fontweight='bold')
    ax.set_ylabel('Prediction', fontsize=11, fontweight='bold')
    ax.set_title('Warning Sign 4: Erratic Predictions', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Recognizing Overfitting: Warning Signs', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()
    
    return fig, axes


def generate_fig_09_ridge_coefficient_paths(save_path='./figures/fig_09_ridge_coefficient_paths.png'):
    """
    Generate figure 9: Ridge coefficient paths showing smooth shrinkage
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    # Generate synthetic data
    np.random.seed(42)
    X = np.random.randn(100, 10)
    y = X[:, 0] * 3 + X[:, 1] * (-2) + X[:, 2] * 1.5 + np.random.randn(100) * 0.5
    
    # Range of lambda values (log scale)
    lambdas = np.logspace(-3, 4, 100)
    coefs = []
    
    for lam in lambdas:
        ridge = Ridge(alpha=lam)
        ridge.fit(X, y)
        coefs.append(ridge.coef_)
    
    coefs = np.array(coefs)
    
    # Plot coefficient paths
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for i in range(10):
        if i == 0:
            # Highlight most important coefficient
            ax.plot(np.log10(lambdas), coefs[:, i], linewidth=3.5, 
                   color=colors[i], label=f'Coefficient {i+1}', alpha=0.9)
        else:
            ax.plot(np.log10(lambdas), coefs[:, i], linewidth=2, 
                   color=colors[i], label=f'Coefficient {i+1}', alpha=0.7)
    
    # Mark optimal lambda (example)
    optimal_lambda = 10
    ax.axvline(x=np.log10(optimal_lambda), color=COLORS['secondary_green'], 
              linestyle='--', linewidth=2.5, label='Optimal λ (from CV)', alpha=0.8)
    
    ax.set_xlabel('log₁₀(λ) [Regularization Strength]', fontsize=12, fontweight='bold')
    ax.set_ylabel('Coefficient Value', fontsize=12, fontweight='bold')
    ax.set_title('Ridge: Smooth Shrinkage (No Sparsity)', fontsize=14, fontweight='bold')
    ax.axhline(y=0, color='black', linewidth=0.5, alpha=0.3)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', ncol=2, fontsize=9)
    
    # Add annotations
    ax.text(-2, 2.5, 'λ → 0\n(No regularization)\nLarge coefficients', 
           fontsize=10, ha='center', 
           bbox=dict(boxstyle='round', facecolor=COLORS['light_red'], alpha=0.7))
    
    ax.text(3, 0.3, 'λ → ∞\n(Strong regularization)\nAll coefficients → 0', 
           fontsize=10, ha='center',
           bbox=dict(boxstyle='round', facecolor=COLORS['light_blue'], alpha=0.7))
    
    ax.text(1, -2, 'Note: No coefficient\nreaches exactly zero', 
           fontsize=10, ha='center', color='red', fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.6))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()
    
    return fig, ax


def generate_fig_10_ridge_geometric(save_path='./figures/fig_10_ridge_geometric.png'):
    """
    Generate figure 10: Ridge geometric interpretation with circle constraint
    """
    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    
    # Create MSE contours (ellipses)
    w1 = np.linspace(-3, 3, 400)
    w2 = np.linspace(-3, 3, 400)
    W1, W2 = np.meshgrid(w1, w2)
    
    # MSE surface (elliptical contours centered at (1.5, 1))
    MSE = (W1 - 1.5)**2 / 0.5 + (W2 - 1)**2 / 1.5
    
    # Plot contours
    levels = [0.5, 1, 2, 4, 8, 12]
    contours = ax.contour(W1, W2, MSE, levels=levels, colors=COLORS['primary_blue'], 
                         linewidths=2, alpha=0.6)
    ax.clabel(contours, inline=True, fontsize=9, fmt='MSE=%.1f')
    
    # Ridge constraint: circle
    t = 1.2  # constraint parameter
    theta = np.linspace(0, 2*np.pi, 100)
    circle_w1 = t * np.cos(theta)
    circle_w2 = t * np.sin(theta)
    ax.plot(circle_w1, circle_w2, color=COLORS['warning_red'], 
           linewidth=3, label='Ridge Constraint\n(w₁² + w₂² ≤ t)', alpha=0.8)
    ax.fill(circle_w1, circle_w2, color=COLORS['warning_red'], alpha=0.1)
    
    # Unconstrained minimum (center of ellipses)
    ax.plot(1.5, 1, 'o', markersize=12, color=COLORS['primary_blue'], 
           markeredgecolor='black', markeredgewidth=2, label='Unconstrained\nMinimum', zorder=5)
    
    # Ridge solution (tangent point)
    # Approximate tangent point
    ridge_w1 = 0.9
    ridge_w2 = 0.8
    ax.plot(ridge_w1, ridge_w2, '*', markersize=20, color=COLORS['secondary_green'], 
           markeredgecolor='black', markeredgewidth=2, label='Ridge Solution', zorder=5)
    
    # Annotations
    ax.annotate('', xy=(ridge_w1, ridge_w2), xytext=(1.5, 1),
               arrowprops=dict(arrowstyle='->', lw=2, color='black', ls='--'))
    
    ax.text(1.5, 1.3, 'Moves from\nunconstrained\nto constrained', 
           fontsize=10, ha='center',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.text(0, -2.5, 'Constraint: w₁² + w₂² ≤ t\n(Circle)', 
           fontsize=11, ha='center', color=COLORS['warning_red'], fontweight='bold',
           bbox=dict(boxstyle='round', facecolor=COLORS['light_red'], alpha=0.7))
    
    ax.text(-2.5, 2.5, 'Rarely touches\nat axis\n(non-sparse)', 
           fontsize=10, ha='center', fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # Axes
    ax.axhline(y=0, color='black', linewidth=0.5, alpha=0.3)
    ax.axvline(x=0, color='black', linewidth=0.5, alpha=0.3)
    ax.set_xlabel('w₁', fontsize=13, fontweight='bold')
    ax.set_ylabel('w₂', fontsize=13, fontweight='bold')
    ax.set_title('Ridge Regression: Geometric Interpretation', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()
    
    return fig, ax


def generate_fig_11_lasso_coefficient_paths(save_path='./figures/fig_11_lasso_coefficient_paths.png'):
    """
    Generate figure 11: Lasso coefficient paths showing sparsity
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    # Generate synthetic data with many features
    np.random.seed(42)
    X = np.random.randn(100, 30)
    # Only first 5 features are truly relevant
    y = (X[:, 0] * 3 + X[:, 1] * (-2) + X[:, 2] * 1.5 +
         X[:, 3] * 1 + X[:, 4] * (-0.8) + np.random.randn(100) * 0.5)
    
    # Range of lambda values
    lambdas = np.logspace(-3, 2, 100)
    coefs = []
    n_nonzero = []

    from sklearn.linear_model import Lasso  # Ensure import inside function in case called standalone

    for lam in lambdas:
        lasso = Lasso(alpha=lam, max_iter=10000)
        lasso.fit(X, y)
        coefs.append(lasso.coef_.copy())
        n_nonzero.append(np.sum(np.abs(lasso.coef_) > 1e-5))

    coefs = np.array(coefs)

    # Plot coefficient paths (showing wedge pattern)
    colors = plt.cm.tab20(np.linspace(0, 1, 30))

    for i in range(30):
        # Vary line thickness and alpha based on importance
        if i < 5:  # Important features
            alpha = 0.9
            linewidth = 2.5
        else:  # Less important features
            alpha = 0.4
            linewidth = 1.5

        ax.plot(np.log10(lambdas), coefs[:, i], linewidth=linewidth,
                color=colors[i % len(colors)], alpha=alpha)  # modulo to avoid index error

    # Mark optimal lambda
    optimal_lambda = 0.1
    ax.axvline(x=np.log10(optimal_lambda), color=COLORS['secondary_green'],
               linestyle='--', linewidth=2.5, label='Optimal λ (from CV)', alpha=0.8)

    # Annotations for key events
    # First coefficient drops to zero
    n_nonzero_arr = np.array(n_nonzero)
    nz_lt_30 = np.where(n_nonzero_arr < 30)[0]
    if len(nz_lt_30) > 0:
        first_zero_idx = nz_lt_30[0]
    else:
        first_zero_idx = 10 if len(lambdas) > 10 else 0
    ax.annotate('First coefficient\ndrops to zero',
                xy=(np.log10(lambdas[first_zero_idx]), 0),
                xytext=(np.log10(lambdas[first_zero_idx])-0.5, -1.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='red'),
                fontsize=9, ha='center',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    # Half coefficients zeroed
    nz_le_15 = np.where(n_nonzero_arr <= 15)[0]
    if len(nz_le_15) > 0:
        half_zero_idx = nz_le_15[0]
    else:
        half_zero_idx = 40 if len(lambdas) > 40 else int(len(lambdas)/2)
    ax.text(np.log10(lambdas[half_zero_idx]), 2,
            f'Half of coefficients\nzeroed\n({n_nonzero[half_zero_idx]} remain)',
            fontsize=9, ha='center',
            bbox=dict(boxstyle='round', facecolor=COLORS['light_orange'], alpha=0.7))

    # Only few survive
    few_idx = -10 if len(lambdas) >= 10 else -1
    ax.text(np.log10(lambdas[few_idx]), -2,
            f'Only {n_nonzero[few_idx]} coefficients\nremain non-zero',
            fontsize=9, ha='center',
            bbox=dict(boxstyle='round', facecolor=COLORS['light_red'], alpha=0.7))

    ax.set_xlabel('log₁₀(λ) [Regularization Strength]', fontsize=12, fontweight='bold')
    ax.set_ylabel('Coefficient Value', fontsize=12, fontweight='bold')
    ax.set_title('Lasso: Sparse Solutions (Automatic Feature Selection)', fontsize=14, fontweight='bold')
    ax.axhline(y=0, color='black', linewidth=0.5, alpha=0.3)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)

    # Add text box
    ax.text(-2.5, 3, 'λ → 0\nAll features', fontsize=10, ha='center',
            bbox=dict(boxstyle='round', facecolor=COLORS['light_blue'], alpha=0.7))
    ax.text(1.5, 3, 'λ → ∞\nSparse\n(few features)', fontsize=10, ha='center',
            bbox=dict(boxstyle='round', facecolor=COLORS['light_green'], alpha=0.7))

    # Highlight the wedge pattern
    ax.text(0, -3, 'Note: Coefficients drop to\nEXACTLY zero (sparsity)!',
            fontsize=11, ha='center', color='red', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

    return fig, ax


def generate_fig_12_lasso_geometric(save_path='./figures/fig_12_lasso_geometric.png'):
    """
    Generate figure 12: Lasso geometric interpretation with diamond constraint
    """
    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    
    # Create MSE contours (same as Ridge)
    w1 = np.linspace(-3, 3, 400)
    w2 = np.linspace(-3, 3, 400)
    W1, W2 = np.meshgrid(w1, w2)
    
    # MSE surface
    MSE = (W1 - 1.5)**2 / 0.5 + (W2 - 1)**2 / 1.5
    
    # Plot contours
    levels = [0.5, 1, 2, 4, 8, 12]
    contours = ax.contour(W1, W2, MSE, levels=levels, colors=COLORS['primary_blue'], 
                         linewidths=2, alpha=0.6)
    ax.clabel(contours, inline=True, fontsize=9, fmt='MSE=%.1f')
    
    # Lasso constraint: diamond
    t = 1.8
    diamond_w1 = np.array([t, 0, -t, 0, t])
    diamond_w2 = np.array([0, t, 0, -t, 0])
    ax.plot(diamond_w1, diamond_w2, color=COLORS['warning_red'], 
           linewidth=3, label='Lasso Constraint\n(|w₁| + |w₂| ≤ t)', alpha=0.8)
    ax.fill(diamond_w1, diamond_w2, color=COLORS['warning_red'], alpha=0.1)
    
    # Unconstrained minimum
    ax.plot(1.5, 1, 'o', markersize=12, color=COLORS['primary_blue'], 
           markeredgecolor='black', markeredgewidth=2, label='Unconstrained\nMinimum', zorder=5)
    
    # Lasso solution (at corner - sparse!)
    lasso_w1 = 1.8
    lasso_w2 = 0
    ax.plot(lasso_w1, lasso_w2, '*', markersize=20, color=COLORS['secondary_green'], 
           markeredgecolor='black', markeredgewidth=2, label='Lasso Solution\n(Sparse: w₂=0)', zorder=5)
    
    # Annotations
    ax.annotate('', xy=(lasso_w1, lasso_w2), xytext=(1.5, 1),
               arrowprops=dict(arrowstyle='->', lw=2, color='black', ls='--'))
    
    ax.text(1.5, 1.4, 'Solution moves\nto corner\n(on axis)', 
           fontsize=10, ha='center',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.text(0, -2.5, 'Constraint: |w₁| + |w₂| ≤ t\n(Diamond)', 
           fontsize=11, ha='center', color=COLORS['warning_red'], fontweight='bold',
           bbox=dict(boxstyle='round', facecolor=COLORS['light_red'], alpha=0.7))
    
    # Highlight corners
    for w1_c, w2_c in [(t, 0), (0, t), (-t, 0), (0, -t)]:
        ax.plot(w1_c, w2_c, 'o', markersize=10, color='yellow', 
               markeredgecolor='red', markeredgewidth=2, zorder=4)
    
    ax.text(-2.5, 2.5, 'Diamond corners\n→ SPARSE solutions\n(on axes)', 
           fontsize=10, ha='center', fontweight='bold', color='red',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    # Axes
    ax.axhline(y=0, color='black', linewidth=0.5, alpha=0.3)
    ax.axvline(x=0, color='black', linewidth=0.5, alpha=0.3)
    ax.set_xlabel('w₁', fontsize=13, fontweight='bold')
    ax.set_ylabel('w₂', fontsize=13, fontweight='bold')
    ax.set_title('Lasso Regression: Geometric Interpretation', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()
    
    return fig, ax


def generate_fig_13_ridge_vs_lasso_comparison(save_path='./figures/fig_13_ridge_vs_lasso_comparison.png'):
    """
    Generate figure 13: Side-by-side Ridge vs Lasso with properties table
    """
    fig = plt.figure(figsize=(14, 6), dpi=300)
    gs = GridSpec(1, 3, width_ratios=[2, 2, 1.5], wspace=0.3)
    
    # Common MSE contours
    w1 = np.linspace(-3, 3, 400)
    w2 = np.linspace(-3, 3, 400)
    W1, W2 = np.meshgrid(w1, w2)
    MSE = (W1 - 1.5)**2 / 0.5 + (W2 - 1)**2 / 1.5
    levels = [0.5, 1, 2, 4, 8, 12]
    
    # Left panel: Ridge
    ax_ridge = fig.add_subplot(gs[0, 0])
    ax_ridge.contour(W1, W2, MSE, levels=levels, colors=COLORS['primary_blue'], 
                    linewidths=1.5, alpha=0.6)
    
    # Ridge circle
    t = 1.2
    theta = np.linspace(0, 2*np.pi, 100)
    ax_ridge.plot(t * np.cos(theta), t * np.sin(theta), 
                 color=COLORS['warning_red'], linewidth=3, alpha=0.8)
    ax_ridge.fill(t * np.cos(theta), t * np.sin(theta), 
                 color=COLORS['warning_red'], alpha=0.1)
    
    ax_ridge.plot(1.5, 1, 'o', markersize=10, color=COLORS['primary_blue'], 
                 markeredgecolor='black', markeredgewidth=2)
    ax_ridge.plot(0.9, 0.8, '*', markersize=15, color=COLORS['secondary_green'], 
                 markeredgecolor='black', markeredgewidth=2)
    
    ax_ridge.axhline(y=0, color='black', linewidth=0.5, alpha=0.3)
    ax_ridge.axvline(x=0, color='black', linewidth=0.5, alpha=0.3)
    ax_ridge.set_xlabel('w₁', fontsize=12, fontweight='bold')
    ax_ridge.set_ylabel('w₂', fontsize=12, fontweight='bold')
    ax_ridge.set_title('Ridge (L2)\nCircle Constraint', fontsize=13, fontweight='bold')
    ax_ridge.set_xlim(-3, 3)
    ax_ridge.set_ylim(-3, 3)
    ax_ridge.set_aspect('equal')
    ax_ridge.grid(True, alpha=0.3)
    
    # Middle panel: Lasso
    ax_lasso = fig.add_subplot(gs[0, 1])
    ax_lasso.contour(W1, W2, MSE, levels=levels, colors=COLORS['primary_blue'], 
                    linewidths=1.5, alpha=0.6)
    
    # Lasso diamond
    t = 1.8
    diamond_w1 = np.array([t, 0, -t, 0, t])
    diamond_w2 = np.array([0, t, 0, -t, 0])
    ax_lasso.plot(diamond_w1, diamond_w2, 
                 color=COLORS['warning_red'], linewidth=3, alpha=0.8)
    ax_lasso.fill(diamond_w1, diamond_w2, 
                 color=COLORS['warning_red'], alpha=0.1)
    
    ax_lasso.plot(1.5, 1, 'o', markersize=10, color=COLORS['primary_blue'], 
                 markeredgecolor='black', markeredgewidth=2)
    ax_lasso.plot(1.8, 0, '*', markersize=15, color=COLORS['secondary_green'], 
                 markeredgecolor='black', markeredgewidth=2)
    
    # Highlight corners
    for w1_c, w2_c in [(t, 0), (0, t), (-t, 0), (0, -t)]:
        ax_lasso.plot(w1_c, w2_c, 'o', markersize=8, color='yellow', 
                     markeredgecolor='red', markeredgewidth=2)
    
    ax_lasso.axhline(y=0, color='black', linewidth=0.5, alpha=0.3)
    ax_lasso.axvline(x=0, color='black', linewidth=0.5, alpha=0.3)
    ax_lasso.set_xlabel('w₁', fontsize=12, fontweight='bold')
    ax_lasso.set_ylabel('w₂', fontsize=12, fontweight='bold')
    ax_lasso.set_title('Lasso (L1)\nDiamond Constraint', fontsize=13, fontweight='bold')
    ax_lasso.set_xlim(-3, 3)
    ax_lasso.set_ylim(-3, 3)
    ax_lasso.set_aspect('equal')
    ax_lasso.grid(True, alpha=0.3)
    
    # Right panel: Properties table
    ax_table = fig.add_subplot(gs[0, 2])
    ax_table.axis('off')
    
    properties = ['Penalty', 'Geometry', 'Sparsity', 'Computation', 'Correlated\nFeatures', 'Use When']
    ridge_props = ['λΣw²', 'Circle', '✗ No', '✓ Fast', '✓ Handles\nwell', 'All features\nrelevant']
    lasso_props = ['λΣ|w|', 'Diamond', '✓ Yes', 'Slower', '⚠ Picks\none', 'Many\nirrelevant']
    
    # Create table
    table_data = []
    for prop, ridge, lasso in zip(properties, ridge_props, lasso_props):
        table_data.append([prop, ridge, lasso])
    
    table = ax_table.table(cellText=table_data, 
                          colLabels=['Property', 'Ridge', 'Lasso'],
                          cellLoc='center',
                          loc='center',
                          bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Color code cells
    for i in range(len(properties) + 1):
        for j in range(3):
            cell = table[(i, j)]
            if i == 0:  # Header
                cell.set_facecolor(COLORS['neutral_gray'])
                cell.set_text_props(weight='bold', color='white')
            else:
                if j == 0:  # Property column
                    cell.set_facecolor(COLORS['light_blue'])
                    cell.set_text_props(weight='bold')
                elif '✓' in cell.get_text().get_text():
                    cell.set_facecolor(COLORS['light_green'])
                elif '✗' in cell.get_text().get_text():
                    cell.set_facecolor(COLORS['light_red'])
                elif '⚠' in cell.get_text().get_text():
                    cell.set_facecolor(COLORS['light_orange'])
    
    plt.suptitle('Ridge vs. Lasso: Comprehensive Comparison', 
                fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()
    
    return fig


def generate_fig_14_regularization_results(save_path='./figures/fig_14_regularization_results.png'):
    """
    Generate figure 14: Bar chart comparing model performance
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    models = ['Linear\nBaseline', 'Poly deg 3\n(No Reg)', 'Ridge\n(λ=10)', 'Lasso\n(λ=1)']
    train_r2 = [0.60, 0.98, 0.82, 0.80]
    test_r2 = [0.58, 0.42, 0.77, 0.76]
    
    x_pos = np.arange(len(models))
    width = 0.35
    
    bars1 = ax.bar(x_pos - width/2, train_r2, width, label='Training R²', 
                  color=COLORS['primary_blue'], alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x_pos + width/2, test_r2, width, label='Test R²', 
                  color=COLORS['accent_orange'], alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{height:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Add gap indicators with lines
    for i, (t, te) in enumerate(zip(train_r2, test_r2)):
        gap = t - te
        mid_x = i
        ax.plot([mid_x - width/4, mid_x + width/4], [te + 0.01, t - 0.01], 
               'k-', linewidth=2, alpha=0.3)
        
        # Gap annotation
        if gap > 0.15:  # Large gap = overfitting
            color = 'red'
            ax.text(mid_x, (t + te)/2, f'Gap:\n{gap:.2f}\nOVERFIT!', 
                   fontsize=9, ha='center', color=color, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        elif gap < 0.1:  # Small gap = good
            color = 'green'
            ax.text(mid_x, t + 0.08, f'Gap: {gap:.2f}\n✓ Good', 
                   fontsize=8, ha='center', color=color, fontweight='bold')
    
    # Highlight best model
    best_idx = 2  # Ridge
    rect = Rectangle((best_idx - 0.5, -0.05), 1, 1.15, 
                     linewidth=4, edgecolor='gold', facecolor='none', 
                     linestyle='--', zorder=10)
    ax.add_patch(rect)
    ax.text(best_idx, 1.08, '★ BEST ★', fontsize=12, ha='center', 
           color='gold', fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    # Horizontal line for acceptable performance
    ax.axhline(y=0.75, color=COLORS['secondary_green'], linestyle='--', 
              linewidth=2, alpha=0.5, label='Acceptable Performance (0.75)')
    
    ax.set_ylabel('R² Score', fontsize=12, fontweight='bold')
    ax.set_title('Performance Comparison: Impact of Regularization', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()
    
    return fig, ax


def generate_fig_15_kfold_illustration(save_path='./figures/fig_15_kfold_illustration.png'):
    """
    Generate figure 15: K-fold cross-validation illustration
    """
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    ax.axis('off')
    
    # Colors for 5 folds
    fold_colors = [COLORS['primary_blue'], COLORS['secondary_green'], 
                   COLORS['accent_orange'], COLORS['warning_red'], '#9D4EDD']
    
    # Full dataset at top
    y_start = 0.9
    rect_height = 0.08
    rect_width = 0.8
    x_start = 0.1
    
    # Draw full dataset
    for i in range(5):
        rect = Rectangle((x_start + i * rect_width/5, y_start), 
                        rect_width/5, rect_height,
                        facecolor=fold_colors[i], edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x_start + (i + 0.5) * rect_width/5, y_start + rect_height/2,
               f'Fold {i+1}', ha='center', va='center', fontsize=10, fontweight='bold')
    
    ax.text(0.5, y_start + rect_height + 0.03, 'Full Training Dataset', 
           ha='center', fontsize=12, fontweight='bold')
    
    # 5 rows showing each fold
    y_positions = [0.72, 0.58, 0.44, 0.30, 0.16]
    
    for fold_idx, y_pos in enumerate(y_positions):
        # Draw 5 sections
        for i in range(5):
            if i == fold_idx:
                # Validation fold - use hatching
                rect = Rectangle((x_start + i * rect_width/5, y_pos), 
                                rect_width/5, rect_height,
                                facecolor=fold_colors[i], edgecolor='black', 
                                linewidth=2, hatch='///', alpha=0.7)
                ax.add_patch(rect)
                ax.text(x_start + (i + 0.5) * rect_width/5, y_pos + rect_height/2,
                       'Valid', ha='center', va='center', fontsize=9, 
                       fontweight='bold', style='italic')
            else:
                # Training fold - solid
                rect = Rectangle((x_start + i * rect_width/5, y_pos), 
                                rect_width/5, rect_height,
                                facecolor=fold_colors[i], edgecolor='black', linewidth=2)
                ax.add_patch(rect)
                ax.text(x_start + (i + 0.5) * rect_width/5, y_pos + rect_height/2,
                       'Train', ha='center', va='center', fontsize=9)
        
        # Label on the left
        ax.text(0.03, y_pos + rect_height/2, f'Fold {fold_idx + 1}:', 
               ha='left', va='center', fontsize=11, fontweight='bold')
        
        # Score on the right
        score = np.random.uniform(0.75, 0.82)
        ax.text(0.93, y_pos + rect_height/2, f'R²₍{fold_idx+1}₎ = {score:.3f}', 
               ha='left', va='center', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))
    
    # Average at bottom
    y_bottom = 0.05
    avg_score = 0.782
    ax.text(0.5, y_bottom, 
           f'Cross-Validation Score = (R²₁ + R²₂ + R²₃ + R²₄ + R²₅) / 5 = {avg_score:.3f}',
           ha='center', fontsize=12, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor=COLORS['light_green'], 
                    edgecolor='black', linewidth=2, pad=0.5))
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='gray', label='Training Section (solid)'),
        Patch(facecolor='gray', hatch='///', label='Validation Section (hatched)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # Title
    ax.text(0.5, 0.98, '5-Fold Cross-Validation Process', 
           ha='center', fontsize=14, fontweight='bold')
    
    # Arrows showing rotation
    arrow_y = 0.74
    for i in range(4):
        arrow = FancyArrowPatch((0.05, y_positions[i] - 0.02), 
                               (0.05, y_positions[i+1] + rect_height + 0.02),
                               arrowstyle='->', mutation_scale=20, 
                               linewidth=2, color='black', alpha=0.5)
        ax.add_patch(arrow)
    
    ax.text(0.02, 0.5, 'Rotate\nValidation\nFold', 
           ha='center', va='center', fontsize=10, fontweight='bold', rotation=90)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()
    
    return fig, ax


def generate_fig_16_grid_search_process(save_path='./figures/fig_16_grid_search_process.png'):
    """
    Generate figure 16: Grid search flowchart
    """
    fig, ax = plt.subplots(figsize=(8, 12), dpi=300)
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    
    # Helper function to draw boxes
    def draw_box(x, y, width, height, text, color, shape='rect'):
        if shape == 'rect':
            box = Rectangle((x - width/2, y - height/2), width, height,
                           facecolor=color, edgecolor='black', linewidth=2)
        elif shape == 'diamond':
            # Diamond shape
            points = np.array([[x, y + height/2], [x + width/2, y], 
                             [x, y - height/2], [x - width/2, y]])
            box = plt.Polygon(points, facecolor=color, edgecolor='black', linewidth=2)
        elif shape == 'parallelogram':
            # Parallelogram for input/output
            offset = 0.3
            points = np.array([[x - width/2 + offset, y + height/2], 
                             [x + width/2 + offset, y + height/2],
                             [x + width/2 - offset, y - height/2], 
                             [x - width/2 - offset, y - height/2]])
            box = plt.Polygon(points, facecolor=color, edgecolor='black', linewidth=2)
        
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=9, 
               fontweight='bold', wrap=True)
    
    def draw_arrow(x1, y1, x2, y2, label=''):
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                               arrowstyle='->', mutation_scale=20, 
                               linewidth=2, color='black')
        ax.add_patch(arrow)
        if label:
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mid_x + 0.5, mid_y, label, fontsize=8, 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Flowchart
    y = 13
    
    # Input
    draw_box(5, y, 3, 0.8, 'Training Data\nλ values = [0.01, 0.1, 1, 10, 100]', 
            COLORS['light_blue'], 'parallelogram')
    draw_arrow(5, y - 0.5, 5, y - 1.2)
    
    y -= 1.5
    # Start loop
    draw_box(5, y, 3.5, 0.8, 'FOR each λ value:', COLORS['light_orange'])
    draw_arrow(5, y - 0.5, 5, y - 1.2)
    
    y -= 1.5
    # Run CV
    draw_box(5, y, 3.5, 0.8, 'Run 5-Fold Cross-Validation', 'white')
    draw_arrow(5, y - 0.5, 5, y - 1.2)
    
    y -= 1.5
    # Nested loop indication
    draw_box(5, y, 4, 1.2, 'FOR each fold (1 to 5):\n' + 
            '• Train on 4 folds\n• Validate on 1 fold\n• Store score', 
            COLORS['light_green'])
    draw_arrow(5, y - 0.7, 5, y - 1.5)
    
    y -= 2
    # Compute average
    draw_box(5, y, 3.5, 0.8, 'Compute Average CV Score', 'white')
    draw_arrow(5, y - 0.5, 5, y - 1.2)
    
    y -= 1.5
    # Store result
    draw_box(5, y, 3, 0.8, 'Store (λ, CV Score)', 'white')
    
    # Loop back arrow
    draw_arrow(7, y, 8.5, y)
    draw_arrow(8.5, y, 8.5, y + 6)
    draw_arrow(8.5, y + 6, 5, y + 6)
    ax.text(8.7, y + 3, 'Next λ', fontsize=8, rotation=90, va='center')
    
    draw_arrow(5, y - 0.5, 5, y - 1.2)
    
    y -= 1.5
    # Compare
    draw_box(5, y, 3.5, 0.8, 'Compare all CV Scores', COLORS['light_orange'])
    draw_arrow(5, y - 0.5, 5, y - 1.2)
    
    y -= 1.5
    # Decision
    draw_box(5, y, 2.5, 1, 'Best λ\nfound?', 'yellow', 'diamond')
    draw_arrow(5, y - 0.6, 5, y - 1.4)
    
    y -= 1.7
    # Select best
    draw_box(5, y, 3, 0.8, 'Select λ with\nLowest CV Error', COLORS['secondary_green'])
    draw_arrow(5, y - 0.5, 5, y - 1.2)
    
    y -= 1.5
    # Retrain
    draw_box(5, y, 3.5, 0.8, 'Retrain on\nFull Training Set', COLORS['secondary_green'])
    draw_arrow(5, y - 0.5, 5, y - 1.2)
    
    y -= 1.5
    # Output
    draw_box(5, y, 3, 0.8, 'Best Model Ready!\nEvaluate on Test Set', 
            COLORS['light_green'], 'parallelogram')
    
    # Title
    ax.text(5, 13.8, 'Grid Search with Cross-Validation Algorithm', 
           ha='center', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()
    
    return fig, ax


def generate_fig_17_cv_error_vs_lambda(save_path='./figures/fig_17_cv_error_vs_lambda.png'):
    """
    Generate figure 17: CV error vs lambda showing U-curve
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    # Lambda values (log scale)
    log_lambdas = np.linspace(-3, 3, 50)
    lambdas = 10 ** log_lambdas
    
    # Training error (increases with lambda)
    train_error = 5 + 8 * (1 - np.exp(-log_lambdas/2))
    
    # CV error (U-shaped)
    cv_error = 5 + 3 * np.exp(-log_lambdas/2) + 0.8 * (log_lambdas - 0.5)**2
    
    # Add some noise/uncertainty to CV
    cv_std = np.random.uniform(0.3, 0.8, len(log_lambdas))
    
    # Plot
    ax.plot(log_lambdas, train_error, 'o-', color=COLORS['primary_blue'], 
           linewidth=3, markersize=6, label='Training Error', alpha=0.8)
    ax.errorbar(log_lambdas, cv_error, yerr=cv_std, 
               fmt='s-', color=COLORS['accent_orange'], linewidth=3, 
               markersize=6, label='CV Error (± std)', alpha=0.8, capsize=3)
    
    # Find optimal lambda
    optimal_idx = np.argmin(cv_error)
    optimal_log_lambda = log_lambdas[optimal_idx]
    optimal_lambda = lambdas[optimal_idx]
    
    # Mark optimal
    ax.axvline(x=optimal_log_lambda, color=COLORS['secondary_green'], 
              linestyle='--', linewidth=2.5, label=f'Optimal λ = {optimal_lambda:.2f}', 
              alpha=0.8)
    
    ax.plot(optimal_log_lambda, cv_error[optimal_idx], '*', 
           markersize=20, color='gold', markeredgecolor='black', 
           markeredgewidth=2, zorder=10, label='Minimum CV Error')
    
    # Shade regions
    ax.axvspan(-3, optimal_log_lambda, alpha=0.1, color=COLORS['warning_red'])
    ax.axvspan(optimal_log_lambda, 3, alpha=0.1, color=COLORS['warning_red'])
    ax.axvspan(optimal_log_lambda - 0.3, optimal_log_lambda + 0.3, 
              alpha=0.15, color=COLORS['secondary_green'])
    
    # Annotations
    ax.text(-2, 13, 'λ → 0\nUnderfitting\n(Too little regularization)', 
           fontsize=10, ha='center', color=COLORS['warning_red'], fontweight='bold',
           bbox=dict(boxstyle='round', facecolor=COLORS['light_red'], alpha=0.7))
    
    ax.text(2, 13, 'λ → ∞\nUnderfitting\n(Too much regularization)', 
           fontsize=10, ha='center', color=COLORS['warning_red'], fontweight='bold',
           bbox=dict(boxstyle='round', facecolor=COLORS['light_red'], alpha=0.7))
    
    ax.text(optimal_log_lambda, 15, 'Sweet Spot\n(Optimal λ)', 
           fontsize=10, ha='center', color=COLORS['secondary_green'], fontweight='bold',
           bbox=dict(boxstyle='round', facecolor=COLORS['light_green'], alpha=0.7))
    
    ax.annotate('CV error minimized here', 
               xy=(optimal_log_lambda, cv_error[optimal_idx]), 
               xytext=(optimal_log_lambda + 1, cv_error[optimal_idx] + 2),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'),
               fontsize=10, ha='center',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.annotate('Training error\nalways increases with λ', 
               xy=(1, train_error[-20]), 
               xytext=(1.5, 16),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'),
               fontsize=10, ha='center',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('log₁₀(λ) [Regularization Strength]', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Squared Error', fontsize=12, fontweight='bold')
    ax.set_title('Cross-Validation Error vs. Regularization Strength', 
                fontsize=14, fontweight='bold')
    ax.set_xlim(-3, 3)
    ax.set_ylim(4, 18)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()
    
    return fig, ax


def generate_fig_18_three_dataset_roles(save_path='./figures/fig_18_three_dataset_roles.png'):
    """
    Generate figure 18: Three dataset roles diagram
    """
    fig, ax = plt.subplots(figsize=(12, 4), dpi=300)
    ax.axis('off')
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    
    # Full dataset
    rect_full = Rectangle((0.5, 2.5), 10, 1, facecolor='lightgray', 
                          edgecolor='black', linewidth=3)
    ax.add_patch(rect_full)
    ax.text(5.5, 3, 'Full Dataset (100%)', ha='center', va='center', 
           fontsize=12, fontweight='bold')
    
    # Split arrow
    arrow1 = FancyArrowPatch((5.5, 2.4), (3, 2),
                            arrowstyle='->', mutation_scale=20, 
                            linewidth=3, color='black')
    arrow2 = FancyArrowPatch((5.5, 2.4), (9, 2),
                            arrowstyle='->', mutation_scale=20, 
                            linewidth=3, color='black')
    ax.add_patch(arrow1)
    ax.add_patch(arrow2)
    
    ax.text(5.5, 2.2, 'SPLIT FIRST!', ha='center', fontsize=10, 
           fontweight='bold', color='red',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    # Training set (70%)
    train_rect = Rectangle((0.5, 0.5), 7, 1.2, facecolor=COLORS['light_blue'], 
                           edgecolor='black', linewidth=2)
    ax.add_patch(train_rect)
    ax.text(4, 1.3, 'Training Set (70%)', ha='center', fontsize=11, fontweight='bold')
    ax.text(4, 0.9, '• Fit model parameters (w)\n• Used for cross-validation\n• Never touch test set!', 
           ha='center', fontsize=9, va='center')
    
    # Show CV within training
    cv_y = 0.5
    for i in range(5):
        small_rect = Rectangle((0.5 + i * 1.4, cv_y), 1.3, 0.15,
                               facecolor=COLORS['secondary_green'] if i != 2 else 'white',
                               edgecolor='black', linewidth=1,
                               hatch=None if i != 2 else '///')
        ax.add_patch(small_rect)
    ax.text(4, 0.35, '5-fold CV (within training only)', ha='center', fontsize=8, style='italic')
    
    # Arrow from training
    arrow_train = FancyArrowPatch((4, 0.45), (4, 0.1),
                                 arrowstyle='->', mutation_scale=15, 
                                 linewidth=2, color='black')
    ax.add_patch(arrow_train)
    ax.text(4.5, 0.25, 'Tune λ', fontsize=9, fontweight='bold')
    
    # Test set (30%)
    test_rect = Rectangle((8, 0.5), 3, 1.2, facecolor=COLORS['light_orange'], 
                           edgecolor='black', linewidth=2)
    ax.add_patch(test_rect)
    ax.text(9.5, 1.3, 'Test Set (30%)', ha='center', fontsize=11, fontweight='bold')
    ax.text(9.5, 0.9, '• Final evaluation ONLY\n• Touch once at the end\n• Reports performance', 
           ha='center', fontsize=9, va='center')
    
    # Warning on test set
    warning_rect = Rectangle((8, 0.3), 3, 0.15, facecolor='red', 
                            edgecolor='black', linewidth=2)
    ax.add_patch(warning_rect)
    ax.text(9.5, 0.375, '⚠ SACRED - TOUCH ONLY ONCE! ⚠', ha='center', va='center',
           fontsize=8, fontweight='bold', color='white')
    
    # Timeline at bottom
    ax.text(4, -0.2, 'During Development', ha='center', fontsize=10, 
           fontweight='bold', color=COLORS['primary_blue'])
    ax.text(9.5, -0.2, 'Final Evaluation', ha='center', fontsize=10, 
           fontweight='bold', color=COLORS['accent_orange'])
    
    # Title
    ax.text(5.5, 3.8, 'The Three Dataset Roles: Proper Data Splitting', 
           ha='center', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()
    
    return fig, ax


def generate_fig_19_complete_workflow(save_path='./figures/fig_19_complete_workflow.png'):
    """
    Generate figure 19: Complete ML workflow flowchart
    """
    fig, ax = plt.subplots(figsize=(10, 14), dpi=300)
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 16)
    
    def draw_box(x, y, width, height, text, color, shape='rect'):
        if shape == 'rect':
            box = Rectangle((x - width/2, y - height/2), width, height,
                           facecolor=color, edgecolor='black', linewidth=2)
        elif shape == 'diamond':
            points = np.array([[x, y + height/2], [x + width/2, y], 
                             [x, y - height/2], [x - width/2, y]])
            box = plt.Polygon(points, facecolor=color, edgecolor='black', linewidth=2)
        
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=9, 
               fontweight='bold', multialignment='center')
    
    def draw_arrow(x1, y1, x2, y2):
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                               arrowstyle='->', mutation_scale=20, 
                               linewidth=2, color='black')
        ax.add_patch(arrow)
    
    # Workflow steps
    y = 15
    
    # 1. Load Data
    draw_box(5, y, 4, 0.8, '1. Load & Explore Data', COLORS['light_blue'])
    draw_arrow(5, y - 0.5, 5, y - 1.2)
    
    y -= 1.5
    # 2. Split
    draw_box(5, y, 4, 0.8, '2. Train/Test Split\n⚠ DO THIS FIRST!', 'yellow')
    ax.text(8, y, '← CRITICAL', fontsize=10, fontweight='bold', color='red')
    draw_arrow(5, y - 0.5, 5, y - 1.2)
    
    y -= 1.5
    # Bracket for training phase
    ax.plot([0.5, 0.5], [y + 0.3, 2], 'k-', linewidth=3)
    ax.text(0.3, (y + 2.3)/2, 'Training\nPhase\n(Only use\ntraining\ndata)', 
           fontsize=9, fontweight='bold', rotation=90, va='center', ha='center',
           bbox=dict(boxstyle='round', facecolor=COLORS['light_blue'], alpha=0.7))
    
    # 3. Feature Engineering
    draw_box(5, y, 4, 0.8, '3. Feature Engineering\n(Polynomial, Scaling)', COLORS['light_green'])
    draw_arrow(5, y - 0.5, 5, y - 1.2)
    
    # Checkpoint
    y -= 1.2
    draw_box(7.5, y, 2, 0.5, 'Fit on TRAIN\nonly!', 'red')
    
    y -= 0.8
    # 4. Cross-validation loop
    draw_box(5, y, 4.5, 1.2, '4. Cross-Validation\n• Try different λ values\n• 5-fold CV on training set\n• Select best λ', 
            COLORS['light_orange'])
    
    # Loop back arrow
    draw_arrow(7.5, y, 8.5, y)
    draw_arrow(8.5, y, 8.5, y + 1.5)
    draw_arrow(8.5, y + 1.5, 5, y + 1.5)
    ax.text(8.7, y + 0.75, 'Try next λ', fontsize=8, rotation=90, va='center')
    
    draw_arrow(5, y - 0.7, 5, y - 1.5)
    
    y -= 2
    # 5. Select best
    draw_box(5, y, 4, 0.8, '5. Select Best Model\n(Lowest CV error)', COLORS['secondary_green'])
    draw_arrow(5, y - 0.5, 5, y - 1.2)
    
    y -= 1.5
    # 6. Retrain
    draw_box(5, y, 4, 0.8, '6. Retrain on Full Training Set\n(with best λ)', COLORS['secondary_green'])
    draw_arrow(5, y - 0.5, 5, y - 1.2)
    
    y -= 1.5
    # Checkpoint
    draw_box(7.5, y, 2, 0.5, 'No test\ndata yet!', 'red')
    
    # 7. Final evaluation
    draw_box(5, y, 4, 0.8, '7. Final Evaluation on Test Set\n⚠ ONE TIME ONLY!', COLORS['light_red'])
    draw_arrow(5, y - 0.5, 5, y - 1.2)
    
    y -= 1.5
    # 8. Deploy
    draw_box(5, y, 4, 0.8, '8. Deploy Model\n(Ready for production)', COLORS['light_green'])
    
    # Leakage warnings (stop signs)
    checkpoints = [
        (2, 12, 'Checkpoint:\nFit scaler\nonly on train'),
        (2, 9, 'Checkpoint:\nNo test data\nin CV'),
        (2, 4.5, 'Checkpoint:\nHyperparameters\nfrozen')
    ]
    
    for x, y_cp, text in checkpoints:
        # Stop sign shape (octagon approximation)
        octagon = plt.Circle((x, y_cp), 0.4, color='red', ec='darkred', linewidth=3)
        ax.add_patch(octagon)
        ax.text(x, y_cp, '🛑', fontsize=20, ha='center', va='center')
        ax.text(x, y_cp - 0.7, text, fontsize=7, ha='center',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    # Title
    ax.text(5, 15.8, 'Complete Machine Learning Workflow', 
           ha='center', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()
    
    return fig, ax


def generate_fig_20_bias_variance_summary(save_path='./figures/fig_20_bias_variance_summary.png'):
    """
    Generate figure 20: Comprehensive bias-variance summary
    """
    fig = plt.figure(figsize=(14, 8), dpi=300)
    gs = GridSpec(3, 3, height_ratios=[2, 1, 1], hspace=0.3, wspace=0.3)
    
    # Generate data for examples
    np.random.seed(42)
    x = np.linspace(0, 10, 50)
    y_true = 2 + 0.5*x - 0.05*x**2
    y = y_true + np.random.normal(0, 0.5, 50)
    x_smooth = np.linspace(0, 10, 200)
    y_true_smooth = 2 + 0.5*x_smooth - 0.05*x_smooth**2
    
    # Top row: Three scenarios
    scenarios = [
        ('Underfitting\n(High Bias)', 1, COLORS['warning_red'], 0),
        ('Sweet Spot\n(Balanced)', 2, COLORS['secondary_green'], 1),
        ('Overfitting\n(High Variance)', 8, COLORS['primary_blue'], 2)
    ]
    
    for idx, (title, degree, color, col) in enumerate(scenarios):
        ax = fig.add_subplot(gs[0, col])
        
        # Fit polynomial
        poly = PolynomialFeatures(degree=degree)
        x_poly = poly.fit_transform(x.reshape(-1, 1))
        lr = LinearRegression()
        lr.fit(x_poly, y)
        
        x_smooth_poly = poly.transform(x_smooth.reshape(-1, 1))
        y_pred = lr.predict(x_smooth_poly)
        
        # Plot
        ax.scatter(x, y, alpha=0.5, s=40, color='black', label='Data')
        ax.plot(x_smooth, y_true_smooth, '--', color='gray', linewidth=2, alpha=0.6)
        ax.plot(x_smooth, y_pred, color=color, linewidth=3)
        
        ax.set_title(title, fontsize=12, fontweight='bold', color=color)
        ax.set_ylim(-2, 5)
        ax.grid(True, alpha=0.3)
        
        if col == 0:
            ax.set_ylabel('Prediction', fontsize=11)
        if col == 1:
            # Add star
            ax.plot(5, 2.5, marker='*', markersize=25, color='gold', 
                   markeredgecolor='black', markeredgewidth=2, zorder=10)
        
        ax.set_xlabel('Input', fontsize=11)
    
    # Middle row: Control mechanisms
    ax_control = fig.add_subplot(gs[1, :])
    
    # Draw spectrum
    x_spectrum = np.linspace(0, 10, 100)
    y_spectrum = 5
    
    # Color gradient
    for i in range(len(x_spectrum) - 1):
        if x_spectrum[i] < 3.5:
            color = COLORS['warning_red']
        elif x_spectrum[i] > 6.5:
            color = COLORS['warning_red']
        else:
            color = COLORS['secondary_green']
        ax_control.plot([x_spectrum[i], x_spectrum[i+1]], [y_spectrum, y_spectrum], 
                       color=color, linewidth=20, alpha=0.3)
    
    # Main spectrum line
    ax_control.plot([0, 10], [y_spectrum, y_spectrum], 'k-', linewidth=3)
    
    # Markers
    ax_control.plot(2, y_spectrum, 'o', markersize=15, color=COLORS['warning_red'], 
                   markeredgecolor='black', markeredgewidth=2)
    ax_control.text(2, y_spectrum - 0.8, 'High Bias\nUnderfitting', ha='center', fontsize=10)
    
    ax_control.plot(5, y_spectrum, '*', markersize=25, color='gold', 
                   markeredgecolor='black', markeredgewidth=2)
    ax_control.text(5, y_spectrum - 0.8, 'Optimal\nBalance', ha='center', fontsize=10, 
                   fontweight='bold', color=COLORS['secondary_green'])
    
    ax_control.plot(8, y_spectrum, 'o', markersize=15, color=COLORS['primary_blue'], 
                   markeredgecolor='black', markeredgewidth=2)
    ax_control.text(8, y_spectrum - 0.8, 'High Variance\nOverfitting', ha='center', fontsize=10)
    
    # Control arrows
    arrow_left = FancyArrowPatch((1, y_spectrum + 1), (4, y_spectrum + 1),
                                arrowstyle='<->', mutation_scale=20, 
                                linewidth=2, color='black')
    ax_control.add_patch(arrow_left)
    ax_control.text(2.5, y_spectrum + 1.5, 'Add Features\n(Feature Engineering)', 
                   ha='center', fontsize=9, fontweight='bold')
    
    arrow_right = FancyArrowPatch((6, y_spectrum + 1), (9, y_spectrum + 1),
                                 arrowstyle='<->', mutation_scale=20, 
                                 linewidth=2, color='black')
    ax_control.add_patch(arrow_right)
    ax_control.text(7.5, y_spectrum + 1.5, 'Add Regularization\n(Increase λ)', 
                   ha='center', fontsize=9, fontweight='bold')
    
    # Lambda slider
    ax_control.plot([0, 10], [y_spectrum - 1.5, y_spectrum - 1.5], 'k-', linewidth=2)
    ax_control.text(0, y_spectrum - 2, 'λ = 0', ha='center', fontsize=9)
    ax_control.text(5, y_spectrum - 2, 'λ_optimal', ha='center', fontsize=9, 
                   fontweight='bold', color=COLORS['secondary_green'])
    ax_control.text(10, y_spectrum - 2, 'λ → ∞', ha='center', fontsize=9)
    
    ax_control.set_xlim(-0.5, 10.5)
    ax_control.set_ylim(2, 7)
    ax_control.axis('off')
    ax_control.set_title('Control Mechanisms: Moving Along the Spectrum', 
                        fontsize=12, fontweight='bold')
    
    # Bottom row: Error decomposition
    ax_error = fig.add_subplot(gs[2, :])
    
    categories = ['Underfitting', 'Optimal', 'Overfitting']
    bias_vals = [5, 1, 0.5]
    variance_vals = [0.5, 1, 5]
    irreducible = [1, 1, 1]
    
    x_pos = np.arange(len(categories))
    width = 0.5
    
    p1 = ax_error.bar(x_pos, bias_vals, width, label='Bias²', 
                     color=COLORS['warning_red'], alpha=0.7, edgecolor='black', linewidth=1.5)
    p2 = ax_error.bar(x_pos, variance_vals, width, bottom=bias_vals, label='Variance',
                     color=COLORS['primary_blue'], alpha=0.7, edgecolor='black', linewidth=1.5)
    p3 = ax_error.bar(x_pos, irreducible, width,
                     bottom=[b+v for b,v in zip(bias_vals, variance_vals)],
                     label='Irreducible Error', color=COLORS['neutral_gray'], 
                     alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for i, (b, v, irr) in enumerate(zip(bias_vals, variance_vals, irreducible)):
        total = b + v + irr
        ax_error.text(i, total + 0.3, f'Total: {total:.1f}', 
                     ha='center', fontweight='bold', fontsize=10)
    
    ax_error.set_ylabel('Error Contribution', fontsize=11, fontweight='bold')
    ax_error.set_xticks(x_pos)
    ax_error.set_xticklabels(categories, fontsize=11)
    ax_error.set_title('Error Decomposition: Total Error = Bias² + Variance + Irreducible', 
                      fontsize=11, fontweight='bold')
    ax_error.legend(loc='upper right', ncol=3, fontsize=10)
    ax_error.grid(axis='y', alpha=0.3)
    ax_error.set_ylim(0, 8)
    
    # Highlight optimal
    rect_highlight = Rectangle((1 - width/2, -0.5), width, 8.5, 
                               linewidth=3, edgecolor='gold', facecolor='none', 
                               linestyle='--')
    ax_error.add_patch(rect_highlight)
    
    plt.suptitle('The Bias-Variance Tradeoff: Complete Picture', 
                fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()
    
    return fig


# Main execution
if __name__ == "__main__":
    print("Generating all figures for Lesson 3 Lecture Notes...")
    print("=" * 60)
    
    # Generate all figures
    generate_fig_01_netflix_improvement()
    generate_fig_02_curved_relationship()
    generate_fig_03_polynomial_progression()
    generate_fig_04_interaction_effect()
    generate_fig_05_overfitting_visualization()
    generate_fig_06_bias_variance_tradeoff()
    generate_fig_07_train_test_error_vs_complexity()
    generate_fig_08_overfitting_indicators()
    generate_fig_09_ridge_coefficient_paths()
    generate_fig_10_ridge_geometric()
    generate_fig_11_lasso_coefficient_paths()
    generate_fig_12_lasso_geometric()
    generate_fig_13_ridge_vs_lasso_comparison()
    generate_fig_14_regularization_results()
    generate_fig_15_kfold_illustration()
    generate_fig_16_grid_search_process()
    generate_fig_17_cv_error_vs_lambda()
    generate_fig_18_three_dataset_roles()
    generate_fig_19_complete_workflow()
    generate_fig_20_bias_variance_summary()
    
    print("=" * 60)
    print("All figures generated successfully!")
    print(f"Figures saved in: ./figures/")