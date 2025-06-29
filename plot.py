import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

def plot_histogram(data, labels, feature_idx, feature_name, class_filter="all", bins=20, alpha=0.7):
    """
    Plot histogram with class-based filtering
    
    Args:
        data: Feature data matrix
        labels: Class labels
        feature_idx: Index of the feature to plot
        feature_name: Name of the feature
        class_filter: "all", "ok", "ko", or "compare"
        bins: Number of bins for histogram
        alpha: Transparency level
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if class_filter == "all":
        ax.hist(data[feature_idx, :], bins=bins, alpha=alpha, 
                color='blue', edgecolor='black', label='All Classes')
        ax.set_title(f"Histogram of {feature_name} (All Classes)")
    elif class_filter == "compare":
        # Plot both classes for comparison
        ax.hist(data[feature_idx, labels == 0], bins=bins, alpha=alpha, 
                color='green', edgecolor='black', label='OK Class')
        ax.hist(data[feature_idx, labels == 1], bins=bins, alpha=alpha, 
                color='red', edgecolor='black', label='KO Class')
        ax.set_title(f"Histogram of {feature_name} - OK vs KO Classes")
    else:
        # Plot specific class
        label_val = 0 if class_filter.lower() == "ok" else 1
        color = 'green' if class_filter.lower() == "ok" else 'red'
        ax.hist(data[feature_idx, labels == label_val], bins=bins, alpha=alpha,
                color=color, edgecolor='black', label=f'{class_filter.upper()} Class')
        ax.set_title(f"Histogram of {feature_name} ({class_filter.upper()} Class)")
    
    ax.set_xlabel(feature_name)
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig

def plot_boxplot(data, labels, feature_idx, feature_name, class_filter="all"):
    """
    Plot boxplot with class-based filtering
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if class_filter == "all":
        ax.boxplot(data[feature_idx, :], labels=[feature_name])
        ax.set_title(f"Boxplot of {feature_name} (All Classes)")
    elif class_filter == "compare":
        # Compare both classes
        ok_data = data[feature_idx, labels == 0]
        ko_data = data[feature_idx, labels == 1]
        ax.boxplot([ok_data, ko_data], labels=['OK Class', 'KO Class'])
        ax.set_title(f"Boxplot of {feature_name} - OK vs KO Classes")
    else:
        # Plot specific class
        label_val = 0 if class_filter.lower() == "ok" else 1
        class_data = data[feature_idx, labels == label_val]
        ax.boxplot(class_data, labels=[f'{feature_name} ({class_filter.upper()})'])
        ax.set_title(f"Boxplot of {feature_name} ({class_filter.upper()} Class)")
    
    ax.set_ylabel(feature_name)
    ax.grid(True, alpha=0.3)
    
    return fig

def plot_scatter(data, labels, feature1_idx, feature2_idx, feature1_name, feature2_name, class_filter="all"):
    """
    Plot scatter plot with class-based filtering
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if class_filter == "all":
        ax.scatter(data[feature1_idx, :], data[feature2_idx, :], alpha=0.6, s=30)
        ax.set_title(f"Scatter Plot: {feature1_name} vs {feature2_name} (All Classes)")
    elif class_filter == "compare":
        # Plot both classes with different colors
        ax.scatter(data[feature1_idx, labels == 0], data[feature2_idx, labels == 0], 
                  alpha=0.6, s=30, color='green', label='OK Class')
        ax.scatter(data[feature1_idx, labels == 1], data[feature2_idx, labels == 1], 
                  alpha=0.6, s=30, color='red', label='KO Class')
        ax.set_title(f"Scatter Plot: {feature1_name} vs {feature2_name} - OK vs KO Classes")
        ax.legend()
    else:
        # Plot specific class
        label_val = 0 if class_filter.lower() == "ok" else 1
        color = 'green' if class_filter.lower() == "ok" else 'red'
        ax.scatter(data[feature1_idx, labels == label_val], data[feature2_idx, labels == label_val], 
                  alpha=0.6, s=30, color=color, label=f'{class_filter.upper()} Class')
        ax.set_title(f"Scatter Plot: {feature1_name} vs {feature2_name} ({class_filter.upper()} Class)")
        ax.legend()
    
    ax.set_xlabel(feature1_name)
    ax.set_ylabel(feature2_name)
    ax.grid(True, alpha=0.3)
    
    return fig

def plot_correlation(data, labels, feature1_idx, feature2_idx, feature1_name, feature2_name):
    """
    Plot correlation analysis between two features
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Scatter plot with regression line
    ax1.scatter(data[feature1_idx, :], data[feature2_idx, :], alpha=0.6, s=30)
    
    # Add regression line
    z = np.polyfit(data[feature1_idx, :], data[feature2_idx, :], 1)
    p = np.poly1d(z)
    ax1.plot(data[feature1_idx, :], p(data[feature1_idx, :]), "r--", alpha=0.8)
    
    # Calculate correlation coefficient
    correlation = np.corrcoef(data[feature1_idx, :], data[feature2_idx, :])[0, 1]
    ax1.set_title(f"Correlation: {feature1_name} vs {feature2_name}\n(r = {correlation:.3f})")
    ax1.set_xlabel(feature1_name)
    ax1.set_ylabel(feature2_name)
    ax1.grid(True, alpha=0.3)
    
    # Correlation by class
    ok_corr = np.corrcoef(data[feature1_idx, labels == 0], data[feature2_idx, labels == 0])[0, 1]
    ko_corr = np.corrcoef(data[feature1_idx, labels == 1], data[feature2_idx, labels == 1])[0, 1]
    
    ax2.bar(['OK Class', 'KO Class'], [ok_corr, ko_corr], 
            color=['green', 'red'], alpha=0.7)
    ax2.set_title(f"Correlation by Class")
    ax2.set_ylabel('Correlation Coefficient')
    ax2.set_ylim(-1, 1)
    ax2.grid(True, alpha=0.3)
    
    # Add correlation values on bars
    ax2.text(0, ok_corr + 0.02, f'{ok_corr:.3f}', ha='center', va='bottom')
    ax2.text(1, ko_corr + 0.02, f'{ko_corr:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    return fig

def plot_feature_comparison(data, labels, feature_idx, feature_name):
    """
    Plot comprehensive comparison of a feature across classes
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Histogram comparison
    ax1.hist(data[feature_idx, labels == 0], bins=20, alpha=0.7, 
             color='green', edgecolor='black', label='OK Class')
    ax1.hist(data[feature_idx, labels == 1], bins=20, alpha=0.7, 
             color='red', edgecolor='black', label='KO Class')
    ax1.set_title(f"Histogram Comparison: {feature_name}")
    ax1.set_xlabel(feature_name)
    ax1.set_ylabel('Frequency')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Boxplot comparison
    ok_data = data[feature_idx, labels == 0]
    ko_data = data[feature_idx, labels == 1]
    ax2.boxplot([ok_data, ko_data], labels=['OK Class', 'KO Class'])
    ax2.set_title(f"Boxplot Comparison: {feature_name}")
    ax2.set_ylabel(feature_name)
    ax2.grid(True, alpha=0.3)
    
    # Violin plot
    ax3.violinplot([ok_data, ko_data], positions=[1, 2])
    ax3.set_xticks([1, 2])
    ax3.set_xticklabels(['OK Class', 'KO Class'])
    ax3.set_title(f"Violin Plot: {feature_name}")
    ax3.set_ylabel(feature_name)
    ax3.grid(True, alpha=0.3)
    
    # Statistics summary
    ok_mean = np.mean(ok_data)
    ko_mean = np.mean(ko_data)
    ok_std = np.std(ok_data)
    ko_std = np.std(ko_data)
    
    stats_text = f"""Statistics Summary:
    
OK Class:
  Mean: {ok_mean:.2f}
  Std: {ok_std:.2f}
  Count: {len(ok_data)}

KO Class:
  Mean: {ko_mean:.2f}
  Std: {ko_std:.2f}
  Count: {len(ko_data)}

Difference: {abs(ok_mean - ko_mean):.2f}"""
    
    ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes, 
             fontsize=12, verticalalignment='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    ax4.set_title("Statistical Summary")
    ax4.axis('off')
    
    plt.tight_layout()
    return fig

def plot_distribution_analysis(data, labels, feature_idx, feature_name):
    """
    Plot detailed distribution analysis including density plots and Q-Q plots
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    ok_data = data[feature_idx, labels == 0]
    ko_data = data[feature_idx, labels == 1]
    
    # Density plot
    ax1.hist(ok_data, bins=20, alpha=0.7, density=True, color='green', label='OK Class')
    ax1.hist(ko_data, bins=20, alpha=0.7, density=True, color='red', label='KO Class')
    ax1.set_title(f"Density Plot: {feature_name}")
    ax1.set_xlabel(feature_name)
    ax1.set_ylabel('Density')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Q-Q plot for OK class
    stats.probplot(ok_data, dist="norm", plot=ax2)
    ax2.set_title(f"Q-Q Plot: {feature_name} (OK Class)")
    ax2.grid(True, alpha=0.3)
    
    # Q-Q plot for KO class
    stats.probplot(ko_data, dist="norm", plot=ax3)
    ax3.set_title(f"Q-Q Plot: {feature_name} (KO Class)")
    ax3.grid(True, alpha=0.3)
    
    # Statistical test results
    # Perform t-test
    t_stat, p_value = stats.ttest_ind(ok_data, ko_data)
    
    # Perform Mann-Whitney U test
    u_stat, u_p_value = stats.mannwhitneyu(ok_data, ko_data, alternative='two-sided')
    
    test_text = f"""Statistical Tests:

T-Test:
  t-statistic: {t_stat:.3f}
  p-value: {p_value:.3f}

Mann-Whitney U Test:
  U-statistic: {u_stat:.3f}
  p-value: {u_p_value:.3f}

Interpretation:
  {'Significant difference' if p_value < 0.05 else 'No significant difference'}"""
    
    ax4.text(0.1, 0.5, test_text, transform=ax4.transAxes, 
             fontsize=11, verticalalignment='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    ax4.set_title("Statistical Tests")
    ax4.axis('off')
    
    plt.tight_layout()
    return fig

def plot_statistics_summary(data, labels, feature_idx, feature_name):
    """
    Display simplified statistical summary for a feature across classes
    """
    ok_data = data[feature_idx, labels == 0]
    ko_data = data[feature_idx, labels == 1]
    
    # Calculate simplified statistics
    ok_stats = {
        'count': len(ok_data),
        'mean': np.mean(ok_data),
        'median': np.median(ok_data),
        'mode': stats.mode(ok_data, keepdims=True)[0][0],
        'std': np.std(ok_data),
        'variance': np.var(ok_data)
    }
    
    ko_stats = {
        'count': len(ko_data),
        'mean': np.mean(ko_data),
        'median': np.median(ko_data),
        'mode': stats.mode(ko_data, keepdims=True)[0][0],
        'std': np.std(ko_data),
        'variance': np.var(ko_data)
    }
    
    # Perform T-test only
    t_stat, t_p_value = stats.ttest_ind(ok_data, ko_data)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Basic statistics comparison
    metrics = ['mean', 'median', 'mode', 'std', 'variance']
    ok_values = [ok_stats[m] for m in metrics]
    ko_values = [ko_stats[m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax1.bar(x - width/2, ok_values, width, label='OK Class', color='green', alpha=0.7)
    ax1.bar(x + width/2, ko_values, width, label='KO Class', color='red', alpha=0.7)
    ax1.set_xlabel('Statistical Measures')
    ax1.set_ylabel('Value')
    ax1.set_title(f'Statistics Comparison: {feature_name}')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # T-test results
    test_text = f"""T-Test Results:

t-statistic: {t_stat:.4f}
p-value: {t_p_value:.4f}

Interpretation:
{'Significant difference' if t_p_value < 0.05 else 'No significant difference'}

Sample Sizes:
OK Class: {ok_stats['count']}
KO Class: {ko_stats['count']}"""
    
    ax2.text(0.1, 0.5, test_text, transform=ax2.transAxes, 
             fontsize=12, verticalalignment='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    ax2.set_title("T-Test Results")
    ax2.axis('off')
    
    plt.tight_layout()
    return fig

def display_detailed_statistics(data, labels, feature_idx, feature_name):
    """
    Display simplified statistics in a formatted text format
    """
    ok_data = data[feature_idx, labels == 0]
    ko_data = data[feature_idx, labels == 1]
    
    # Calculate simplified statistics
    ok_stats = {
        'count': len(ok_data),
        'mean': np.mean(ok_data),
        'median': np.median(ok_data),
        'mode': stats.mode(ok_data, keepdims=True)[0][0],
        'std': np.std(ok_data),
        'variance': np.var(ok_data)
    }
    
    ko_stats = {
        'count': len(ko_data),
        'mean': np.mean(ko_data),
        'median': np.median(ko_data),
        'mode': stats.mode(ko_data, keepdims=True)[0][0],
        'std': np.std(ko_data),
        'variance': np.var(ko_data)
    }
    
    # Perform T-test only
    t_stat, t_p_value = stats.ttest_ind(ok_data, ko_data)
    
    return {
        'ok_stats': ok_stats,
        'ko_stats': ko_stats,
        't_test': {'statistic': t_stat, 'p_value': t_p_value}
    }

def plot_decision_boundary(data, labels, feature1_idx, feature2_idx, feature1_name, feature2_name, model_type="logistic"):
    """
    Plot decision boundary for 2D features using Logistic Regression or SVM
    
    Args:
        data: Feature data matrix
        labels: Class labels
        feature1_idx: Index of first feature
        feature2_idx: Index of second feature
        feature1_name: Name of first feature
        feature2_name: Name of second feature
        model_type: "logistic" or "svm"
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt
    
    # Extract the two features
    X = data[[feature1_idx, feature2_idx], :].T  # Transpose to get (n_samples, 2)
    y = labels
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create and train the model
    if model_type == "logistic":
        model = LogisticRegression(random_state=42, max_iter=1000)
        model_name = "Logistic Regression"
    elif model_type == "svm":
        model = SVC(kernel='linear', random_state=42)
        model_name = "SVM (Linear)"
    else:
        raise ValueError("model_type must be 'logistic' or 'svm'")
    
    model.fit(X_scaled, y)
    
    # Create mesh grid for decision boundary
    x_min, x_max = X_scaled[:, 0].min() - 0.5, X_scaled[:, 0].max() + 0.5
    y_min, y_max = X_scaled[:, 1].min() - 0.5, X_scaled[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    # Predict on mesh grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Decision boundary with original data
    ax1.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
    scatter1 = ax1.scatter(X_scaled[y == 0, 0], X_scaled[y == 0, 1], 
                          c='green', label='OK Class', alpha=0.7, s=50)
    scatter2 = ax1.scatter(X_scaled[y == 1, 0], X_scaled[y == 1, 1], 
                          c='red', label='KO Class', alpha=0.7, s=50)
    ax1.set_xlabel(f'{feature1_name} (Standardized)')
    ax1.set_ylabel(f'{feature2_name} (Standardized)')
    ax1.set_title(f'Decision Boundary: {model_name}\n{feature1_name} vs {feature2_name}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Original scale data with decision boundary
    # Transform mesh back to original scale
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    mesh_original = scaler.inverse_transform(mesh_points)
    xx_orig = mesh_original[:, 0].reshape(xx.shape)
    yy_orig = mesh_original[:, 1].reshape(yy.shape)
    
    ax2.contourf(xx_orig, yy_orig, Z, alpha=0.4, cmap='RdYlBu')
    scatter3 = ax2.scatter(X[y == 0, 0], X[y == 0, 1], 
                          c='green', label='OK Class', alpha=0.7, s=50)
    scatter4 = ax2.scatter(X[y == 1, 0], X[y == 1, 1], 
                          c='red', label='KO Class', alpha=0.7, s=50)
    ax2.set_xlabel(feature1_name)
    ax2.set_ylabel(feature2_name)
    ax2.set_title(f'Decision Boundary (Original Scale)\n{feature1_name} vs {feature2_name}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Calculate model performance
    y_pred = model.predict(X_scaled)
    accuracy = np.mean(y_pred == y)
    
    # Get feature importance (for logistic regression)
    if model_type == "logistic":
        importance = np.abs(model.coef_[0])
        feature_importance = {
            feature1_name: importance[0],
            feature2_name: importance[1]
        }
    else:
        feature_importance = None
    
    return fig, accuracy, feature_importance

def plot_feature_importance_comparison(data, labels, feature_names, model_type="logistic"):
    """
    Plot feature importance comparison for all features using Logistic Regression
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt
    
    # Prepare data
    X = data.T  # Transpose to get (n_samples, n_features)
    y = labels
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train logistic regression
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_scaled, y)
    
    # Get feature importance
    importance = np.abs(model.coef_[0])
    
    # Create feature importance plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Bar plot of feature importance
    bars = ax1.bar(range(len(feature_names)), importance, 
                   color=['green' if i < 4 else 'blue' for i in range(len(feature_names))])
    ax1.set_xlabel('Features')
    ax1.set_ylabel('Feature Importance (|Coefficient|)')
    ax1.set_title('Feature Importance - Logistic Regression')
    ax1.set_xticks(range(len(feature_names)))
    ax1.set_xticklabels(feature_names, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{height:.3f}', ha='center', va='bottom')
    
    # Top features ranking
    feature_importance_pairs = list(zip(feature_names, importance))
    feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
    
    top_features = [f[0] for f in feature_importance_pairs[:5]]
    top_importance = [f[1] for f in feature_importance_pairs[:5]]
    
    bars2 = ax2.bar(range(len(top_features)), top_importance, color='orange', alpha=0.7)
    ax2.set_xlabel('Top Features')
    ax2.set_ylabel('Feature Importance')
    ax2.set_title('Top 5 Most Important Features')
    ax2.set_xticks(range(len(top_features)))
    ax2.set_xticklabels(top_features, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    return fig, feature_importance_pairs

