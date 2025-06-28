import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

def plot_histogram(D, L, feature_idx, class_label, bins=20, feature_names=None):
    """Plot histogram for a specific feature and class."""
    values = D[feature_idx, L == class_label]
    feature_label = f"{feature_idx}"
    if feature_names is not None and feature_idx < len(feature_names):
        feature_label = f"{feature_idx} - {feature_names[feature_idx]}"
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(values, bins=bins, alpha=0.7, color='blue' if class_label == 0 else 'red')
    ax.set_title(f"Histogram of Feature {feature_label} for Class {class_label}")
    ax.set_xlabel(f"Feature {feature_label} value")
    ax.set_ylabel("Frequency")
    ax.grid(True)
    return fig

def plot_feature_ranking(ranking, top_k=5, feature_names=None):
    """Plot bar chart of top-k ranked features by discriminative score."""
    features = [int(f[0]) for f in ranking[:top_k]]
    scores = [f[1] for f in ranking[:top_k]]
    if feature_names is not None:
        featuresLabels = [f"{i} - {feature_names[i]}" for i in features]
    else:
        featuresLabels = [str(i) for i in features]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(featuresLabels, scores, color='green')
    ax.set_title(f"Top {top_k} Discriminative Features")
    ax.set_xlabel("Feature (Number - Name)")
    ax.set_ylabel("Discriminative Score")
    ax.grid(True)
    return fig

def plot_time_series(D, L, feature_idx, class_label, feature_names=None):
    """Plot time series for a specific feature and class."""
    values = D[feature_idx, L == class_label]
    feature_label = f"{feature_idx}"
    if feature_names is not None and feature_idx < len(feature_names):
        feature_label = f"{feature_idx} - {feature_names[feature_idx]}"
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(values, marker='o', linestyle='-')
    ax.set_title(f"Time Series of Feature {feature_label} for Class {class_label}")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel(f"Feature {feature_label} Value")
    ax.grid(True)
    return fig

def plot_frequency_spectrum(D, L, feature_idx, class_label, feature_names=None):
    """Plot frequency spectrum (FFT) for a specific feature and class."""
    values = D[feature_idx, L == class_label]
    N = len(values)
    fft_vals = np.fft.fft(values)
    fft_freqs = np.fft.fftfreq(N)
    feature_label = f"{feature_idx}"
    if feature_names is not None and feature_idx < len(feature_names):
        feature_label = f"{feature_idx} - {feature_names[feature_idx]}"
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(fft_freqs[:N//2], np.abs(fft_vals)[:N//2])
    ax.set_title(f"Frequency Spectrum of Feature {feature_label} for Class {class_label}")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Magnitude")
    ax.grid(True)
    return fig

def plot_feature_statistics_comparison(D, L, feature_names=None, group_labels=("OK", "KO")):
    """
    Create comprehensive plots comparing different statistics between classes for each feature.
    
    Args:
        D: Data matrix (features x samples)
        L: Labels vector
        feature_names: List of feature names
        group_labels: Labels for the two groups
    """
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(D.shape[0])]
    
    n_features = D.shape[0]
    n_stats = 5  # mean, median, mode, std, var
    stats_names = ["Mean", "Median", "Mode", "Std", "Variance"]
    
    # Create subplots
    fig, axes = plt.subplots(n_stats, n_features, figsize=(4*n_features, 3*n_stats))
    if n_features == 1:
        axes = axes.reshape(-1, 1)
    if n_stats == 1:
        axes = axes.reshape(1, -1)
    
    # Colors for the two groups
    colors = ['skyblue', 'lightcoral']
    
    for feat_idx in range(n_features):
        group0_data = D[feat_idx, L == 0]
        group1_data = D[feat_idx, L == 1]
        
        # Calculate statistics for each group
        stats_group0 = [
            np.mean(group0_data),
            np.median(group0_data),
            stats.mode(group0_data, keepdims=False).mode,
            np.std(group0_data),
            np.var(group0_data)
        ]
        
        stats_group1 = [
            np.mean(group1_data),
            np.median(group1_data),
            stats.mode(group1_data, keepdims=False).mode,
            np.std(group1_data),
            np.var(group1_data)
        ]
        
        # Plot each statistic
        for stat_idx in range(n_stats):
            ax = axes[stat_idx, feat_idx]
            
            # Create bar plot
            x_pos = np.arange(2)
            bars = ax.bar(x_pos, [stats_group0[stat_idx], stats_group1[stat_idx]], 
                         color=colors, alpha=0.7)
            
            # Add value labels on bars
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{height:.2f}', ha='center', va='bottom', fontsize=8)
            
            # Customize plot
            ax.set_xticks(x_pos)
            ax.set_xticklabels(group_labels, rotation=45)
            ax.set_title(f'{stats_names[stat_idx]}', fontsize=10)
            
            # Add separation score
            diff = abs(stats_group0[stat_idx] - stats_group1[stat_idx])
            ax.text(0.5, 0.95, f'Diff: {diff:.3f}', 
                   transform=ax.transAxes, ha='center', va='top',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                   fontsize=8)
            
            # Only add ylabel for first column
            if feat_idx == 0:
                ax.set_ylabel(stats_names[stat_idx])
            
            # Only add xlabel for last row
            if stat_idx == n_stats - 1:
                ax.set_xlabel(feature_names[feat_idx])
    
    plt.tight_layout()
    return fig

def plot_statistic_separation_ranking(D, L, group_labels=("OK", "KO")):
    """
    Create a plot showing which statistics best separate the classes across all features.
    
    Args:
        D: Data matrix (features x samples)
        L: Labels vector
        group_labels: Labels for the two groups
    """
    from stats import rank_separating_statistics
    
    # Convert string group labels to numeric labels for stats functions
    # The stats functions expect (0, 1) but we might pass ("OK", "KO")
    numeric_group_labels = (0, 1)  # Always use numeric labels for stats functions
    
    # Get ranked statistics
    ranked_stats = rank_separating_statistics(D, L, numeric_group_labels)
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Bar plot of average differences
    stats_names = [stat.replace('_diff', '').title() for stat, _ in ranked_stats]
    avg_diffs = [diff for _, diff in ranked_stats]
    
    bars = ax1.bar(stats_names, avg_diffs, color='lightblue', alpha=0.7)
    ax1.set_title('Average Statistical Separation Across Features')
    ax1.set_ylabel('Average Absolute Difference')
    ax1.set_xlabel('Statistic')
    
    # Add value labels on bars
    for bar, diff in zip(bars, avg_diffs):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{diff:.3f}', ha='center', va='bottom')
    
    # Pie chart showing relative importance
    ax2.pie(avg_diffs, labels=stats_names, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Relative Importance of Statistics for Class Separation')
    
    plt.tight_layout()
    return fig

def plot_feature_p_values_comparison(D, L, feature_names=None):
    """
    Create plots comparing p-values for different statistical tests across features.
    Only T-test is shown.
    """
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(D.shape[0])]

    from stats import feature_p_values

    # Get p-values for t-test only
    pvals_ttest = feature_p_values(D, L, test="ttest")

    # Create the plot
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))

    # T-test p-values
    feature_indices = [feat for feat, _ in pvals_ttest]
    ttest_pvals = [pval for _, pval in pvals_ttest]

    bars1 = ax1.bar(range(len(feature_indices)), ttest_pvals, color='lightgreen', alpha=0.7)
    ax1.set_title('T-Test P-Values by Feature')
    ax1.set_ylabel('P-Value')
    ax1.set_xlabel('Feature Index')
    ax1.set_xticks(range(len(feature_indices)))
    ax1.set_xticklabels([feature_names[i] for i in feature_indices], rotation=45)

    # Add significance line
    ax1.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='α=0.05')
    ax1.legend()

    plt.tight_layout()
    return fig

def plot_statistic_discriminative_power(D, L, group_labels=("OK", "KO")):
    """
    Create a plot showing the discriminative power of different statistics (T-Test only).
    """
    from stats import statistic_discriminative_power

    numeric_group_labels = (0, 1)

    # Get discriminative power for t-test only
    discriminative_ttest = statistic_discriminative_power(D, L, numeric_group_labels, test="ttest")

    # Create the plot
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))

    # T-test results
    stats_names = [stat.title() for stat, _ in discriminative_ttest]
    ttest_pvals = [pval for _, pval in discriminative_ttest]

    bars1 = ax1.bar(stats_names, ttest_pvals, color='lightblue', alpha=0.7)
    ax1.set_title('Discriminative Power - T-Test')
    ax1.set_ylabel('P-Value')
    ax1.set_xlabel('Statistic')

    # Add value labels and significance
    for bar, pval in zip(bars1, ttest_pvals):
        height = bar.get_height()
        color = 'red' if pval < 0.05 else 'black'
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{pval:.3f}', ha='center', va='bottom', color=color, fontweight='bold')

    ax1.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='α=0.05')
    ax1.legend()

    plt.tight_layout()
    return fig

def plot_feature_distributions_by_statistic(D, L, feature_names=None, group_labels=("OK", "KO")):
    """
    Create distribution plots for each feature showing how well different statistics separate the classes.
    
    Args:
        D: Data matrix (features x samples)
        L: Labels vector
        feature_names: List of feature names
        group_labels: Labels for the two groups
    """
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(D.shape[0])]
    
    n_features = D.shape[0]
    fig, axes = plt.subplots(1, n_features, figsize=(4*n_features, 4))
    if n_features == 1:
        axes = axes.reshape(1, 1)
    
    for feat_idx in range(n_features):
        group0_data = D[feat_idx, L == 0]
        group1_data = D[feat_idx, L == 1]
        
        # Histogram only
        ax = axes[feat_idx]
        ax.hist(group0_data, bins=20, alpha=0.6, label=group_labels[0], color='skyblue')
        ax.hist(group1_data, bins=20, alpha=0.6, label=group_labels[1], color='lightcoral')
        ax.set_title(f'{feature_names[feat_idx]} - Histogram')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.legend()
        
        # Add statistical summary
        mean_diff = abs(np.mean(group0_data) - np.mean(group1_data))
        ax.text(0.5, 0.95, f'Mean Diff: {mean_diff:.3f}', 
                transform=ax.transAxes, ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    return fig

def create_comprehensive_statistical_analysis(D, L, feature_names=None, group_labels=("OK", "KO")):
    """
    Create a comprehensive set of plots for statistical analysis.
    
    Args:
        D: Data matrix (features x samples)
        L: Labels vector
        feature_names: List of feature names
        group_labels: Labels for the two groups
    """
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(D.shape[0])]
    
    # Create all plots
    fig1 = plot_feature_statistics_comparison(D, L, feature_names, group_labels)
    fig2 = plot_statistic_separation_ranking(D, L, group_labels)
    fig3 = plot_feature_p_values_comparison(D, L, feature_names)
    fig4 = plot_statistic_discriminative_power(D, L, group_labels)
    fig5 = plot_feature_distributions_by_statistic(D, L, feature_names, group_labels)
    
    return {
        'statistics_comparison': fig1,
        'separation_ranking': fig2,
        'p_values_comparison': fig3,
        'discriminative_power': fig4,
        'feature_distributions': fig5
    }

def main():
    """Main function to demonstrate the plotting capabilities."""
    import utils
    
    # Load data
    D, L = utils.load_data("diabetes_short.csv")
    (DTR, LTR), (DVAL, LVAL) = utils.split_db_2to1(D, L)
    
    # Get feature names
    feature_names = utils.get_feature_names("diabetes_short.csv")
    
    # Create comprehensive analysis
    plots = create_comprehensive_statistical_analysis(DTR, LTR, feature_names)
    
    # Display plots
    for plot_name, fig in plots.items():
        plt.figure(fig.number)
        plt.show()
        print(f"Displayed: {plot_name}")

if __name__ == "__main__":
    main()

