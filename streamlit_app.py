import streamlit as st
import utils
import stats
import plot
import matplotlib.pyplot as plt

# Helper functions for custom analysis with feature selection
def create_feature_statistics_comparison(D, L, selected_feature_names, all_feature_names):
    """Create statistics comparison plot with only selected features."""
    import numpy as np
    from scipy import stats
    
    # Get indices of selected features
    selected_indices = [all_feature_names.index(name) for name in selected_feature_names]
    
    n_features = len(selected_indices)
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
    
    for feat_idx, original_idx in enumerate(selected_indices):
        group0_data = D[original_idx, L == 0]
        group1_data = D[original_idx, L == 1]
        
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
            ax.set_xticklabels(["OK", "KO"], rotation=45)
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
                ax.set_xlabel(selected_feature_names[feat_idx])
    
    plt.tight_layout()
    return fig

def create_feature_p_values_comparison(D, L, selected_feature_names, all_feature_names):
    """Create p-values comparison plot with only selected features."""
    from stats import feature_p_values
    
    # Get p-values for all features first
    all_pvals = feature_p_values(D, L, test="ttest")
    
    # Filter to only selected features
    selected_indices = [all_feature_names.index(name) for name in selected_feature_names]
    filtered_pvals = [(feat, pval) for feat, pval in all_pvals if feat in selected_indices]
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(max(8, len(selected_feature_names)*2), 6))
    
    # P-values
    feature_indices = [feat for feat, _ in filtered_pvals]
    p_values = [pval for _, pval in filtered_pvals]
    
    bars = ax.bar(range(len(feature_indices)), p_values, color='lightgreen', alpha=0.7)
    ax.set_title('T-Test P-Values by Feature')
    ax.set_ylabel('P-Value')
    ax.set_xlabel('Feature Index')
    ax.set_xticks(range(len(feature_indices)))
    ax.set_xticklabels([selected_feature_names[selected_indices.index(i)] for i in feature_indices], rotation=45)
    
    # Add significance line
    ax.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='α=0.05')
    ax.legend()
    
    plt.tight_layout()
    return fig

def create_feature_distributions(D, L, selected_feature_names, all_feature_names):
    """Create distribution plots with only selected features."""
    import numpy as np
    
    # Get indices of selected features
    selected_indices = [all_feature_names.index(name) for name in selected_feature_names]
    
    n_features = len(selected_indices)
    fig, axes = plt.subplots(1, n_features, figsize=(4*n_features, 4))
    if n_features == 1:
        axes = [axes]  # Convert single axes to list for consistent indexing
    
    for feat_idx, original_idx in enumerate(selected_indices):
        group0_data = D[original_idx, L == 0]
        group1_data = D[original_idx, L == 1]
        
        # Histogram only
        ax = axes[feat_idx]
        ax.hist(group0_data, bins=20, alpha=0.6, label="OK", color='skyblue')
        ax.hist(group1_data, bins=20, alpha=0.6, label="KO", color='lightcoral')
        ax.set_title(f'{selected_feature_names[feat_idx]} - Histogram')
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

def create_comprehensive_analysis(D, L, selected_feature_names, all_feature_names):
    """Create comprehensive analysis with selected features."""
    plots = {}
    
    # Create custom plots
    plots['statistics_comparison'] = create_feature_statistics_comparison(D, L, selected_feature_names, all_feature_names)
    plots['p_values_comparison'] = create_feature_p_values_comparison(D, L, selected_feature_names, all_feature_names)
    plots['feature_distributions'] = create_feature_distributions(D, L, selected_feature_names, all_feature_names)
    
    # These plots are for all features (not feature-specific)
    plots['separation_ranking'] = plot.plot_statistic_separation_ranking(D, L)
    plots['discriminative_power'] = plot.plot_statistic_discriminative_power(D, L)
    
    return plots

st.title("Diabetes Feature Analysis App")

# Feature selection for statistical analysis - Move to top level
st.sidebar.header("Feature Selection")
show_all = st.sidebar.checkbox("Show all features", value=True)

D, L = utils.load_data("diabetes.csv")
(DTR, LTR), (DVAL, LVAL) = utils.split_db_2to1(D, L)

feature_names = [
    'Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness',
    'Insulin', 'BMI', 'DPF', 'Age'
]

# Get selected features based on sidebar selection
if show_all:
    selected_feature_names = feature_names
else:
    selected_features = st.sidebar.multiselect(
        "Select features to analyze:",
        list(enumerate(feature_names)),
        format_func=lambda x: x[1]
    )
    selected_feature_names = [feature_names[idx] for idx, _ in selected_features]

# Move action selection to main page
action = st.selectbox(
    "Choose an action:",
    [
        "Show dataset info",
        "Show feature importances (Logistic Regression)",
        "Show feature importances (SVM)",
        "Show statistical separation",
        "Plot histogram (feature)",
        "Statistical Analysis: Feature Statistics Comparison",
        "Statistical Analysis: Separation Ranking",
        "Statistical Analysis: P-Values Comparison",
        "Statistical Analysis: Discriminative Power",
        "Statistical Analysis: Feature Distributions",
        "Comprehensive Statistical Analysis",
        # Add more actions as needed
    ]
)

if action == "Show dataset info":
    st.subheader("Dataset Info")
    st.write(f"Training samples: {DTR.shape[1]}")
    st.write(f"Validation samples: {DVAL.shape[1]}")
    st.write(f"Features: {feature_names}")

elif action == "Show feature importances (Logistic Regression)":
    from model import train_logistic_regression_sklearn
    w_sklearn, b_sklearn = train_logistic_regression_sklearn(DTR, LTR)
    importances = abs(w_sklearn.flatten())
    st.subheader("Logistic Regression Feature Importances")
    for i, imp in sorted(enumerate(importances), key=lambda x: -x[1]):
        st.write(f"{feature_names[i]}: {imp:.4f}")

elif action == "Show feature importances (SVM)":
    from model import train_svm
    svm_model = train_svm(DTR, LTR, C=1.0, kernel='linear')
    if hasattr(svm_model, 'coef_'):
        importances = abs(svm_model.coef_.flatten())
        st.subheader("SVM Feature Importances")
        for i, imp in sorted(enumerate(importances), key=lambda x: -x[1]):
            st.write(f"{feature_names[i]}: {imp:.4f}")
    else:
        st.write("SVM model does not provide feature importances for this kernel.")

elif action == "Show statistical separation":
    separation = stats.rank_separating_statistics(DTR, LTR)
    st.subheader("Statistical Separation (by average difference)")
    for stat, diff in separation:
        st.write(f"{stat}: {diff:.4f}")

elif action == "Plot histogram (feature)":
    feature_idx = st.selectbox("Select feature:", list(enumerate(feature_names)), format_func=lambda x: x[1])[0]
    fig, ax = plt.subplots()
    ax.hist(DTR[feature_idx, LTR == 0], bins=20, alpha=0.5, label="OK")
    ax.hist(DTR[feature_idx, LTR == 1], bins=20, alpha=0.5, label="KO")
    ax.set_title(f"Histogram of {feature_names[feature_idx]}")
    ax.legend()
    st.pyplot(fig)

# New Statistical Analysis Options
elif action == "Statistical Analysis: Feature Statistics Comparison":
    st.subheader("Feature Statistics Comparison")
    
    if not selected_feature_names:
        st.warning("Please select at least one feature to analyze.")
    else:
        st.write(f"This plot shows how different statistics (mean, median, mode, std, variance) compare between OK and KO classes for selected features: {', '.join(selected_feature_names)}.")
        
        # Create custom statistics comparison for selected features
        fig = create_feature_statistics_comparison(DTR, LTR, selected_feature_names, feature_names)
        st.pyplot(fig)
        
        # Add explanation
        st.markdown("""
        **Interpretation:**
        - Each row represents a different statistic (Mean, Median, Mode, Std, Variance)
        - Each column represents a different feature
        - The yellow box shows the absolute difference between classes
        - Larger differences indicate better class separation
        """)

elif action == "Statistical Analysis: Separation Ranking":
    st.subheader("Statistical Separation Ranking")
    st.write("This plot shows which statistics best separate the classes across all features.")
    fig = plot.plot_statistic_separation_ranking(DTR, LTR)
    st.pyplot(fig)
    
    # Add explanation
    st.markdown("""
    **Interpretation:**
    - The bar chart shows average absolute differences across all features
    - The pie chart shows relative importance of each statistic
    - Higher values indicate better class separation
    - This helps identify which statistical measures are most discriminative
    """)

elif action == "Statistical Analysis: P-Values Comparison":
    st.subheader("P-Values Comparison")
    
    if not selected_feature_names:
        st.warning("Please select at least one feature to analyze.")
    else:
        st.write(f"This plot shows statistical significance of differences between classes for selected features: {', '.join(selected_feature_names)}.")
        
        # Create custom p-values comparison for selected features
        fig = create_feature_p_values_comparison(DTR, LTR, selected_feature_names, feature_names)
        st.pyplot(fig)
        
        # Add explanation
        st.markdown("""
        **Interpretation:**
        - Lower p-values indicate more significant differences between classes
        - The red dashed line shows the α=0.05 significance threshold
        - T-test assumes normal distribution, Mann-Whitney is non-parametric
        - Features with p < 0.05 are statistically significantly different
        """)

elif action == "Statistical Analysis: Discriminative Power":
    st.subheader("Discriminative Power of Statistics")
    st.write("This plot shows how well each statistic discriminates between classes across all features.")
    fig = plot.plot_statistic_discriminative_power(DTR, LTR)
    st.pyplot(fig)
    
    # Add explanation
    st.markdown("""
    **Interpretation:**
    - Shows p-values for each statistic across all features
    - Red labels indicate statistically significant differences (p < 0.05)
    - Lower p-values indicate better discriminative power
    - Helps identify which statistics are most useful for classification
    """)

elif action == "Statistical Analysis: Feature Distributions":
    st.subheader("Feature Distributions by Class")
    
    if not selected_feature_names:
        st.warning("Please select at least one feature to analyze.")
    else:
        st.write(f"This plot shows the distribution of selected features for both classes: {', '.join(selected_feature_names)}.")
        
        # Create custom feature distributions for selected features
        fig = create_feature_distributions(DTR, LTR, selected_feature_names, feature_names)
        st.pyplot(fig)
        
        # Add explanation
        st.markdown("""
        **Interpretation:**
        - Histograms showing frequency distributions for each feature
        - Yellow boxes show mean differences between classes
        - Clear separation in distributions indicates good discriminative features
        """)

elif action == "Comprehensive Statistical Analysis":
    st.subheader("Comprehensive Statistical Analysis")
    
    if not selected_feature_names:
        st.warning("Please select at least one feature to analyze.")
    else:
        st.write(f"Generating all statistical analysis plots for selected features: {', '.join(selected_feature_names)}...")
        
        with st.spinner("Creating comprehensive analysis..."):
            plots = create_comprehensive_analysis(DTR, LTR, selected_feature_names, feature_names)
        
        # Display all plots with explanations
        for plot_name, fig in plots.items():
            st.write(f"### {plot_name.replace('_', ' ').title()}")
            st.pyplot(fig)
            
            # Add specific explanations for each plot type
            if plot_name == "statistics_comparison":
                st.markdown("**Feature Statistics Comparison:** Shows how different statistics compare between classes for each feature.")
            elif plot_name == "separation_ranking":
                st.markdown("**Separation Ranking:** Shows which statistics best separate classes across all features.")
            elif plot_name == "p_values_comparison":
                st.markdown("**P-Values Comparison:** Shows statistical significance of class differences for each feature.")
            elif plot_name == "discriminative_power":
                st.markdown("**Discriminative Power:** Shows how well each statistic discriminates between classes.")
            elif plot_name == "feature_distributions":
                st.markdown("**Feature Distributions:** Shows distribution of each feature for both classes.")
            
            st.markdown("---")

# --- Natural Language Plotting Section ---
st.subheader("AI Agent: Natural Language Plotting")
user_cmd = st.text_input("Describe your plot or analysis request:", "show statistical separation ranking")

if user_cmd:
    import naturalLanguage
    parsed = naturalLanguage.interpret_plot_command(user_cmd, feature_names)
    if parsed:
        plot_type, feature_idx, group = parsed
        
        # Handle statistical analysis plots
        if plot_type == "statistical_separation_ranking":
            st.write("### Statistical Separation Ranking")
            fig = plot.plot_statistic_separation_ranking(DTR, LTR)
            st.pyplot(fig)
            st.markdown("**Interpretation:** Shows which statistics best separate the classes across all features.")
            
        elif plot_type == "discriminative_power":
            st.write("### Discriminative Power of Statistics")
            fig = plot.plot_statistic_discriminative_power(DTR, LTR)
            st.pyplot(fig)
            st.markdown("**Interpretation:** Shows how well each statistic discriminates between classes.")
            
        elif plot_type == "p_values_comparison":
            st.write("### P-Values Comparison")
            if feature_idx is not None:
                # Single feature p-values comparison
                st.write(f"### P-Values for {feature_names[feature_idx]}")
                fig = create_feature_p_values_comparison(DTR, LTR, [feature_names[feature_idx]], feature_names)
                st.pyplot(fig)
            else:
                # All features p-values comparison
                fig = plot.plot_feature_p_values_comparison(DTR, LTR, feature_names)
                st.pyplot(fig)
            st.markdown("**Interpretation:** Shows statistical significance of differences between classes.")
            
        elif plot_type == "feature_statistics_comparison":
            st.write("### Feature Statistics Comparison")
            if feature_idx is not None:
                # Single feature statistics comparison
                st.write(f"### Statistics Comparison for {feature_names[feature_idx]}")
                fig = create_feature_statistics_comparison(DTR, LTR, [feature_names[feature_idx]], feature_names)
                st.pyplot(fig)
            else:
                # All features statistics comparison
                fig = plot.plot_feature_statistics_comparison(DTR, LTR, feature_names)
                st.pyplot(fig)
            st.markdown("**Interpretation:** Shows how different statistics compare between classes for each feature.")
            
        elif plot_type == "feature_distributions":
            st.write("### Feature Distributions by Class")
            if feature_idx is not None:
                # Single feature distribution
                st.write(f"### Distribution for {feature_names[feature_idx]}")
                fig = create_feature_distributions(DTR, LTR, [feature_names[feature_idx]], feature_names)
                st.pyplot(fig)
            else:
                # All features distributions
                fig = plot.plot_feature_distributions_by_statistic(DTR, LTR, feature_names)
                st.pyplot(fig)
            st.markdown("**Interpretation:** Shows the distribution of each feature for both classes.")
            
        elif plot_type == "comprehensive_analysis":
            st.write("### Comprehensive Statistical Analysis")
            if feature_idx is not None:
                # Single feature comprehensive analysis
                st.write(f"### Comprehensive Analysis for {feature_names[feature_idx]}")
                with st.spinner("Creating comprehensive analysis..."):
                    plots = create_comprehensive_analysis(DTR, LTR, [feature_names[feature_idx]], feature_names)
            else:
                # All features comprehensive analysis
                with st.spinner("Creating comprehensive analysis..."):
                    plots = plot.create_comprehensive_statistical_analysis(DTR, LTR, feature_names)
            
            for plot_name, fig in plots.items():
                st.write(f"#### {plot_name.replace('_', ' ').title()}")
                st.pyplot(fig)
                st.markdown("---")
        
        # Handle traditional plots
        elif plot_type == "histogram":
            fig, ax = plt.subplots()
            if group == "all":
                ax.hist(DTR[feature_idx, :], bins=20, alpha=0.7, label="All")
            else:
                label_val = 0 if group.lower() == "ok" else 1
                ax.hist(DTR[feature_idx, LTR == label_val], bins=20, alpha=0.7, label=group.upper())
            ax.set_title(f"Histogram of {feature_names[feature_idx]} ({group.upper()})")
            ax.legend()
            st.pyplot(fig)
        # Add more plot types as needed
    else:
        st.warning("Could not interpret your request. Try: 'show statistical separation ranking', 'show glucose distribution', 'show feature glucose statistics', or 'show feature distribution for glucose'.")
