import streamlit as st
import utils
import stats
import plot
import naturalLanguage
import numpy as np

st.set_page_config(page_title="Diabetes Analysis with AI", layout="wide")

st.title("🤖 Diabetes Feature Analysis with AI Assistant")

# Load data
@st.cache_data
def load_cached_data():
    D, L = utils.load_data("diabetes.csv")
    (DTR, LTR), (DVAL, LVAL) = utils.split_db_2to1(D, L)
    return DTR, LTR, DVAL, LVAL

DTR, LTR, DVAL, LVAL = load_cached_data()

feature_names = [
    'Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness',
    'Insulin', 'BMI', 'DPF', 'Age'
]

# Main AI Assistant Section
st.markdown("---")
st.subheader("🤖 AI Assistant: Natural Language Analysis")

# Check if Llama-2 is available
llama_status = "🟢 Available" if naturalLanguage.LLAMA_AVAILABLE else "🔴 Not Available"
st.info(f"**Llama-2 Status:** {llama_status}")

# Example queries
st.markdown("**💡 Example Queries:**")
example_queries = [
    "show BMI feature distribution",
    "show BMI feature statistics",
    "plot decision boundary between Glucose and BMI",
    "plot histogram of Glucose for KO",
    "compare BMI distributions",
    "display Blood Pressure distribution for OK class",
    "plot boxplot of Age for all classes",
    "t-test BMI mean",
    "t-test Glucose median",
    "t-test Blood Pressure variance"
]

# Display example queries as clickable buttons
cols = st.columns(3)
for i, query in enumerate(example_queries):
    with cols[i % 3]:
        if st.button(f"💬 {query}", key=f"example_{i}"):
            st.session_state.user_query = query

# Initialize session state once
if "user_query" not in st.session_state:
    st.session_state.user_query = ""

# User input with session state key
st.text_input(
    "🎯 Describe your analysis request:",
    placeholder="e.g., 'show BMI feature distribution' or 'plot histogram of Glucose for KO'",
    key="user_query"
)

if st.session_state.user_query:
    with st.spinner("🤖 AI is processing your request..."):
        parsed = naturalLanguage.interpret_plot_command_with_llama(st.session_state.user_query, feature_names)
        
        if parsed:
            if parsed[0] == "error":
                st.error(parsed[1])
                st.stop()
                
            if len(parsed) == 4:  # Llama-2 response
                plot_type, feature_idx, class_filter, additional_params = parsed
            elif len(parsed) == 5:  # T-test response
                plot_type, feature_idx, class_filter, additional_params, stat_measure = parsed
            else:  # Regex fallback
                plot_type, feature_idx, class_filter = parsed
                additional_params = {}
                stat_measure = "mean"  # default for t-tests
            
            # Handle decision boundary requests differently
            if plot_type == "decision_boundary":
                if len(parsed) >= 3 and isinstance(parsed[1], int) and isinstance(parsed[2], int):
                    feature1_idx, feature2_idx = parsed[1], parsed[2]
                    st.success(f"✅ AI understood: Decision boundary analysis between {feature_names[feature1_idx]} and {feature_names[feature2_idx]}")
                else:
                    st.warning("Decision boundary analysis requires two features.")
            elif plot_type == "t_test":
                st.success(f"✅ AI understood: T-test for {stat_measure} of {feature_names[feature_idx]}")
            else:
                st.success(f"✅ AI understood: {naturalLanguage.generate_plot_description(plot_type, feature_names[feature_idx], class_filter)}")
            
            # Generate the appropriate plot
            if plot_type == "histogram":
                fig = plot.plot_histogram(DTR, LTR, feature_idx, feature_names[feature_idx], class_filter)
                st.pyplot(fig)
                
            elif plot_type == "boxplot":
                fig = plot.plot_boxplot(DTR, LTR, feature_idx, feature_names[feature_idx], class_filter)
                st.pyplot(fig)
                
            elif plot_type == "statistics":
                # Display simplified statistics
                st.subheader(f"📊 Statistics for {feature_names[feature_idx]}")
                
                # Get statistics data
                stats_data = plot.display_detailed_statistics(DTR, LTR, feature_idx, feature_names[feature_idx])
                
                # Display in columns
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🟢 OK Class Statistics")
                    ok_stats = stats_data['ok_stats']
                    st.metric("Count", ok_stats['count'])
                    st.metric("Mean", f"{ok_stats['mean']:.2f}")
                    st.metric("Median", f"{ok_stats['median']:.2f}")
                    st.metric("Mode", f"{ok_stats['mode']:.2f}")
                    st.metric("Standard Deviation", f"{ok_stats['std']:.2f}")
                    st.metric("Variance", f"{ok_stats['variance']:.2f}")
                
                with col2:
                    st.subheader("🔴 KO Class Statistics")
                    ko_stats = stats_data['ko_stats']
                    st.metric("Count", ko_stats['count'])
                    st.metric("Mean", f"{ko_stats['mean']:.2f}")
                    st.metric("Median", f"{ko_stats['median']:.2f}")
                    st.metric("Mode", f"{ko_stats['mode']:.2f}")
                    st.metric("Standard Deviation", f"{ko_stats['std']:.2f}")
                    st.metric("Variance", f"{ko_stats['variance']:.2f}")
                
                # T-test only
                st.subheader("🔬 T-Test Results")
                t_test = stats_data['t_test']
                col3, col4 = st.columns(2)
                
                with col3:
                    st.metric("T-statistic", f"{t_test['statistic']:.4f}")
                    st.metric("P-value", f"{t_test['p_value']:.4f}")
                
                with col4:
                    if t_test['p_value'] < 0.05:
                        st.success("✅ Significant difference between classes")
                    else:
                        st.warning("❌ No significant difference between classes")
                
                # Visual statistics summary
                st.subheader("📈 Visual Statistics Summary")
                fig = plot.plot_statistics_summary(DTR, LTR, feature_idx, feature_names[feature_idx])
                st.pyplot(fig)
                
            elif plot_type == "t_test":
                # Display t-test for specific statistical measure
                st.subheader(f"🔬 T-Test for {stat_measure.title()} of {feature_names[feature_idx]}")
                
                # Get data for both classes
                ok_data = DTR[feature_idx, LTR == 0]
                ko_data = DTR[feature_idx, LTR == 1]
                
                # Calculate the specific statistical measure
                from scipy import stats
                
                if stat_measure == "mean":
                    ok_stat = np.mean(ok_data)
                    ko_stat = np.mean(ko_data)
                elif stat_measure == "median":
                    ok_stat = np.median(ok_data)
                    ko_stat = np.median(ko_data)
                elif stat_measure == "mode":
                    ok_stat = stats.mode(ok_data, keepdims=True)[0][0]
                    ko_stat = stats.mode(ko_data, keepdims=True)[0][0]
                elif stat_measure == "variance":
                    ok_stat = np.var(ok_data)
                    ko_stat = np.var(ko_data)
                elif stat_measure == "std":
                    ok_stat = np.std(ok_data)
                    ko_stat = np.std(ko_data)
                else:
                    ok_stat = np.mean(ok_data)
                    ko_stat = np.mean(ko_data)
                
                # Perform t-test
                t_stat, p_value = stats.ttest_ind(ok_data, ko_data)
                
                # Display results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.subheader("🟢 OK Class")
                    st.metric(f"{stat_measure.title()}", f"{ok_stat:.4f}")
                    st.metric("Count", len(ok_data))
                
                with col2:
                    st.subheader("🔴 KO Class")
                    st.metric(f"{stat_measure.title()}", f"{ko_stat:.4f}")
                    st.metric("Count", len(ko_data))
                
                with col3:
                    st.subheader("📊 T-Test Results")
                    st.metric("T-statistic", f"{t_stat:.4f}")
                    st.metric("P-value", f"{p_value:.4f}")
                    if p_value < 0.05:
                        st.success("✅ Significant difference")
                    else:
                        st.warning("❌ No significant difference")
                
                # Show difference
                diff = abs(ok_stat - ko_stat)
                st.metric(f"Difference in {stat_measure}", f"{diff:.4f}")
                
            elif plot_type == "scatter":
                st.info("Scatter plot functionality is not available yet. Please try another type of analysis or visualization.")
                
            elif plot_type == "decision_boundary":
                # Feature indices are already extracted above
                st.subheader(f"🎯 Decision Boundary Analysis")
                st.write(f"**Features:** {feature_names[feature1_idx]} vs {feature_names[feature2_idx]}")
                
                # Model selection
                model_type = st.selectbox("Select Model:", ["logistic", "svm"], 
                                        format_func=lambda x: "Logistic Regression" if x == "logistic" else "SVM (Linear)")
                
                # Generate decision boundary plot
                fig, accuracy, feature_importance = plot.plot_decision_boundary(
                    DTR, LTR, feature1_idx, feature2_idx, 
                    feature_names[feature1_idx], feature_names[feature2_idx], 
                    model_type
                )
                st.pyplot(fig)
                
                # Display model performance
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Model Accuracy", f"{accuracy:.3f}")
                
                with col2:
                    if feature_importance:
                        st.subheader("Feature Importance")
                        for feature, importance in feature_importance.items():
                            st.metric(feature, f"{importance:.3f}")
                
                # Add feature importance comparison for all features
                st.subheader("📊 Overall Feature Importance")
                fig_importance, importance_pairs = plot.plot_feature_importance_comparison(
                    DTR, LTR, feature_names, model_type
                )
                st.pyplot(fig_importance)
                
                # Display top features
                st.subheader("🏆 Top Discriminative Features")
                cols = st.columns(3)
                for i, (feature, importance) in enumerate(importance_pairs[:6]):
                    with cols[i % 3]:
                        st.metric(feature, f"{importance:.3f}", delta=f"Rank {i+1}")
            
            elif plot_type == "distribution":
                fig = plot.plot_histogram(DTR, LTR, feature_idx, feature_names[feature_idx], class_filter)
                st.pyplot(fig)
                
            elif plot_type == "feature_ranking":
                top_k = feature_idx  # In this case, feature_idx contains the top_k value
                separation = stats.rank_separating_statistics(DTR, LTR)
                st.subheader(f"🏆 Top {top_k} Feature Rankings")
                
                # Display rankings
                for i, (stat, diff) in enumerate(separation[:top_k]):
                    st.metric(f"{feature_names[stat]}", f"{diff:.4f}", delta=f"Rank {i+1}")
        
        else:
            st.error("❌ Sorry, I couldn't understand your request. Please try one of the example queries or rephrase your request.")
            
            # Show suggestions
            st.markdown("**💡 Try these formats:**")
            st.markdown("- `show [feature] feature distribution`")
            st.markdown("- `plot histogram of [feature] for [OK/KO/all]`")
            st.markdown("- `compare [feature] distributions`")
            st.markdown("- `t-test [feature] [mean/median/mode/variance/std]`")

# Footer
st.markdown("---")
st.markdown("*Powered by Llama-2 AI Assistant* 🚀")
