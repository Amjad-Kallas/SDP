import streamlit as st
import utils
import stats
import matplotlib.pyplot as plt

st.title("Diabetes Feature Analysis App")

st.sidebar.header("Actions")
action = st.sidebar.selectbox(
    "Choose an action:",
    [
        "Show dataset info",
        "Show feature importances (Logistic Regression)",
        "Show feature importances (SVM)",
        "Show statistical separation",
        "Plot histogram (feature)",
        "Plot boxplot (feature)",
        # Add more actions as needed
    ]
)

D, L = utils.load_data("diabetes.csv")
(DTR, LTR), (DVAL, LVAL) = utils.split_db_2to1(D, L)

feature_names = [
    'Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness',
    'Insulin', 'BMI', 'DPF', 'Age'
]

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

elif action == "Plot boxplot (feature)":
    feature_idx = st.selectbox("Select feature:", list(enumerate(feature_names)), format_func=lambda x: x[1])[0]
    fig, ax = plt.subplots()
    ax.boxplot([DTR[feature_idx, LTR == 0], DTR[feature_idx, LTR == 1]], labels=["OK", "KO"])
    ax.set_title(f"Boxplot of {feature_names[feature_idx]}")
    st.pyplot(fig)

# --- Natural Language Plotting Section ---
st.subheader("AI Agent: Natural Language Plotting")
user_cmd = st.text_input("Describe your plot or analysis request:", "plot histogram of Glucose for KO")

if user_cmd:
    import naturalLanguage
    parsed = naturalLanguage.interpret_plot_command(user_cmd, feature_names)
    if parsed:
        plot_type, feature_idx, group = parsed
        fig, ax = plt.subplots()
        if plot_type == "histogram":
            if group == "all":
                ax.hist(DTR[feature_idx, :], bins=20, alpha=0.7, label="All")
            else:
                label_val = 0 if group.lower() == "ok" else 1
                ax.hist(DTR[feature_idx, LTR == label_val], bins=20, alpha=0.7, label=group.upper())
            ax.set_title(f"Histogram of {feature_names[feature_idx]} ({group.upper()})")
            ax.legend()
            st.pyplot(fig)
        elif plot_type == "boxplot":
            if group == "all":
                ax.boxplot(DTR[feature_idx, :], labels=[feature_names[feature_idx]])
            else:
                label_val = 0 if group.lower() == "ok" else 1
                ax.boxplot(DTR[feature_idx, LTR == label_val], labels=[f"{feature_names[feature_idx]} ({group.upper()})"])
            ax.set_title(f"Boxplot of {feature_names[feature_idx]} ({group.upper()})")
            st.pyplot(fig)
        # Add more plot types as needed
    else:
        st.warning("Could not interpret your request. Try: 'plot histogram of Glucose for KO'.")
