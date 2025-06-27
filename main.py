import utils
import stats
import model
import plot
import naturalLanguage

from llama_cpp import Llama


def main():
    D, L = utils.load_data("diabetes_short.csv")

    (DTR, LTR), (DVAL, LVAL) = utils.split_db_2to1(D, L)

    stats_dict = stats.compute_statistics(DTR, LTR)
    # print(stats_dict)
    ranking = stats.rank_features_by_separation(stats_dict)
    # print(ranking) # sorted by mean_diff (for each feature)

    w, b = model.train_logistic_regression(DTR, LTR)
    print(w, b)

    # Example: Logistic Regression using scikit-learn
    w_sklearn, b_sklearn = model.train_logistic_regression_sklearn(DTR, LTR)
    print("[sklearn Logistic Regression] Weights:", w_sklearn)
    print("[sklearn Logistic Regression] Bias:", b_sklearn)

    # Example: SVM using scikit-learn
    svm_model = model.train_svm(DTR, LTR, C=1.0, kernel='linear')
    # Predict on validation set
    DVAL_T = DVAL.T
    svm_preds = svm_model.predict(DVAL_T)
    print("[SVM] Validation predictions:", svm_preds)


    # Create a more informative prompt with the ranking data
    feature_names = {
        1: 'Pregnancies',
        2: 'Glucose',
        3: 'Blood Pressure',
        4: 'Skin Thickness',
        5: 'Insulin',
        6: 'BMI',
        7: 'DPF',
        8: 'Age'
    }
    
    # ranking_text = "\n".join([f"{feature_names[int(f[0])]}: {f[1]:.3f}" for f in ranking[:5]])
    # prompt = f"""[INST] Based on the following feature rankings from a diabetes dataset, which feature appears to be the most discriminative and why?

    # Feature Rankings (top 5):
    # {ranking_text}

    # Please analyze which feature is most discriminative and explain why. [/INST]"""

    # llm = Llama(model_path="llama-2-7b-chat.Q3_K_S.gguf")
    # response = llm(prompt, max_tokens=256)
    # print("\nLlama's Analysis:")
    # print(response["choices"][0]["text"])




    # for the moment, we use a search algorithm to find the most relevant plot command
    # and then we plot the corresponding plot

    # cmd = "plot feature ranking top 3"
    # parsed = naturalLanguage.interpret_plot_command(cmd)

    # if parsed:
    #     plot_type, feature, cls = parsed
    #     if plot_type == "histogram":
    #         plot.plot_histogram(DTR, LTR, feature, cls)
    #     elif plot_type == "boxplot":
    #         plot.plot_boxplot(DTR, LTR, feature)
    #     elif plot_type == "feature ranking":
    #         plot.plot_feature_ranking(ranking, top_k=feature)


    return 0

if __name__ == "__main__":
    main()