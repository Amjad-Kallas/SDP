import numpy as np
from scipy import stats

def feature_statistical_separation(D, L, group_labels=(0, 1)):
    """
    Calculate statistical measures for each feature, for each group (OK, KO),
    and compute separation scores for each statistic.
    Returns a dict of statistics for each feature and their separation scores.
    """
    results = {}
    for feat_idx in range(D.shape[0]):
        feat_stats = {}
        for group in group_labels:
            group_data = D[feat_idx, L == group]
            feat_stats[f"mean_{group}"] = np.mean(group_data)
            feat_stats[f"median_{group}"] = np.median(group_data)
            feat_stats[f"mode_{group}"] = stats.mode(group_data, keepdims=False).mode
            feat_stats[f"std_{group}"] = np.std(group_data)
            feat_stats[f"var_{group}"] = np.var(group_data)
        # Separation scores (absolute difference between groups)
        feat_stats["mean_diff"] = abs(feat_stats["mean_0"] - feat_stats["mean_1"])
        feat_stats["median_diff"] = abs(feat_stats["median_0"] - feat_stats["median_1"])
        feat_stats["mode_diff"] = abs(feat_stats["mode_0"] - feat_stats["mode_1"])
        feat_stats["std_diff"] = abs(feat_stats["std_0"] - feat_stats["std_1"])
        feat_stats["var_diff"] = abs(feat_stats["var_0"] - feat_stats["var_1"])
        results[f"feature_{feat_idx}"] = feat_stats
    return results

def rank_separating_statistics(D, L, group_labels=(0, 1)):
    """
    Ranks statistics (mean, median, mode, std, var) by their average separation across all features.
    Returns a sorted list of (statistic, average_diff) tuples.
    """
    results = feature_statistical_separation(D, L, group_labels)
    stat_names = ["mean_diff", "median_diff", "mode_diff", "std_diff", "var_diff"]
    stat_sums = {stat: 0.0 for stat in stat_names}
    n_features = len(results)
    for feat_stats in results.values():
        for stat in stat_names:
            stat_sums[stat] += feat_stats[stat]
    stat_avgs = {stat: stat_sums[stat] / n_features for stat in stat_names}
    ranked = sorted(stat_avgs.items(), key=lambda x: x[1], reverse=True)
    return ranked

def feature_p_values(D, L, test="ttest"):
    """
    Computes p-values for each feature between the two groups using the specified test.
    test: 'ttest' (default) or 'mannwhitney'.
    Returns a sorted list of (feature_idx, p_value) tuples (ascending p-value).
    """
    pvals = []
    for feat_idx in range(D.shape[0]):
        group0 = D[feat_idx, L == 0]
        group1 = D[feat_idx, L == 1]
        if test == "ttest":
            stat, p = stats.ttest_ind(group0, group1, equal_var=False)
        elif test == "mannwhitney":
            stat, p = stats.mannwhitneyu(group0, group1, alternative="two-sided")
        else:
            raise ValueError("Unknown test type")
        pvals.append((feat_idx, p))
    pvals_sorted = sorted(pvals, key=lambda x: x[1])
    return pvals_sorted

def statistic_discriminative_power(D, L, group_labels=(0, 1), test="ttest"):
    """
    For each statistic (mean, median, std, var, mode), computes the p-value between groups across all features.
    Returns a sorted list of (statistic, average_p_value) tuples (ascending p-value).
    """
    stat_names = ["mean", "median", "mode", "std", "var"]
    stat_pvals = {}
    for stat in stat_names:
        group0_stats = []
        group1_stats = []
        for feat_idx in range(D.shape[0]):
            group0 = D[feat_idx, L == group_labels[0]]
            group1 = D[feat_idx, L == group_labels[1]]
            if stat == "mean":
                group0_stats.append(np.mean(group0))
                group1_stats.append(np.mean(group1))
            elif stat == "median":
                group0_stats.append(np.median(group0))
                group1_stats.append(np.median(group1))
            elif stat == "mode":
                group0_stats.append(stats.mode(group0, keepdims=False).mode)
                group1_stats.append(stats.mode(group1, keepdims=False).mode)
            elif stat == "std":
                group0_stats.append(np.std(group0))
                group1_stats.append(np.std(group1))
            elif stat == "var":
                group0_stats.append(np.var(group0))
                group1_stats.append(np.var(group1))
        if test == "ttest":
            _, p = stats.ttest_ind(group0_stats, group1_stats, equal_var=False)
        elif test == "mannwhitney":
            _, p = stats.mannwhitneyu(group0_stats, group1_stats, alternative="two-sided")
        else:
            raise ValueError("Unknown test type")
        stat_pvals[stat] = p
    ranked = sorted(stat_pvals.items(), key=lambda x: x[1])
    return ranked


def main():
    import utils
    # Load and split the real data
    D, L = utils.load_data("diabetes_short.csv")
    (DTR, LTR), (DVAL, LVAL) = utils.split_db_2to1(D, L)

    # print("Feature Statistical Separation:")
    # results = feature_statistical_separation(DTR, LTR)
    # for feat, feat_stats in results.items():
    #     print(f"{feat}: {feat_stats}")

    # print("\nRanked Separating Statistics:")
    # ranked_stats = rank_separating_statistics(DTR, LTR)
    # for stat, avg_diff in ranked_stats:
    #     print(f"{stat}: {avg_diff}")

    print("\nFeature P-Values (T-Test):")
    pvals_ttest = feature_p_values(DTR, LTR, test="ttest")
    for feat_idx, pval in pvals_ttest:
        print(f"Feature {feat_idx}: p-value = {pval}")

    print("\nDiscriminative Power of Statistics:")
    discriminative_power = statistic_discriminative_power(DTR, LTR)
    for stat, pval in discriminative_power:
        print(f"{stat}: p-value = {pval}")


if __name__ == "__main__":
    main()

