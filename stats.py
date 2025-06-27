# in this file I will implement the statistics functions

import numpy
from scipy import stats

def compute_statistics(D, L):
    """
    Compute statistics for each feature, separated by class (OK/KO).
    Returns a dict: feature_index -> {stat_name: value}
    Statistics: mean, median, mode, std, var, min, max (for each class)
    Separation: mean_diff, median_diff, mode_diff, std_diff, var_diff, min_diff, max_diff
    """
    stats_dict = {}
    OK = D[:, L == 0]
    KO = D[:, L == 1]
    n_features = D.shape[0]
    for i in range(n_features):
        feature_stats = {}
        ok_feat = OK[i, :]
        ko_feat = KO[i, :]
        feature_stats['mean_OK'] = numpy.mean(ok_feat)
        feature_stats['mean_KO'] = numpy.mean(ko_feat)
        feature_stats['median_OK'] = numpy.median(ok_feat)
        feature_stats['median_KO'] = numpy.median(ko_feat)
        feature_stats['mode_OK'] = stats.mode(ok_feat, keepdims=True)[0][0]
        feature_stats['mode_KO'] = stats.mode(ko_feat, keepdims=True)[0][0]
        feature_stats['std_OK'] = numpy.std(ok_feat)
        feature_stats['std_KO'] = numpy.std(ko_feat)
        feature_stats['var_OK'] = numpy.var(ok_feat)
        feature_stats['var_KO'] = numpy.var(ko_feat)
        feature_stats['min_OK'] = numpy.min(ok_feat)
        feature_stats['min_KO'] = numpy.min(ko_feat)
        feature_stats['max_OK'] = numpy.max(ok_feat)
        feature_stats['max_KO'] = numpy.max(ko_feat)

        # Separation measures
        feature_stats['mean_diff'] = abs(feature_stats['mean_OK'] - feature_stats['mean_KO'])
        feature_stats['median_diff'] = abs(feature_stats['median_OK'] - feature_stats['median_KO'])
        feature_stats['mode_diff'] = abs(feature_stats['mode_OK'] - feature_stats['mode_KO'])
        feature_stats['std_diff'] = abs(feature_stats['std_OK'] - feature_stats['std_KO'])
        feature_stats['var_diff'] = abs(feature_stats['var_OK'] - feature_stats['var_KO'])
        feature_stats['min_diff'] = abs(feature_stats['min_OK'] - feature_stats['min_KO'])
        feature_stats['max_diff'] = abs(feature_stats['max_OK'] - feature_stats['max_KO'])
        stats_dict[i] = feature_stats
    return stats_dict

def rank_features_by_separation(stats_dict, stat_name='mean_diff'):
    # Returns a list of (feature_index, separation_value) sorted descending
    ranking = [(i, stats_dict[i][stat_name]) for i in stats_dict]
    ranking.sort(key=lambda x: x[1], reverse=True)
    return ranking


