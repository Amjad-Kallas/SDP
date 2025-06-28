import re

def interpret_plot_command(text, feature_names=None, default_feature=0):
    text = text.lower()

    # Try: "plot histogram of <feature> for <group>"
    m = re.search(r"plot (histogram|boxplot) of ([\w ]+?)(?: for (ok|ko|all))?$", text)
    if m:
        plot_type = m.group(1)
        feature = m.group(2).strip()
        group = m.group(3) if m.group(3) else "all"
        # Map feature name to index
        if feature_names:
            feature_names_clean = [f.lower().replace(' ', '') for f in feature_names]
            feature_clean = feature.lower().replace(' ', '')
            if feature_clean in feature_names_clean:
                feature_idx = feature_names_clean.index(feature_clean)
                return plot_type, feature_idx, group
            else:
                return None  # Feature not found
        else:
            return None

    # New: "show feature <feature> <analysis_type>"
    m = re.search(r"show feature ([\w ]+?) ([\w ]+?)(?: for ([\w ]+?))?$", text)
    if m:
        feature = m.group(1).strip()
        analysis_type = m.group(2).strip()
        group = m.group(3).strip() if m.group(3) else None
        
        # Map feature name to index
        if feature_names:
            feature_names_clean = [f.lower().replace(' ', '') for f in feature_names]
            feature_clean = feature.lower().replace(' ', '')
            if feature_clean in feature_names_clean:
                feature_idx = feature_names_clean.index(feature_clean)
                
                # Map analysis type to plot type
                if analysis_type in ['distribution', 'distributions']:
                    return "feature_distributions", feature_idx, group
                elif analysis_type in ['statistics', 'statistic']:
                    return "feature_statistics_comparison", feature_idx, group
                elif analysis_type in ['p-values', 'p values', 'pvalues']:
                    return "p_values_comparison", feature_idx, group
                elif analysis_type in ['histogram']:
                    return "histogram", feature_idx, group
            else:
                return None  # Feature not found
        else:
            return None

    # New: "show <feature> <analysis_type>"
    m = re.search(r"show ([\w ]+?) ([\w ]+?)(?: for ([\w ]+?))?$", text)
    if m:
        feature = m.group(1).strip()
        analysis_type = m.group(2).strip()
        group = m.group(3).strip() if m.group(3) else None
        
        # Map feature name to index
        if feature_names:
            feature_names_clean = [f.lower().replace(' ', '') for f in feature_names]
            feature_clean = feature.lower().replace(' ', '')
            if feature_clean in feature_names_clean:
                feature_idx = feature_names_clean.index(feature_clean)
                
                # Map analysis type to plot type
                if analysis_type in ['distribution', 'distributions']:
                    return "feature_distributions", feature_idx, group
                elif analysis_type in ['statistics', 'statistic']:
                    return "feature_statistics_comparison", feature_idx, group
                elif analysis_type in ['p-values', 'p values', 'pvalues']:
                    return "p_values_comparison", feature_idx, group
                elif analysis_type in ['histogram']:
                    return "histogram", feature_idx, group
            else:
                return None  # Feature not found
        else:
            return None

    # New: "show <analysis_type> for <feature>"
    m = re.search(r"show ([\w ]+?) for ([\w ]+?)(?: for ([\w ]+?))?$", text)
    if m:
        analysis_type = m.group(1).strip()
        feature = m.group(2).strip()
        group = m.group(3).strip() if m.group(3) else None
        
        # Map feature name to index
        if feature_names:
            feature_names_clean = [f.lower().replace(' ', '') for f in feature_names]
            feature_clean = feature.lower().replace(' ', '')
            if feature_clean in feature_names_clean:
                feature_idx = feature_names_clean.index(feature_clean)
                
                # Map analysis type to plot type
                if analysis_type in ['distribution', 'distributions']:
                    return "feature_distributions", feature_idx, group
                elif analysis_type in ['statistics', 'statistic']:
                    return "feature_statistics_comparison", feature_idx, group
                elif analysis_type in ['p-values', 'p values', 'pvalues']:
                    return "p_values_comparison", feature_idx, group
                elif analysis_type in ['histogram']:
                    return "histogram", feature_idx, group
            else:
                return None  # Feature not found
        else:
            return None

    # Try full pattern: "histogram of feature X for class Y"
    match = re.search(r"(histogram|boxplot).*feature (\d+).*class (\d+)", text)
    if match:
        plot_type = match.group(1)
        feature = int(match.group(2))
        cls = int(match.group(3))
        return (plot_type, feature, cls)

    # Try: "histogram for class Y" (use default feature)
    match = re.search(r"(histogram|boxplot).*class (\d+)", text)
    if match:
        plot_type = match.group(1)
        cls = int(match.group(2))
        return (plot_type, default_feature, cls)

    # Try: "boxplot of feature X" (no class needed)
    match = re.search(r"boxplot.*feature (\d+)", text)
    if match:
        return ("boxplot", int(match.group(1)), None)

    match = re.search(r"feature ranking top (\d+)", text)
    if match:
        top_k = int(match.group(1))
        return ("feature ranking", top_k, None)

    # New: time series of feature X for class Y
    match = re.search(r"time series.*feature (\d+).*class (\d+)", text)
    if match:
        feature = int(match.group(1))
        cls = int(match.group(2))
        return ("time series", feature, cls)

    # New: frequency spectrum of feature X for class Y
    match = re.search(r"frequency spectrum.*feature (\d+).*class (\d+)", text)
    if match:
        feature = int(match.group(1))
        cls = int(match.group(2))
        return ("frequency spectrum", feature, cls)

    # New: Statistical analysis patterns
    # "show statistical separation ranking"
    if re.search(r"statistical separation ranking", text):
        return ("statistical_separation_ranking", None, None)
    
    # "show discriminative power"
    if re.search(r"discriminative power", text):
        return ("discriminative_power", None, None)
    
    # "show p values comparison"
    if re.search(r"p.?values? comparison", text):
        return ("p_values_comparison", None, None)
    
    # "show feature statistics comparison"
    if re.search(r"feature statistics comparison", text):
        return ("feature_statistics_comparison", None, None)
    
    # "show feature distributions"
    if re.search(r"feature distributions", text):
        return ("feature_distributions", None, None)
    
    # "comprehensive statistical analysis"
    if re.search(r"comprehensive statistical analysis", text):
        return ("comprehensive_analysis", None, None)

    return None

