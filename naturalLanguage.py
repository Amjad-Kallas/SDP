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
            print("feature_names_clean:", feature_names_clean)
            print("feature_clean:", feature_clean)
            if feature_clean in feature_names_clean:
                feature_idx = feature_names_clean.index(feature_clean)
                return plot_type, feature_idx, group
            else:
                print("Feature not found!")
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

    return None

