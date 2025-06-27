import re

def interpret_plot_command(text, default_feature=0):
    text = text.lower()

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

    return None

