import re
from llama_cpp import Llama
import json

# Initialize Llama-2 model
try:
    llm = Llama(
        model_path="./llama-2-7b-chat.Q4_K_M.gguf",  # Adjust path as needed
        n_ctx=2048,
        n_threads=4
    )
    LLAMA_AVAILABLE = True
except Exception as e:
    print(f"Llama-2 not available: {e}")
    LLAMA_AVAILABLE = False

def interpret_plot_command_with_llama(text, feature_names=None, default_feature=0):
    """
    Use Llama-2 to interpret natural language plot commands
    """
    if not LLAMA_AVAILABLE:
        return interpret_plot_command(text, feature_names, default_feature)
    
    # Create a prompt for Llama-2
    feature_list = ", ".join([f"{i}: {name}" for i, name in enumerate(feature_names)]) if feature_names else ""
    
    prompt = f"""You are a data analysis assistant. Given a user's natural language request for plotting diabetes data, extract the following information in JSON format:
- plot_type: "histogram", "boxplot", "scatter", "correlation", "distribution", "statistics", or "decision_boundary"
- feature_name: the feature to plot (from the available features)
- feature_index: the index of the feature (0-7)
- class_filter: "all", "ok", "ko", or "compare"
- additional_params: any additional parameters

IMPORTANT RULES:
1. When users ask for "feature distribution" or "show distribution", use "compare" as class_filter to show OK vs KO classes with different colors for better insights.
2. When users ask for "feature statistics" or "show statistics", use plot_type "statistics" and class_filter "compare" to show detailed statistical parameters for both classes.
3. When users ask for "decision boundary between X and Y", use plot_type "decision_boundary" and set feature_name to the first feature (X).

Available features: {feature_list}

User request: "{text}"

Respond with only valid JSON:
"""
    
    try:
        response = llm(prompt, max_tokens=200, temperature=0.1, stop=["\n\n"])
        result = json.loads(response['choices'][0]['text'].strip())
        
        # Validate and return the parsed result
        if 'plot_type' in result and 'feature_index' in result:
            return (
                result['plot_type'],
                result['feature_index'],
                result.get('class_filter', 'all'),
                result.get('additional_params', {})
            )
    except Exception as e:
        print(f"Llama-2 parsing failed: {e}")
    
    # Fallback to regex parsing
    return interpret_plot_command(text, feature_names, default_feature)

def interpret_plot_command(text, feature_names=None, default_feature=0):
    """
    Enhanced regex-based parsing with more patterns
    """
    text = text.lower()
    
    # Enhanced patterns for better matching
    patterns = [
        # "show BMI feature distribution"
        r"show\s+([\w\s]+)\s+feature\s+distribution",
        # "show BMI feature statistics"
        r"show\s+([\w\s]+)\s+feature\s+statistics",
        # "plot decision boundary between X and Y" - MORE SPECIFIC FIRST
        r"plot\s+decision\s+boundary\s+between\s+([\w\s]+)\s+and\s+([\w\s]+)",
        # "show decision boundary for X vs Y" - MORE SPECIFIC FIRST
        r"show\s+decision\s+boundary\s+for\s+([\w\s]+)\s+vs\s+([\w\s]+)",
        # "show correlation between X and Y"
        r"show\s+correlation\s+between\s+([\w\s]+)\s+and\s+([\w\s]+)",
        # "compare BMI distributions"
        r"compare\s+([\w\s]+)\s+distributions",
        # "plot histogram of Glucose for KO" - GENERAL PLOT PATTERN LAST
        r"plot\s+(histogram|boxplot|scatter)\s+of\s+([\w\s]+?)(?:\s+for\s+(ok|ko|all))?",
        # "display BMI distribution for OK class"
        r"display\s+([\w\s]+)\s+distribution\s+for\s+(ok|ko|all)\s+class",
        # "histogram of feature X for class Y"
        r"(histogram|boxplot|scatter).*feature\s+(\d+).*class\s+(\d+)",
        # "feature ranking top X"
        r"feature\s+ranking\s+top\s+(\d+)",
    ]
    
    for i, pattern in enumerate(patterns):
        match = re.search(pattern, text)
        if match:
            # Pattern 0: "show BMI feature distribution"
            if i == 0:
                feature = match.group(1)
                return _parse_feature_request(feature, "histogram", "compare", feature_names)
            # Pattern 1: "show BMI feature statistics"
            elif i == 1:
                feature = match.group(1)
                return _parse_feature_request(feature, "statistics", "compare", feature_names)
            # Pattern 2: "plot decision boundary between X and Y"
            elif i == 2:
                feature1 = match.group(1).strip()
                feature2 = match.group(2).strip()
                return _parse_decision_boundary_request(feature1, feature2, feature_names)
            # Pattern 3: "show decision boundary for X vs Y"
            elif i == 3:
                feature1 = match.group(1).strip()
                feature2 = match.group(2).strip()
                return _parse_decision_boundary_request(feature1, feature2, feature_names)
            # Pattern 4: "show correlation between X and Y"
            elif i == 4:
                feature1 = match.group(1).strip()
                feature2 = match.group(2).strip()
                return _parse_correlation_request(feature1, feature2, feature_names)
            # Pattern 5: "compare BMI distributions"
            elif i == 5:
                feature = match.group(1)
                return _parse_feature_request(feature, "histogram", "compare", feature_names)
            # Pattern 6: "plot histogram of Glucose for KO"
            elif i == 6:
                plot_type = match.group(1)
                feature = match.group(2).strip()
                group = match.group(3) if match.group(3) else "all"
                return _parse_feature_request(feature, plot_type, group, feature_names)
            # Pattern 7: "display BMI distribution for OK class"
            elif i == 7:
                feature = match.group(1)
                group = match.group(2)
                return _parse_feature_request(feature, "histogram", group, feature_names)
            # Pattern 8: "histogram of feature X for class Y"
            elif i == 8:
                plot_type = match.group(1)
                feature = int(match.group(2))
                cls = int(match.group(3))
                group = "ok" if cls == 0 else "ko"
                return (plot_type, feature, group)
            # Pattern 9: "feature ranking top X"
            elif i == 9:
                top_k = int(match.group(1))
                return ("feature_ranking", top_k, None)
    
    return None

def _parse_feature_request(feature, plot_type, group, feature_names):
    """Helper function to parse feature requests"""
    if feature_names:
        feature_names_clean = [f.lower().replace(' ', '') for f in feature_names]
        feature_clean = feature.lower().replace(' ', '')
        
        if feature_clean in feature_names_clean:
            feature_idx = feature_names_clean.index(feature_clean)
            return plot_type, feature_idx, group
        else:
            print(f"Feature '{feature}' not found in available features!")
            return None
    return None

def _parse_correlation_request(feature1, feature2, feature_names):
    """Helper function to parse correlation requests"""
    if feature_names:
        feature_names_clean = [f.lower().replace(' ', '') for f in feature_names]
        feature1_clean = feature1.lower().replace(' ', '')
        feature2_clean = feature2.lower().replace(' ', '')
        
        if feature1_clean in feature_names_clean and feature2_clean in feature_names_clean:
            idx1 = feature_names_clean.index(feature1_clean)
            idx2 = feature_names_clean.index(feature2_clean)
            return ("correlation", idx1, idx2)
    return None

def _parse_decision_boundary_request(feature1, feature2, feature_names):
    """Helper function to parse decision boundary requests"""
    if feature_names:
        feature_names_clean = [f.lower().replace(' ', '') for f in feature_names]
        feature1_clean = feature1.lower().replace(' ', '')
        feature2_clean = feature2.lower().replace(' ', '')
        
        if feature1_clean in feature_names_clean and feature2_clean in feature_names_clean:
            idx1 = feature_names_clean.index(feature1_clean)
            idx2 = feature_names_clean.index(feature2_clean)
            return ("decision_boundary", idx1, idx2)
        else:
            print(f"Features '{feature1}' or '{feature2}' not found in available features!")
            return None
    return None

def generate_plot_description(plot_type, feature_name, class_filter="all"):
    """Generate a natural language description of the plot"""
    descriptions = {
        "histogram": f"Histogram of {feature_name} distribution",
        "boxplot": f"Boxplot of {feature_name} values",
        "scatter": f"Scatter plot of {feature_name}",
        "correlation": f"Correlation analysis",
        "distribution": f"Distribution of {feature_name}",
        "statistics": f"Detailed statistics for {feature_name}",
        "decision_boundary": f"Decision boundary analysis"
    }
    
    base_desc = descriptions.get(plot_type, f"{plot_type} of {feature_name}")
    
    if class_filter == "all":
        return f"{base_desc} for all classes"
    elif class_filter == "compare":
        return f"{base_desc} comparing OK and KO classes"
    else:
        return f"{base_desc} for {class_filter.upper()} class"

