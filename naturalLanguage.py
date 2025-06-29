import re
import json
from llama_cpp import Llama

# Initialize Llama-2 model
try:
    llm = Llama(
        model_path="./llama-2-7b-chat.Q4_K_M.gguf",
        n_ctx=2048,
        n_threads=4
    )
    LLAMA_AVAILABLE = True
except Exception as e:
    print(f"Llama-2 not available: {e}")
    LLAMA_AVAILABLE = False

def interpret_plot_command_with_llama(text, feature_names=None, default_feature=0):
    if not LLAMA_AVAILABLE:
        print("Llama-2 not available, falling back to regex")
        return interpret_plot_command(text, feature_names, default_feature)

    feature_list = ", ".join([f"{i}: {name}" for i, name in enumerate(feature_names)]) if feature_names else ""

    prompt = f"""
You are a helpful data analysis assistant. The user will describe a plot they want from diabetes data. You must return a valid JSON object with the following fields:

{{
  "plot_type": "histogram" | "boxplot" | "scatter" | "correlation" | "distribution" | "statistics" | "decision_boundary",
  "feature_name": "<feature name from list below>",
  "feature_index": <index from the list>,
  "class_filter": "all" | "ok" | "ko" | "compare",
  "additional_params": {{ }}
}}

INSTRUCTIONS:
- If the user says things like "how X behaves" or "what does X look like", use plot_type = "histogram".
- If the user mentions both "ok" and "ko", set class_filter = "compare".
- Match the feature name to the closest match from this list and provide the correct index.
- IMPORTANT: If the feature is "Blood Pressure", set the feature_index to 2.
- 

FEATURES:
{feature_list}

USER INPUT: "{text}"

Respond with JSON only:
###
"""


    try:
        print(f"Llama-2 processing query: {text}")
        response = llm(prompt=prompt, max_tokens=200, temperature=0.1, stop=["###"])
        result_text = response['choices'][0]['text'].strip()

        match = re.search(r"\{.*?\}", result_text, re.DOTALL)
        if not match:
            raise ValueError("No valid JSON found")

        result = json.loads(match.group())
        print(f"Parsed JSON: {result}")
        print("----------------------------------------------------------------")
        print(text, feature_names)
        if 'plot_type' in result and 'feature_index' in result:
            return (
                result['plot_type'],
                result['feature_index'],
                result.get('class_filter', 'all'),
                result.get('additional_params', {})
            )
    except Exception as e:
        print(f"Llama-2 parsing failed: {e}")

    print("Falling back to regex parsing")
    return interpret_plot_command(text, feature_names, default_feature)

def interpret_plot_command(text, feature_names=None, default_feature=0):
    text = re.sub(r'\s+', ' ', text).strip().lower()  # normalize whitespace
    patterns = [
        r"show\s+([\w\s]+)\s+feature\s+distribution",
        r"show\s+([\w\s]+)\s+feature\s+statistics",
        r"plot\s+decision\s+boundary\s+between\s+([\w\s]+)\s+and\s+([\w\s]+)",
        r"show\s+decision\s+boundary\s+for\s+([\w\s]+)\s+vs\s+([\w\s]+)",
        r"show\s+correlation\s+between\s+([\w\s]+)\s+and\s+([\w\s]+)",
        r"compare\s+([\w\s]+)\s+distributions",
        r"plot\s+(histogram|boxplot|scatter)\s+of\s+([\w\s]+?)\s+for\s+(ok|ko|all)",
        r"plot\s+(histogram|boxplot|scatter)\s+of\s+([\w\s]+)",
        r"display\s+([\w\s]+)\s+distribution\s+for\s+(ok|ko|all)\s+class",
        r"(histogram|boxplot|scatter).*feature\s+(\d+).*class\s+(\d+)",
        r"feature\s+ranking\s+top\s+(\d+)",
    ]

    for i, pattern in enumerate(patterns):
        match = re.search(pattern, text)
        if match:
            if i == 0:
                return _parse_feature_request(match.group(1), "histogram", "compare", feature_names)
            elif i == 1:
                return _parse_feature_request(match.group(1), "statistics", "compare", feature_names)
            elif i in [2, 3]:
                return _parse_decision_boundary_request(match.group(1), match.group(2), feature_names)
            elif i == 4:
                return _parse_correlation_request(match.group(1), match.group(2), feature_names)
            elif i == 5:
                return _parse_feature_request(match.group(1), "histogram", "compare", feature_names)
            elif i == 6:
                return _parse_feature_request(match.group(2).strip(), match.group(1), match.group(3) or "all", feature_names)
            elif i == 7:
                return _parse_feature_request(match.group(2), match.group(1), "all", feature_names)
            elif i == 8:
                return _parse_feature_request(match.group(1), "histogram", match.group(2), feature_names)
            elif i == 9:
                feature = int(match.group(2))
                group = "ok" if int(match.group(3)) == 0 else "ko"
                return match.group(1), feature, group
            elif i == 10:
                return "feature_ranking", int(match.group(1)), None

    return None

def _parse_feature_request(feature, plot_type, group, feature_names):
    print("------------------------------------------------------------")
    if feature_names:
        feature_map = {
            f.lower().replace(' ', ''): (i, f) for i, f in enumerate(feature_names)
        }
        feature_clean = feature.strip().lower().replace(' ', '')
        print(f"[DEBUG] Normalized feature: '{feature_clean}'")

        if feature_clean in feature_map:
            idx, _ = feature_map[feature_clean]

            # Manual fix for 'blood pressure'
            if feature_clean == "bloodpressure":
                print("[MANUAL FIX] Adjusting index +1 for 'blood pressure'")
                idx = 2

            print(f"[DEBUG] Final match index: {idx}")
            return plot_type, idx, group
        else:
            print(f"[WARN] Feature '{feature_clean}' not found")
    print("------------------------------------------------------------")
    return None

def _parse_correlation_request(f1, f2, feature_names):
    if feature_names:
        clean = [f.lower().replace(' ', '') for f in feature_names]
        f1, f2 = f1.lower().replace(' ', ''), f2.lower().replace(' ', '')
        if f1 in clean and f2 in clean:
            return "correlation", clean.index(f1), clean.index(f2)
    return None

def _parse_decision_boundary_request(f1, f2, feature_names):
    if feature_names:
        clean = [f.lower().replace(' ', '') for f in feature_names]
        f1, f2 = f1.lower().replace(' ', ''), f2.lower().replace(' ', '')
        if f1 in clean and f2 in clean:
            return "decision_boundary", clean.index(f1), clean.index(f2)
    return None

def generate_plot_description(plot_type, feature_name, class_filter="all"):
    desc = {
        "histogram": f"Histogram of {feature_name} distribution",
        "boxplot": f"Boxplot of {feature_name} values",
        "scatter": f"Scatter plot of {feature_name}",
        "correlation": f"Correlation analysis",
        "distribution": f"Distribution of {feature_name}",
        "statistics": f"Detailed statistics for {feature_name}",
        "decision_boundary": f"Decision boundary analysis"
    }
    base = desc.get(plot_type, f"{plot_type} of {feature_name}")
    if class_filter == "compare":
        return f"{base} comparing OK and KO classes"
    elif class_filter == "all":
        return f"{base} for all classes"
    else:
        return f"{base} for {class_filter.upper()} class"
