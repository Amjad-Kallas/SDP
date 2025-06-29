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

    # Try multiple prompt strategies
    prompts = [
        # Strategy 1: Direct JSON generation
        f"""Generate JSON for diabetes data analysis. Return ONLY valid JSON.

FEATURES: {feature_list}

RULES:
- "distribution" → plot_type: "histogram"
- "decision boundary" or "decision" → plot_type: "decision_boundary" (requires two features)
- "Blood Pressure" → feature_index: 2
- "for ok" or "ok class" → class_filter: "ok"
- "for ko" or "ko class" → class_filter: "ko"
- "for all" or "all classes" → class_filter: "all"
- "compare" or "both" → class_filter: "compare"
- No class specified → class_filter: "compare"
- Match feature names exactly from the list above

INPUT: "{text}"
OUTPUT:""",
        
        # Strategy 2: Structured format
        f"""You are a data analysis assistant. Parse this request and return JSON.

Available features: {feature_list}

Request: "{text}"

Class detection:
- "for ok" = "ok" class only
- "for ko" = "ko" class only  
- "for all" = all classes combined
- "compare" = both classes separately
- Default = "compare"

Decision boundary format (for two features):
{{"plot_type": "decision_boundary", "feature1_index": 1, "feature2_index": 5, "feature1_name": "Glucose", "feature2_name": "BMI", "class_filter": "compare", "additional_params": {{}}}}

Single feature format:
{{"plot_type": "histogram", "feature_name": "BMI", "feature_index": 5, "class_filter": "compare", "additional_params": {{}}}}

JSON:""",
        
        # Strategy 3: Simple mapping
        f"""Map this request to JSON format.

Features: {feature_list}
Request: "{text}"

Rules:
- "distribution" = histogram
- "decision boundary" = decision_boundary (needs two features)
- "Blood Pressure" = index 2
- "BMI" = index 5
- "Glucose" = index 1
- "Pregnancies" = index 0
- "Skin Thickness" = index 3
- "Insulin" = index 4
- "DPF" = index 6
- "Age" = index 7
- "for ok" = ok class
- "for ko" = ko class
- "for all" = all classes
- "compare" = both classes

JSON:"""
    ]

    for i, prompt in enumerate(prompts):
        try:
            print(f"Llama-2 trying strategy {i+1} for query: {text}")
            response = llm(prompt=prompt, max_tokens=128, temperature=0.0, stop=["\n", "###"])
            result_text = response['choices'][0]['text'].strip()

            print(f"Strategy {i+1} response: {result_text}")

            # Try to extract JSON
            result = None
            
            # Method 1: Direct JSON parsing
            try:
                result = json.loads(result_text)
            except:
                pass
            
            # Method 2: Extract JSON from text
            if not result:
                match = re.search(r"\{.*?\}", result_text, re.DOTALL)
                if match:
                    try:
                        result = json.loads(match.group())
                    except:
                        pass
            
            # Method 3: Extract individual fields
            if not result:
                plot_type_match = re.search(r'"plot_type":\s*"([^"]+)"', result_text)
                feature_index_match = re.search(r'"feature_index":\s*(\d+)', result_text)
                feature_name_match = re.search(r'"feature_name":\s*"([^"]+)"', result_text)
                
                if plot_type_match and feature_index_match:
                    result = {
                        "plot_type": plot_type_match.group(1),
                        "feature_index": int(feature_index_match.group(1)),
                        "feature_name": feature_name_match.group(1) if feature_name_match else "Unknown",
                        "class_filter": "compare",
                        "additional_params": {}
                    }
            
            # Method 4: Manual mapping based on text content
            if not result:
                text_lower = text.lower()
                
                # Check for decision boundary requests first
                if "decision boundary" in text_lower or "decision" in text_lower:
                    # Extract two features for decision boundary
                    feature1_idx = None
                    feature2_idx = None
                    feature1_name = None
                    feature2_name = None
                    
                    # Check for feature pairs
                    if "bmi" in text_lower:
                        if feature1_idx is None:
                            feature1_idx = 5
                            feature1_name = "BMI"
                        else:
                            feature2_idx = 5
                            feature2_name = "BMI"
                    if "blood pressure" in text_lower:
                        if feature1_idx is None:
                            feature1_idx = 2
                            feature1_name = "Blood Pressure"
                        else:
                            feature2_idx = 2
                            feature2_name = "Blood Pressure"
                    if "glucose" in text_lower:
                        if feature1_idx is None:
                            feature1_idx = 1
                            feature1_name = "Glucose"
                        else:
                            feature2_idx = 1
                            feature2_name = "Glucose"
                    if "pregnancies" in text_lower:
                        if feature1_idx is None:
                            feature1_idx = 0
                            feature1_name = "Pregnancies"
                        else:
                            feature2_idx = 0
                            feature2_name = "Pregnancies"
                    if "skin thickness" in text_lower:
                        if feature1_idx is None:
                            feature1_idx = 3
                            feature1_name = "Skin Thickness"
                        else:
                            feature2_idx = 3
                            feature2_name = "Skin Thickness"
                    if "insulin" in text_lower:
                        if feature1_idx is None:
                            feature1_idx = 4
                            feature1_name = "Insulin"
                        else:
                            feature2_idx = 4
                            feature2_name = "Insulin"
                    if "dpf" in text_lower:
                        if feature1_idx is None:
                            feature1_idx = 6
                            feature1_name = "DPF"
                        else:
                            feature2_idx = 6
                            feature2_name = "DPF"
                    if "age" in text_lower:
                        if feature1_idx is None:
                            feature1_idx = 7
                            feature1_name = "Age"
                        else:
                            feature2_idx = 7
                            feature2_name = "Age"
                    
                    if feature1_idx is not None and feature2_idx is not None:
                        result = {
                            "plot_type": "decision_boundary",
                            "feature1_index": feature1_idx,
                            "feature2_index": feature2_idx,
                            "feature1_name": feature1_name,
                            "feature2_name": feature2_name,
                            "class_filter": "compare",
                            "additional_params": {}
                        }
                
                # If not decision boundary, handle single feature requests
                if not result:
                    # Determine plot type
                    if "distribution" in text_lower or "histogram" in text_lower:
                        plot_type = "histogram"
                    elif "boxplot" in text_lower:
                        plot_type = "boxplot"
                    elif "scatter" in text_lower:
                        plot_type = "scatter"
                    elif "statistics" in text_lower:
                        plot_type = "statistics"
                    else:
                        plot_type = "histogram"  # default
                    
                    # Determine class filter
                    class_filter = "compare"  # default to comparing both classes
                    if "for ok" in text_lower or "ok class" in text_lower or "ok only" in text_lower:
                        class_filter = "ok"
                    elif "for ko" in text_lower or "ko class" in text_lower or "ko only" in text_lower:
                        class_filter = "ko"
                    elif "for all" in text_lower or "all classes" in text_lower:
                        class_filter = "all"
                    elif "compare" in text_lower or "both" in text_lower:
                        class_filter = "compare"
                    
                    # Determine feature index
                    feature_index = None
                    feature_name = None
                    
                    if "bmi" in text_lower:
                        feature_index = 5
                        feature_name = "BMI"
                    elif "blood pressure" in text_lower:
                        feature_index = 2
                        feature_name = "Blood Pressure"
                    elif "glucose" in text_lower:
                        feature_index = 1
                        feature_name = "Glucose"
                    elif "pregnancies" in text_lower:
                        feature_index = 0
                        feature_name = "Pregnancies"
                    elif "skin thickness" in text_lower:
                        feature_index = 3
                        feature_name = "Skin Thickness"
                    elif "insulin" in text_lower:
                        feature_index = 4
                        feature_name = "Insulin"
                    elif "dpf" in text_lower:
                        feature_index = 6
                        feature_name = "DPF"
                    elif "age" in text_lower:
                        feature_index = 7
                        feature_name = "Age"
                    
                    if feature_index is not None:
                        result = {
                            "plot_type": plot_type,
                            "feature_index": feature_index,
                            "feature_name": feature_name,
                            "class_filter": class_filter,
                            "additional_params": {}
                        }
            
            # If we got a valid result, return it
            if result and 'plot_type' in result:
                print(f"Strategy {i+1} succeeded!")
                
                # Handle decision boundary results (they have two feature indices)
                if result['plot_type'] == 'decision_boundary' and 'feature1_index' in result and 'feature2_index' in result:
                    return (
                        result['plot_type'],
                        result['feature1_index'],
                        result['feature2_index']
                    )
                # Handle regular single-feature results
                elif 'feature_index' in result:
                    return (
                        result['plot_type'],
                        result['feature_index'],
                        result.get('class_filter', 'all'),
                        result.get('additional_params', {})
                    )
                
        except Exception as e:
            print(f"Strategy {i+1} failed: {e}")
            continue

    print("All Llama-2 strategies failed, falling back to regex parsing")
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
