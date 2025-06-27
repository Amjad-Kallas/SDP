import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_histogram(D, L, feature_idx, class_label, bins=20):
    """Plot histogram for a specific feature and class."""
    values = D[feature_idx, L == class_label]
    plt.figure(figsize=(8, 4))
    plt.hist(values, bins=bins, alpha=0.7, color='blue' if class_label == 0 else 'red')
    plt.title(f"Histogram of Feature {feature_idx} for Class {class_label}")
    plt.xlabel(f"Feature {feature_idx} value")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

def plot_boxplot(D, L, feature_idx):
    """Plot boxplot for a specific feature split by class."""
    values_ok = D[feature_idx, L == 0]
    values_ko = D[feature_idx, L == 1]
    plt.figure(figsize=(8, 4))
    plt.boxplot([values_ok, values_ko], labels=['OK (0)', 'KO (1)'])
    plt.title(f"Boxplot of Feature {feature_idx} by Class")
    plt.ylabel("Feature Value")
    plt.grid(True)
    plt.show()

def plot_feature_ranking(ranking, top_k=5):
    """Plot bar chart of top-k ranked features by discriminative score."""

    myDict = {
        1: 'Pregnancies',
        2: 'Glucose',
        3: 'Blood Pressure',
        4: 'Skin Thickness',
        5: 'Insulin',
        6: 'BMI',
        7: 'DPF',
        8: 'Age'
    }

    features = [str(int(f[0])) for f in ranking[:top_k]]
    scores = [f[1] for f in ranking[:top_k]]
    
    featuresNames = [myDict[int(f)] for f in features]

    plt.figure(figsize=(10, 5))
    plt.bar(featuresNames, scores, color='green')
    plt.title(f"Top {top_k} Discriminative Features")
    plt.xlabel("Feature Index")
    plt.ylabel("Discriminative Score")
    plt.grid(True)
    plt.show()

