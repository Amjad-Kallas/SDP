import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_histogram(D, L, feature_idx, class_label, bins=20, feature_names=None):
    """Plot histogram for a specific feature and class."""
    values = D[feature_idx, L == class_label]
    feature_label = f"{feature_idx}"
    if feature_names is not None and feature_idx < len(feature_names):
        feature_label = f"{feature_idx} - {feature_names[feature_idx]}"
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(values, bins=bins, alpha=0.7, color='blue' if class_label == 0 else 'red')
    ax.set_title(f"Histogram of Feature {feature_label} for Class {class_label}")
    ax.set_xlabel(f"Feature {feature_label} value")
    ax.set_ylabel("Frequency")
    ax.grid(True)
    return fig

def plot_boxplot(D, L, feature_idx, feature_names=None):
    """Plot boxplot for a specific feature split by class."""
    values_ok = D[feature_idx, L == 0]
    values_ko = D[feature_idx, L == 1]
    feature_label = f"{feature_idx}"
    if feature_names is not None and feature_idx < len(feature_names):
        feature_label = f"{feature_idx} - {feature_names[feature_idx]}"
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.boxplot([values_ok, values_ko], labels=['OK (0)', 'KO (1)'])
    ax.set_title(f"Boxplot of Feature {feature_label} by Class")
    ax.set_ylabel(f"Feature {feature_label} Value")
    ax.grid(True)
    return fig

def plot_feature_ranking(ranking, top_k=5, feature_names=None):
    """Plot bar chart of top-k ranked features by discriminative score."""
    features = [int(f[0]) for f in ranking[:top_k]]
    scores = [f[1] for f in ranking[:top_k]]
    if feature_names is not None:
        featuresLabels = [f"{i} - {feature_names[i]}" for i in features]
    else:
        featuresLabels = [str(i) for i in features]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(featuresLabels, scores, color='green')
    ax.set_title(f"Top {top_k} Discriminative Features")
    ax.set_xlabel("Feature (Number - Name)")
    ax.set_ylabel("Discriminative Score")
    ax.grid(True)
    return fig

def plot_time_series(D, L, feature_idx, class_label, feature_names=None):
    """Plot time series for a specific feature and class."""
    values = D[feature_idx, L == class_label]
    feature_label = f"{feature_idx}"
    if feature_names is not None and feature_idx < len(feature_names):
        feature_label = f"{feature_idx} - {feature_names[feature_idx]}"
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(values, marker='o', linestyle='-')
    ax.set_title(f"Time Series of Feature {feature_label} for Class {class_label}")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel(f"Feature {feature_label} Value")
    ax.grid(True)
    return fig

def plot_frequency_spectrum(D, L, feature_idx, class_label, feature_names=None):
    """Plot frequency spectrum (FFT) for a specific feature and class."""
    values = D[feature_idx, L == class_label]
    N = len(values)
    fft_vals = np.fft.fft(values)
    fft_freqs = np.fft.fftfreq(N)
    feature_label = f"{feature_idx}"
    if feature_names is not None and feature_idx < len(feature_names):
        feature_label = f"{feature_idx} - {feature_names[feature_idx]}"
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(fft_freqs[:N//2], np.abs(fft_vals)[:N//2])
    ax.set_title(f"Frequency Spectrum of Feature {feature_label} for Class {class_label}")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Magnitude")
    ax.grid(True)
    return fig

