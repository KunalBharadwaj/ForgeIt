"""
Visualization Module for Fake News Detection
Provides various plotting and visualization functions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.metrics import confusion_matrix, roc_curve, auc
import warnings

warnings.filterwarnings("ignore")


def set_style():
    """Set consistent visualization style"""
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_palette("husl")
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 12


def plot_training_history(
    history: dict, model_name: str, save_path: str = None, figsize=(12, 5)
):
    """
    Plot training history (loss and accuracy)

    Args:
        history: Dictionary with training history
        model_name: Name of the model
        save_path: Optional path to save the plot
        figsize: Figure size
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Plot loss
    if "train_loss" in history:
        ax1.plot(history["train_loss"], label="Train Loss", marker="o")
    if "val_loss" in history:
        ax1.plot(history["val_loss"], label="Validation Loss", marker="s")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"{model_name} - Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot accuracy
    if "train_acc" in history:
        ax2.plot(history["train_acc"], label="Train Accuracy", marker="o")
    if "val_acc" in history:
        ax2.plot(history["val_acc"], label="Validation Accuracy", marker="s")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title(f"{model_name} - Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Training history plot saved to {save_path}")

    plt.show()


def plot_word_cloud(
    texts: list,
    labels: np.ndarray,
    label_value: int,
    title: str,
    save_path: str = None,
    figsize=(12, 8),
):
    """
    Generate and plot word cloud for a specific label

    Args:
        texts: List of texts
        labels: Array of labels
        label_value: Label to filter (0 or 1)
        title: Plot title
        save_path: Optional path to save the plot
        figsize: Figure size
    """
    # Filter texts by label
    filtered_texts = [texts[i] for i in range(len(texts)) if labels[i] == label_value]
    combined_text = " ".join(filtered_texts)

    # Generate word cloud
    wordcloud = WordCloud(
        width=1200,
        height=800,
        background_color="white",
        colormap="viridis",
        max_words=100,
        relative_scaling=0.5,
        random_state=42,
    ).generate(combined_text)

    # Plot
    plt.figure(figsize=figsize)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(title, fontsize=16, pad=20)
    plt.tight_layout(pad=0)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Word cloud saved to {save_path}")

    plt.show()


def plot_text_length_distribution(
    df: pd.DataFrame, save_path: str = None, figsize=(12, 5)
):
    """
    Plot distribution of text lengths by label

    Args:
        df: DataFrame with 'text' and 'label' columns
        save_path: Optional path to save the plot
        figsize: Figure size
    """
    df["text_length"] = df["text"].str.len()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Histogram
    for label in df["label"].unique():
        data = df[df["label"] == label]["text_length"]
        label_name = "Fake" if label == 1 else "Real"
        ax1.hist(data, bins=50, alpha=0.6, label=label_name)

    ax1.set_xlabel("Text Length (characters)")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Distribution of Text Lengths")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Box plot
    data_to_plot = [
        df[df["label"] == 0]["text_length"],
        df[df["label"] == 1]["text_length"],
    ]
    ax2.boxplot(data_to_plot, labels=["Real", "Fake"])
    ax2.set_ylabel("Text Length (characters)")
    ax2.set_title("Text Length by Label")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Text length distribution saved to {save_path}")

    plt.show()


def plot_multiple_roc_curves(
    results_dict: dict, save_path: str = None, figsize=(10, 8)
):
    """
    Plot ROC curves for multiple models

    Args:
        results_dict: Dictionary with model names as keys and
                     (y_true, y_proba) tuples as values
        save_path: Optional path to save the plot
        figsize: Figure size
    """
    plt.figure(figsize=figsize)

    for model_name, (y_true, y_proba) in results_dict.items():
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{model_name} (AUC = {roc_auc:.3f})")

    plt.plot(
        [0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Classifier"
    )
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - Model Comparison")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"ROC curves saved to {save_path}")

    plt.show()


def plot_multiple_confusion_matrices(
    results_dict: dict, save_path: str = None, figsize=(15, 5)
):
    """
    Plot confusion matrices for multiple models

    Args:
        results_dict: Dictionary with model names as keys and
                     (y_true, y_pred) tuples as values
        save_path: Optional path to save the plot
        figsize: Figure size
    """
    n_models = len(results_dict)
    fig, axes = plt.subplots(1, n_models, figsize=figsize)

    if n_models == 1:
        axes = [axes]

    for idx, (model_name, (y_true, y_pred)) in enumerate(results_dict.items()):
        cm = confusion_matrix(y_true, y_pred)

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Real", "Fake"],
            yticklabels=["Real", "Fake"],
            ax=axes[idx],
        )
        axes[idx].set_title(f"{model_name}")
        axes[idx].set_ylabel("True Label")
        axes[idx].set_xlabel("Predicted Label")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Confusion matrices saved to {save_path}")

    plt.show()


def plot_feature_comparison(
    features_dict: dict, save_path: str = None, figsize=(12, 8)
):
    """
    Plot top features from multiple models

    Args:
        features_dict: Dictionary with model names as keys and
                      DataFrames with 'feature' and 'importance' columns as values
        save_path: Optional path to save the plot
        figsize: Figure size
    """
    n_models = len(features_dict)
    fig, axes = plt.subplots(1, n_models, figsize=figsize)

    if n_models == 1:
        axes = [axes]

    for idx, (model_name, df) in enumerate(features_dict.items()):
        # Sort by importance
        df_sorted = df.sort_values("importance", ascending=True).tail(10)

        axes[idx].barh(range(len(df_sorted)), df_sorted["importance"])
        axes[idx].set_yticks(range(len(df_sorted)))
        axes[idx].set_yticklabels(df_sorted["feature"])
        axes[idx].set_xlabel("Importance")
        axes[idx].set_title(f"{model_name}")
        axes[idx].grid(axis="x", alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Feature comparison saved to {save_path}")

    plt.show()


def plot_prediction_distribution(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model_name: str,
    save_path: str = None,
    figsize=(10, 6),
):
    """
    Plot distribution of prediction probabilities

    Args:
        y_true: True labels
        y_proba: Prediction probabilities
        model_name: Name of the model
        save_path: Optional path to save the plot
        figsize: Figure size
    """
    plt.figure(figsize=figsize)

    # Get probabilities for positive class
    if y_proba.ndim == 2:
        proba_fake = y_proba[:, 1]
    else:
        proba_fake = y_proba

    # Plot distributions
    plt.hist(
        proba_fake[y_true == 0], bins=50, alpha=0.6, label="Real News", color="green"
    )
    plt.hist(
        proba_fake[y_true == 1], bins=50, alpha=0.6, label="Fake News", color="red"
    )

    plt.axvline(
        x=0.5, color="black", linestyle="--", linewidth=2, label="Decision Threshold"
    )
    plt.xlabel("Predicted Probability (Fake)")
    plt.ylabel("Frequency")
    plt.title(f"{model_name} - Prediction Distribution")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Prediction distribution saved to {save_path}")

    plt.show()


def plot_error_analysis(
    texts: list,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    n_samples: int = 5,
):
    """
    Display misclassified examples

    Args:
        texts: List of text samples
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model
        n_samples: Number of samples to display
    """
    # Find misclassified samples
    errors = y_true != y_pred
    error_indices = np.where(errors)[0]

    if len(error_indices) == 0:
        print(f"\n{model_name}: No misclassifications found!")
        return

    print(f"\n{'=' * 80}")
    print(f"{model_name} - Error Analysis")
    print(
        f"Total Errors: {len(error_indices)} / {len(y_true)} "
        f"({100 * len(error_indices) / len(y_true):.2f}%)"
    )
    print(f"{'=' * 80}\n")

    # Sample random errors
    sample_indices = np.random.choice(
        error_indices, min(n_samples, len(error_indices)), replace=False
    )

    for i, idx in enumerate(sample_indices, 1):
        text = texts[idx][:300] + "..." if len(texts[idx]) > 300 else texts[idx]
        true_label = "Fake" if y_true[idx] == 1 else "Real"
        pred_label = "Fake" if y_pred[idx] == 1 else "Real"

        print(f"Error {i}:")
        print(f"Text: {text}")
        print(f"True Label: {true_label}")
        print(f"Predicted Label: {pred_label}")
        print(f"{'-' * 80}\n")


def create_performance_report(results_list: list, save_path: str = None):
    """
    Create comprehensive performance report

    Args:
        results_list: List of result dictionaries from evaluate_model
        save_path: Optional path to save the report
    """
    report = []
    report.append("=" * 80)
    report.append("FAKE NEWS DETECTION - PERFORMANCE REPORT")
    report.append("=" * 80)
    report.append("")

    for result in results_list:
        report.append(f"\nModel: {result['model_name']}")
        report.append("-" * 60)
        report.append(f"Accuracy:  {result['accuracy']:.4f}")
        report.append(f"Precision: {result['precision']:.4f}")
        report.append(f"Recall:    {result['recall']:.4f}")
        report.append(f"F1-Score:  {result['f1_score']:.4f}")
        report.append("")

    report.append("=" * 80)

    # Find best model
    best_model = max(results_list, key=lambda x: x["f1_score"])
    report.append(
        f"\nBest Model: {best_model['model_name']} "
        f"(F1-Score: {best_model['f1_score']:.4f})"
    )
    report.append("=" * 80)

    report_text = "\n".join(report)
    print(report_text)

    if save_path:
        with open(save_path, "w") as f:
            f.write(report_text)
        print(f"\nPerformance report saved to {save_path}")

    return report_text


if __name__ == "__main__":
    set_style()
    print("Visualization module loaded successfully!")
