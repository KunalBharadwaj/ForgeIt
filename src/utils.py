"""
Utility Functions for the Fake News Detection Project
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import warnings

warnings.filterwarnings("ignore")


def create_directories():
    """Create necessary project directories"""
    directories = [
        "data/raw",
        "data/processed",
        "models",
        "results/plots",
        "results/reports",
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("Project directories created successfully!")


def save_model(model: Any, filepath: str):
    """
    Save a trained model to disk

    Args:
        model: Trained model object
        filepath: Path to save the model
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {filepath}")


def load_model(filepath: str) -> Any:
    """
    Load a trained model from disk

    Args:
        filepath: Path to the saved model

    Returns:
        Loaded model object
    """
    with open(filepath, "rb") as f:
        model = pickle.load(f)
    print(f"Model loaded from {filepath}")
    return model


def save_results(results: Dict, filepath: str):
    """
    Save evaluation results to JSON file

    Args:
        results: Dictionary containing results
        filepath: Path to save the results
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Convert numpy types to native Python types
    def convert_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        return obj

    results = convert_types(results)

    with open(filepath, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {filepath}")


def load_results(filepath: str) -> Dict:
    """
    Load evaluation results from JSON file

    Args:
        filepath: Path to the results file

    Returns:
        Dictionary containing results
    """
    with open(filepath, "r") as f:
        results = json.load(f)
    return results


def evaluate_model(
    y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Model"
) -> Dict:
    """
    Evaluate model performance with various metrics

    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model

    Returns:
        Dictionary containing evaluation metrics
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    results = {
        "model_name": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="binary"),
        "recall": recall_score(y_true, y_pred, average="binary"),
        "f1_score": f1_score(y_true, y_pred, average="binary"),
        "classification_report": classification_report(
            y_true, y_pred, target_names=["Real", "Fake"]
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    print(f"\n{'=' * 50}")
    print(f"{model_name} Performance Metrics")
    print(f"{'=' * 50}")
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1-Score:  {results['f1_score']:.4f}")
    print(f"\nClassification Report:")
    print(results["classification_report"])

    return results


def plot_confusion_matrix(
    cm: np.ndarray, model_name: str, save_path: str = None, figsize=(8, 6)
):
    """
    Plot confusion matrix

    Args:
        cm: Confusion matrix
        model_name: Name of the model
        save_path: Optional path to save the plot
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Real", "Fake"],
        yticklabels=["Real", "Fake"],
    )
    plt.title(f"Confusion Matrix - {model_name}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Confusion matrix saved to {save_path}")

    plt.show()


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model_name: str,
    save_path: str = None,
    figsize=(8, 6),
):
    """
    Plot ROC curve

    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        model_name: Name of the model
        save_path: Optional path to save the plot
        figsize: Figure size
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=figsize)
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})"
    )
    plt.plot(
        [0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Classifier"
    )
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"ROC curve saved to {save_path}")

    plt.show()

    return roc_auc


def compare_models(results_list: List[Dict], save_path: str = None, figsize=(12, 6)):
    """
    Compare multiple models' performance

    Args:
        results_list: List of result dictionaries from evaluate_model
        save_path: Optional path to save the plot
        figsize: Figure size
    """
    metrics = ["accuracy", "precision", "recall", "f1_score"]
    model_names = [r["model_name"] for r in results_list]

    data = {metric: [r[metric] for r in results_list] for metric in metrics}

    x = np.arange(len(model_names))
    width = 0.2

    fig, ax = plt.subplots(figsize=figsize)

    for i, metric in enumerate(metrics):
        ax.bar(
            x + i * width,
            data[metric],
            width,
            label=metric.capitalize().replace("_", "-"),
        )

    ax.set_xlabel("Models")
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(model_names, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Comparison plot saved to {save_path}")

    plt.show()


def print_sample_predictions(
    texts: List[str], y_true: np.ndarray, y_pred: np.ndarray, n_samples: int = 5
):
    """
    Print sample predictions for inspection

    Args:
        texts: Original text samples
        y_true: True labels
        y_pred: Predicted labels
        n_samples: Number of samples to display
    """
    print(f"\n{'=' * 80}")
    print("Sample Predictions")
    print(f"{'=' * 80}\n")

    indices = np.random.choice(len(texts), min(n_samples, len(texts)), replace=False)

    for i, idx in enumerate(indices, 1):
        text = texts[idx][:200] + "..." if len(texts[idx]) > 200 else texts[idx]
        true_label = "Fake" if y_true[idx] == 1 else "Real"
        pred_label = "Fake" if y_pred[idx] == 1 else "Real"
        correct = "✓" if y_true[idx] == y_pred[idx] else "✗"

        print(f"Sample {i} {correct}")
        print(f"Text: {text}")
        print(f"True: {true_label} | Predicted: {pred_label}")
        print(f"{'-' * 80}\n")


def get_feature_importance(model, vectorizer, top_n: int = 20) -> pd.DataFrame:
    """
    Get top important features from a model

    Args:
        model: Trained model with feature_importances_ or coef_ attribute
        vectorizer: Fitted vectorizer with feature names
        top_n: Number of top features to return

    Returns:
        DataFrame with feature names and importance scores
    """
    try:
        feature_names = vectorizer.get_feature_names_out()
    except AttributeError:
        feature_names = vectorizer.get_feature_names()

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        raise AttributeError(
            "Model doesn't have feature_importances_ or coef_ attribute"
        )

    indices = np.argsort(importances)[::-1][:top_n]

    df = pd.DataFrame(
        {
            "feature": [feature_names[i] for i in indices],
            "importance": importances[indices],
        }
    )

    return df


def plot_feature_importance(
    importance_df: pd.DataFrame, model_name: str, save_path: str = None, figsize=(10, 8)
):
    """
    Plot feature importance

    Args:
        importance_df: DataFrame with 'feature' and 'importance' columns
        model_name: Name of the model
        save_path: Optional path to save the plot
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    plt.barh(range(len(importance_df)), importance_df["importance"])
    plt.yticks(range(len(importance_df)), importance_df["feature"])
    plt.xlabel("Importance")
    plt.title(f"Top Feature Importance - {model_name}")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Feature importance plot saved to {save_path}")

    plt.show()


def set_plotting_style():
    """Set consistent plotting style"""
    plt.style.use("seaborn-v0_8-darkgrid")
    sns.set_palette("husl")
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["font.size"] = 10


if __name__ == "__main__":
    # Create project directories
    create_directories()
    print("Utilities module loaded successfully!")
