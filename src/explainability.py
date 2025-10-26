"""
Explainability Module using LIME and SHAP
Provides interpretable explanations for model predictions
"""

import numpy as np
import pandas as pd
import lime
import lime.lime_text
import shap
from typing import Any, List, Callable
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


class ModelExplainer:
    """
    Explainability wrapper for fake news detection models
    Supports both LIME and SHAP explanations
    """

    def __init__(
        self, model: Any, vectorizer: Any = None, class_names: List[str] = None
    ):
        """
        Initialize explainer

        Args:
            model: Trained model (ML or DL)
            vectorizer: Text vectorizer (for ML models)
            class_names: Names of output classes
        """
        self.model = model
        self.vectorizer = vectorizer
        self.class_names = class_names or ["Real", "Fake"]

    def _predict_proba_ml(self, texts: List[str]) -> np.ndarray:
        """
        Prediction function for ML models

        Args:
            texts: List of text strings

        Returns:
            Prediction probabilities
        """
        if self.vectorizer is None:
            raise ValueError("Vectorizer is required for ML models")

        X = self.vectorizer.transform(texts)

        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        elif hasattr(self.model, "decision_function"):
            # For SVM with decision_function
            decisions = self.model.decision_function(X)
            # Convert to probabilities using sigmoid
            proba = 1 / (1 + np.exp(-decisions))
            return np.column_stack([1 - proba, proba])
        else:
            raise AttributeError("Model doesn't support probability prediction")

    def explain_with_lime(
        self, text: str, num_features: int = 10, num_samples: int = 5000
    ) -> lime.lime_text.LimeTextExplainer:
        """
        Generate LIME explanation for a text

        Args:
            text: Input text to explain
            num_features: Number of features to include in explanation
            num_samples: Number of samples for LIME

        Returns:
            LIME explanation object
        """
        print(f"\nGenerating LIME explanation for text: '{text[:100]}...'")

        # Create LIME explainer
        explainer = lime.lime_text.LimeTextExplainer(
            class_names=self.class_names, split_expression=r"\W+", random_state=42
        )

        # Generate explanation
        explanation = explainer.explain_instance(
            text,
            self._predict_proba_ml,
            num_features=num_features,
            num_samples=num_samples,
        )

        return explanation

    def visualize_lime_explanation(
        self, explanation, save_path: str = None, figsize=(10, 6)
    ):
        """
        Visualize LIME explanation

        Args:
            explanation: LIME explanation object
            save_path: Optional path to save the plot
            figsize: Figure size
        """
        # Get explanation for the predicted class
        pred_class = explanation.available_labels()[0]

        fig = explanation.as_pyplot_figure(label=pred_class)
        plt.title("LIME Explanation - Feature Importance")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"LIME visualization saved to {save_path}")

        plt.show()

    def get_lime_top_features(self, explanation, top_n: int = 10) -> pd.DataFrame:
        """
        Get top features from LIME explanation

        Args:
            explanation: LIME explanation object
            top_n: Number of top features

        Returns:
            DataFrame with feature names and weights
        """
        # Get explanation for predicted class
        pred_class = explanation.available_labels()[0]
        features = explanation.as_list(label=pred_class)[:top_n]

        df = pd.DataFrame(features, columns=["Feature", "Weight"])
        df["Abs_Weight"] = df["Weight"].abs()
        df = df.sort_values("Abs_Weight", ascending=False)

        return df

    def explain_with_shap(self, texts: List[str], background_samples: int = 100):
        """
        Generate SHAP explanations for texts

        Args:
            texts: List of input texts
            background_samples: Number of background samples for SHAP

        Returns:
            SHAP values and explainer
        """
        print(f"\nGenerating SHAP explanations for {len(texts)} texts...")

        # For text data with ML models, we use KernelExplainer
        # Create background dataset
        if isinstance(texts, list):
            background_texts = texts[: min(background_samples, len(texts))]
        else:
            background_texts = texts.tolist()[:background_samples]

        # Transform background data
        X_background = self.vectorizer.transform(background_texts)

        # Define prediction function
        def predict_fn(X):
            if hasattr(self.model, "predict_proba"):
                return self.model.predict_proba(X)
            else:
                decisions = self.model.decision_function(X)
                proba = 1 / (1 + np.exp(-decisions))
                return np.column_stack([1 - proba, proba])

        # Create SHAP explainer
        explainer = shap.KernelExplainer(predict_fn, X_background, link="identity")

        # Transform test data
        X_test = self.vectorizer.transform(
            texts if isinstance(texts, list) else [texts]
        )

        # Calculate SHAP values
        shap_values = explainer.shap_values(X_test)

        print("SHAP explanation complete!")
        return shap_values, explainer

    def visualize_shap_summary(
        self,
        shap_values,
        X,
        feature_names: List[str] = None,
        save_path: str = None,
        max_display: int = 20,
    ):
        """
        Create SHAP summary plot

        Args:
            shap_values: SHAP values array
            X: Feature matrix
            feature_names: Names of features
            save_path: Optional path to save the plot
            max_display: Maximum number of features to display
        """
        plt.figure(figsize=(10, 8))

        if feature_names is None and self.vectorizer is not None:
            try:
                feature_names = self.vectorizer.get_feature_names_out()
            except AttributeError:
                feature_names = self.vectorizer.get_feature_names()

        # For binary classification, use values for positive class
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values_to_plot = shap_values[1]
        else:
            shap_values_to_plot = shap_values

        shap.summary_plot(
            shap_values_to_plot,
            X,
            feature_names=feature_names,
            max_display=max_display,
            show=False,
        )

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"SHAP summary plot saved to {save_path}")

        plt.show()

    def visualize_shap_force_plot(
        self, shap_values, explainer, instance_idx: int = 0, save_path: str = None
    ):
        """
        Create SHAP force plot for a single instance

        Args:
            shap_values: SHAP values
            explainer: SHAP explainer object
            instance_idx: Index of instance to explain
            save_path: Optional path to save the plot
        """
        # For binary classification, use values for positive class
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values_to_plot = shap_values[1][instance_idx]
            expected_value = explainer.expected_value[1]
        else:
            shap_values_to_plot = shap_values[instance_idx]
            expected_value = explainer.expected_value

        # Create force plot
        shap.force_plot(
            expected_value, shap_values_to_plot, matplotlib=True, show=False
        )

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"SHAP force plot saved to {save_path}")

        plt.show()

    def get_shap_top_features(
        self,
        shap_values,
        feature_names: List[str] = None,
        instance_idx: int = 0,
        top_n: int = 10,
    ) -> pd.DataFrame:
        """
        Get top features from SHAP values

        Args:
            shap_values: SHAP values array
            feature_names: Names of features
            instance_idx: Index of instance
            top_n: Number of top features

        Returns:
            DataFrame with feature names and SHAP values
        """
        if feature_names is None and self.vectorizer is not None:
            try:
                feature_names = self.vectorizer.get_feature_names_out()
            except AttributeError:
                feature_names = self.vectorizer.get_feature_names()

        # For binary classification, use values for positive class
        if isinstance(shap_values, list) and len(shap_values) == 2:
            values = shap_values[1][instance_idx]
        else:
            values = shap_values[instance_idx]

        # Get indices of top features by absolute value
        if hasattr(values, "toarray"):
            values = values.toarray().flatten()

        indices = np.argsort(np.abs(values))[::-1][:top_n]

        df = pd.DataFrame(
            {
                "Feature": [feature_names[i] for i in indices],
                "SHAP Value": values[indices],
                "Abs SHAP Value": np.abs(values[indices]),
            }
        )

        return df

    def compare_explanations(
        self, text: str, num_features: int = 10, save_path: str = None
    ):
        """
        Compare LIME and SHAP explanations side by side

        Args:
            text: Input text to explain
            num_features: Number of features to show
            save_path: Optional path to save comparison
        """
        print(f"\n{'=' * 60}")
        print("Comparing LIME and SHAP Explanations")
        print(f"{'=' * 60}")

        # LIME explanation
        lime_exp = self.explain_with_lime(text, num_features=num_features)
        lime_features = self.get_lime_top_features(lime_exp, top_n=num_features)

        print("\nLIME Top Features:")
        print(lime_features.to_string(index=False))

        # Note: SHAP for single instance requires background samples
        print(
            "\nNote: For detailed SHAP comparison, use explain_with_shap() with multiple samples"
        )

        return lime_exp, lime_features


def create_explainer(
    model: Any, model_type: str, vectorizer: Any = None
) -> ModelExplainer:
    """
    Factory function to create appropriate explainer

    Args:
        model: Trained model
        model_type: Type of model ('ml' or 'dl')
        vectorizer: Vectorizer for ML models

    Returns:
        ModelExplainer instance
    """
    if model_type == "ml":
        if vectorizer is None:
            raise ValueError("Vectorizer is required for ML models")
        return ModelExplainer(model, vectorizer)
    elif model_type == "dl":
        # For DL models, might need different approach
        # This is a simplified version
        return ModelExplainer(model, vectorizer)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    print("Explainability module loaded successfully!")
    print(f"LIME version: {lime.__version__}")
    print(f"SHAP version: {shap.__version__}")
