"""
Traditional Machine Learning Models for Fake News Detection
Implements Logistic Regression, Random Forest, and SVM
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from typing import Tuple, Dict, Any
import warnings

warnings.filterwarnings("ignore")


class FakeNewsMLModels:
    """
    Collection of traditional ML models for fake news detection
    """

    def __init__(self, random_state: int = 42):
        """
        Initialize ML models

        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = {}

    def create_logistic_regression(self, **kwargs) -> LogisticRegression:
        """
        Create Logistic Regression model

        Args:
            **kwargs: Additional parameters for LogisticRegression

        Returns:
            Logistic Regression model
        """
        default_params = {
            "max_iter": 1000,
            "random_state": self.random_state,
            "n_jobs": -1,
        }
        default_params.update(kwargs)

        model = LogisticRegression(**default_params)
        self.models["logistic_regression"] = model
        print("Logistic Regression model created")
        return model

    def create_random_forest(self, **kwargs) -> RandomForestClassifier:
        """
        Create Random Forest model

        Args:
            **kwargs: Additional parameters for RandomForestClassifier

        Returns:
            Random Forest model
        """
        default_params = {
            "n_estimators": 100,
            "max_depth": 20,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "random_state": self.random_state,
            "n_jobs": -1,
        }
        default_params.update(kwargs)

        model = RandomForestClassifier(**default_params)
        self.models["random_forest"] = model
        print("Random Forest model created")
        return model

    def create_svm(self, **kwargs) -> SVC:
        """
        Create SVM model

        Args:
            **kwargs: Additional parameters for SVC

        Returns:
            SVM model
        """
        default_params = {
            "kernel": "rbf",
            "C": 1.0,
            "gamma": "scale",
            "probability": True,
            "random_state": self.random_state,
        }
        default_params.update(kwargs)

        model = SVC(**default_params)
        self.models["svm"] = model
        print("SVM model created")
        return model

    def train_model(
        self, model_name: str, X_train, y_train, X_val=None, y_val=None
    ) -> Any:
        """
        Train a specific model

        Args:
            model_name: Name of the model to train
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)

        Returns:
            Trained model
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Create it first.")

        print(f"\nTraining {model_name}...")
        model = self.models[model_name]
        model.fit(X_train, y_train)

        train_score = model.score(X_train, y_train)
        print(f"Training accuracy: {train_score:.4f}")

        if X_val is not None and y_val is not None:
            val_score = model.score(X_val, y_val)
            print(f"Validation accuracy: {val_score:.4f}")

        print(f"{model_name} training complete!")
        return model

    def predict(self, model_name: str, X) -> np.ndarray:
        """
        Make predictions using a trained model

        Args:
            model_name: Name of the model
            X: Features to predict

        Returns:
            Predictions
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")

        return self.models[model_name].predict(X)

    def predict_proba(self, model_name: str, X) -> np.ndarray:
        """
        Get prediction probabilities

        Args:
            model_name: Name of the model
            X: Features to predict

        Returns:
            Prediction probabilities
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")

        model = self.models[model_name]

        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)
        elif hasattr(model, "decision_function"):
            # For SVM without probability=True
            return model.decision_function(X)
        else:
            raise AttributeError(
                f"Model {model_name} doesn't support probability prediction"
            )

    def hyperparameter_tuning(
        self, model_name: str, X_train, y_train, param_grid: Dict, cv: int = 5
    ) -> Any:
        """
        Perform hyperparameter tuning using GridSearchCV

        Args:
            model_name: Name of the model to tune
            X_train: Training features
            y_train: Training labels
            param_grid: Parameter grid for GridSearchCV
            cv: Number of cross-validation folds

        Returns:
            Best model after tuning
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")

        print(f"\nPerforming hyperparameter tuning for {model_name}...")
        print(f"Parameter grid: {param_grid}")

        model = self.models[model_name]
        grid_search = GridSearchCV(
            model, param_grid, cv=cv, n_jobs=-1, scoring="accuracy", verbose=1
        )

        grid_search.fit(X_train, y_train)

        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

        # Update model with best parameters
        self.models[model_name] = grid_search.best_estimator_

        return grid_search.best_estimator_

    def get_model(self, model_name: str) -> Any:
        """
        Get a trained model

        Args:
            model_name: Name of the model

        Returns:
            The model object
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        return self.models[model_name]

    def train_all_models(self, X_train, y_train, X_val=None, y_val=None) -> Dict:
        """
        Train all available models

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)

        Returns:
            Dictionary of trained models
        """
        print("\n" + "=" * 60)
        print("Training All ML Models")
        print("=" * 60)

        # Create models if not already created
        if "logistic_regression" not in self.models:
            self.create_logistic_regression()
        if "random_forest" not in self.models:
            self.create_random_forest()
        if "svm" not in self.models:
            self.create_svm()

        # Train each model
        for model_name in self.models.keys():
            self.train_model(model_name, X_train, y_train, X_val, y_val)

        print("\n" + "=" * 60)
        print("All models trained successfully!")
        print("=" * 60)

        return self.models


def get_default_param_grids() -> Dict[str, Dict]:
    """
    Get default parameter grids for hyperparameter tuning

    Returns:
        Dictionary of parameter grids for each model
    """
    param_grids = {
        "logistic_regression": {
            "C": [0.1, 1.0, 10.0],
            "penalty": ["l2"],
            "solver": ["lbfgs", "liblinear"],
        },
        "random_forest": {
            "n_estimators": [50, 100, 200],
            "max_depth": [10, 20, 30, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        },
        "svm": {
            "C": [0.1, 1.0, 10.0],
            "kernel": ["linear", "rbf"],
            "gamma": ["scale", "auto"],
        },
    }
    return param_grids


if __name__ == "__main__":
    print("ML Models module loaded successfully!")

    # Example usage
    print("\nExample: Creating models...")
    ml_models = FakeNewsMLModels()
    ml_models.create_logistic_regression()
    ml_models.create_random_forest()
    ml_models.create_svm()

    print("\nAvailable models:", list(ml_models.models.keys()))
