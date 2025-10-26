#!/usr/bin/env python3
"""
Main Training and Evaluation Script for Fake News Detection
Trains ML and DL models with LIME & SHAP explainability
"""

import os
import argparse
import numpy as np
import warnings

warnings.filterwarnings("ignore")

from src.data_preprocessing import DataPreprocessor, create_sample_dataset
from src.ml_models import FakeNewsMLModels, get_default_param_grids
from src.utils import (
    create_directories,
    save_model,
    save_results,
    evaluate_model,
    plot_confusion_matrix,
    plot_roc_curve,
    compare_models,
    set_plotting_style,
)
from src.explainability import ModelExplainer
from src.visualization import (
    plot_word_cloud,
    plot_text_length_distribution,
    plot_multiple_roc_curves,
    plot_multiple_confusion_matrices,
    create_performance_report,
)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Train and evaluate fake news detection models"
    )

    parser.add_argument("--data", type=str, help="Path to dataset CSV file")

    parser.add_argument(
        "--models",
        type=str,
        default="all",
        choices=["lr", "rf", "svm", "all"],
        help="Models to train (lr=Logistic Regression, rf=Random Forest, svm=SVM)",
    )

    parser.add_argument(
        "--test-size", type=float, default=0.2, help="Test set size (default: 0.2)"
    )

    parser.add_argument(
        "--val-size",
        type=float,
        default=0.1,
        help="Validation set size from training data (default: 0.1)",
    )

    parser.add_argument(
        "--explain", action="store_true", help="Generate LIME & SHAP explanations"
    )

    parser.add_argument(
        "--tune", action="store_true", help="Perform hyperparameter tuning"
    )

    parser.add_argument(
        "--create-sample", action="store_true", help="Create sample dataset for testing"
    )

    parser.add_argument(
        "--output-dir", type=str, default="results", help="Output directory for results"
    )

    return parser.parse_args()


def main():
    """Main execution function"""
    args = parse_arguments()

    print("=" * 80)
    print("EXPLAINABLE FAKE NEWS DETECTION")
    print("Machine Learning & Deep Learning with LIME & SHAP")
    print("=" * 80)

    # Create directories
    create_directories()
    set_plotting_style()

    # Create sample dataset if requested
    if args.create_sample:
        print("\n[Step 1/7] Creating sample dataset...")
        create_sample_dataset("data/raw/sample_news.csv", n_samples=2000)
        if args.data is None:
            args.data = "data/raw/sample_news.csv"

    # Check if data file exists
    if args.data is None:
        print("\nError: No dataset specified!")
        print("Use --data to specify a dataset or --create-sample to generate one")
        return

    if not os.path.exists(args.data):
        print(f"\nError: Data file not found: {args.data}")
        return

    # Step 1: Load and preprocess data
    print(f"\n[Step 1/7] Loading and preprocessing data...")
    print("-" * 80)

    preprocessor = DataPreprocessor(max_features=5000)

    # Load data
    df = preprocessor.load_data(args.data)

    # Visualize data statistics
    print("\n📊 Generating data visualizations...")
    plot_text_length_distribution(
        df, save_path=f"{args.output_dir}/plots/text_length_distribution.png"
    )

    # Generate word clouds
    plot_word_cloud(
        df["text"].tolist(),
        df["label"].values,
        label_value=0,
        title="Word Cloud - Real News",
        save_path=f"{args.output_dir}/plots/wordcloud_real.png",
    )

    plot_word_cloud(
        df["text"].tolist(),
        df["label"].values,
        label_value=1,
        title="Word Cloud - Fake News",
        save_path=f"{args.output_dir}/plots/wordcloud_fake.png",
    )

    # Preprocess text
    df = preprocessor.preprocess_dataframe(
        df, save_path="data/processed/preprocessed_data.csv"
    )

    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = (
        preprocessor.prepare_train_test_split(
            df, test_size=args.test_size, val_size=args.val_size
        )
    )

    # Create TF-IDF features
    X_train_tfidf, X_val_tfidf, X_test_tfidf = preprocessor.create_tfidf_features(
        X_train, X_val, X_test, save_path="data/processed/tfidf_vectorizer.pkl"
    )

    # Step 2: Train ML models
    print(f"\n[Step 2/7] Training Machine Learning models...")
    print("-" * 80)

    ml_models = FakeNewsMLModels(random_state=42)

    models_to_train = []
    if args.models == "all":
        models_to_train = ["logistic_regression", "random_forest", "svm"]
    elif args.models == "lr":
        models_to_train = ["logistic_regression"]
    elif args.models == "rf":
        models_to_train = ["random_forest"]
    elif args.models == "svm":
        models_to_train = ["svm"]

    # Create and train models
    for model_name in models_to_train:
        if model_name == "logistic_regression":
            ml_models.create_logistic_regression()
        elif model_name == "random_forest":
            ml_models.create_random_forest()
        elif model_name == "svm":
            ml_models.create_svm()

        # Hyperparameter tuning if requested
        if args.tune:
            param_grids = get_default_param_grids()
            ml_models.hyperparameter_tuning(
                model_name,
                X_train_tfidf,
                y_train,
                param_grid=param_grids[model_name],
                cv=3,
            )
        else:
            ml_models.train_model(
                model_name, X_train_tfidf, y_train, X_val_tfidf, y_val
            )

        # Save model
        save_model(ml_models.get_model(model_name), f"models/{model_name}.pkl")

    # Step 3: Evaluate models
    print(f"\n[Step 3/7] Evaluating models on test set...")
    print("-" * 80)

    results_list = []
    roc_data = {}
    cm_data = {}

    for model_name in models_to_train:
        print(f"\n📈 Evaluating {model_name}...")

        # Predictions
        y_pred = ml_models.predict(model_name, X_test_tfidf)
        y_proba = ml_models.predict_proba(model_name, X_test_tfidf)

        # Evaluate
        results = evaluate_model(y_true=y_test, y_pred=y_pred, model_name=model_name)
        results_list.append(results)

        # Save for comparison plots
        roc_data[model_name] = (y_test, y_proba[:, 1] if y_proba.ndim == 2 else y_proba)
        cm_data[model_name] = (y_test, y_pred)

        # Save individual results
        save_results(results, f"{args.output_dir}/reports/{model_name}_results.json")

        # Plot confusion matrix
        cm = np.array(results["confusion_matrix"])
        plot_confusion_matrix(
            cm,
            model_name,
            save_path=f"{args.output_dir}/plots/{model_name}_confusion_matrix.png",
        )

        # Plot ROC curve
        plot_roc_curve(
            y_test,
            y_proba[:, 1] if y_proba.ndim == 2 else y_proba,
            model_name,
            save_path=f"{args.output_dir}/plots/{model_name}_roc_curve.png",
        )

    # Step 4: Compare models
    print(f"\n[Step 4/7] Comparing model performance...")
    print("-" * 80)

    compare_models(
        results_list, save_path=f"{args.output_dir}/plots/model_comparison.png"
    )

    plot_multiple_roc_curves(
        roc_data, save_path=f"{args.output_dir}/plots/roc_curves_comparison.png"
    )

    plot_multiple_confusion_matrices(
        cm_data, save_path=f"{args.output_dir}/plots/confusion_matrices_comparison.png"
    )

    # Create performance report
    create_performance_report(
        results_list, save_path=f"{args.output_dir}/reports/performance_report.txt"
    )

    # Step 5: Generate explanations with LIME & SHAP
    if args.explain:
        print(f"\n[Step 5/7] Generating LIME & SHAP explanations...")
        print("-" * 80)

        # Select best model for explanations
        best_model_result = max(results_list, key=lambda x: x["f1_score"])
        best_model_name = best_model_result["model_name"]
        best_model = ml_models.get_model(best_model_name)

        print(f"\n🔍 Using best model for explanations: {best_model_name}")

        # Create explainer
        explainer = ModelExplainer(
            model=best_model,
            vectorizer=preprocessor.vectorizer,
            class_names=["Real", "Fake"],
        )

        # Select sample texts for explanation
        sample_indices = np.random.choice(
            len(X_test), min(5, len(X_test)), replace=False
        )

        for i, idx in enumerate(sample_indices, 1):
            text = X_test[idx]
            true_label = "Fake" if y_test[idx] == 1 else "Real"

            print(f"\n--- Sample {i} ---")
            print(f"Text: {text[:200]}...")
            print(f"True Label: {true_label}")

            # LIME explanation
            try:
                lime_exp = explainer.explain_with_lime(text, num_features=10)
                explainer.visualize_lime_explanation(
                    lime_exp,
                    save_path=f"{args.output_dir}/plots/lime_explanation_sample{i}.png",
                )

                lime_features = explainer.get_lime_top_features(lime_exp)
                print("\nLIME Top Features:")
                print(lime_features.to_string(index=False))
            except Exception as e:
                print(f"LIME explanation failed: {e}")

        # SHAP explanation (on subset)
        try:
            print("\n\n🔍 Generating SHAP explanations...")
            shap_sample_size = min(100, len(X_test))
            shap_indices = np.random.choice(
                len(X_test), shap_sample_size, replace=False
            )
            X_test_sample = [X_test[i] for i in shap_indices]

            shap_values, shap_explainer = explainer.explain_with_shap(
                X_test_sample, background_samples=50
            )

            # Transform for visualization
            X_test_tfidf_sample = preprocessor.vectorizer.transform(X_test_sample)

            explainer.visualize_shap_summary(
                shap_values,
                X_test_tfidf_sample,
                save_path=f"{args.output_dir}/plots/shap_summary.png",
                max_display=20,
            )

            print("SHAP explanations generated successfully!")
        except Exception as e:
            print(f"SHAP explanation failed: {e}")
    else:
        print(f"\n[Step 5/7] Skipping explanations (use --explain to enable)...")

    # Step 6: Feature importance analysis
    print(f"\n[Step 6/7] Analyzing feature importance...")
    print("-" * 80)

    from src.utils import get_feature_importance, plot_feature_importance

    for model_name in models_to_train:
        if model_name != "svm":  # SVM feature importance is complex
            try:
                model = ml_models.get_model(model_name)
                importance_df = get_feature_importance(
                    model, preprocessor.vectorizer, top_n=20
                )

                print(f"\n{model_name} - Top 10 Features:")
                print(importance_df.head(10).to_string(index=False))

                plot_feature_importance(
                    importance_df,
                    model_name,
                    save_path=f"{args.output_dir}/plots/{model_name}_feature_importance.png",
                )
            except Exception as e:
                print(f"Feature importance for {model_name} failed: {e}")

    # Step 7: Summary
    print(f"\n[Step 7/7] Training Complete!")
    print("=" * 80)
    print("\n📁 Results saved to:")
    print(f"   - Models: models/")
    print(f"   - Plots: {args.output_dir}/plots/")
    print(f"   - Reports: {args.output_dir}/reports/")
    print("\n✅ All done! Check the results directory for outputs.")
    print("=" * 80)


if __name__ == "__main__":
    main()
