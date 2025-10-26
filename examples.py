#!/usr/bin/env python3
"""
Example Usage Script for Fake News Detection
Demonstrates various ways to use the project modules
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 80)
print("FAKE NEWS DETECTION - USAGE EXAMPLES")
print("=" * 80)

# Example 1: Basic Usage
print("\n" + "=" * 80)
print("Example 1: Basic Model Training and Prediction")
print("=" * 80)

from src.data_preprocessing import DataPreprocessor, create_sample_dataset
from src.ml_models import FakeNewsMLModels
from src.utils import evaluate_model

# Create sample data
print("\n1. Creating sample dataset...")
df = create_sample_dataset("data/raw/example_news.csv", n_samples=500)

# Preprocess
print("\n2. Preprocessing data...")
preprocessor = DataPreprocessor(max_features=1000)
df = preprocessor.preprocess_dataframe(df)

# Split data
print("\n3. Splitting data...")
X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.prepare_train_test_split(
    df, test_size=0.2
)

# Create TF-IDF features
print("\n4. Creating TF-IDF features...")
X_train_tfidf, X_val_tfidf, X_test_tfidf = preprocessor.create_tfidf_features(
    X_train, X_val, X_test
)

# Train model
print("\n5. Training Logistic Regression...")
ml_models = FakeNewsMLModels()
ml_models.create_logistic_regression()
ml_models.train_model("logistic_regression", X_train_tfidf, y_train)

# Evaluate
print("\n6. Evaluating model...")
y_pred = ml_models.predict("logistic_regression", X_test_tfidf)
results = evaluate_model(y_test, y_pred, "Logistic Regression")

print(f"\n✅ Example 1 Complete! Accuracy: {results['accuracy']:.4f}")


# Example 2: Multiple Models with Comparison
print("\n\n" + "=" * 80)
print("Example 2: Training Multiple Models and Comparing")
print("=" * 80)

print("\n1. Training Random Forest...")
ml_models.create_random_forest(n_estimators=50)
ml_models.train_model("random_forest", X_train_tfidf, y_train)

print("\n2. Training SVM...")
ml_models.create_svm()
ml_models.train_model("svm", X_train_tfidf, y_train)

print("\n3. Comparing models...")
from src.utils import compare_models

results_list = []
for model_name in ["logistic_regression", "random_forest", "svm"]:
    y_pred = ml_models.predict(model_name, X_test_tfidf)
    result = evaluate_model(y_test, y_pred, model_name)
    results_list.append(result)

compare_models(results_list, save_path="results/plots/example_comparison.png")

print("\n✅ Example 2 Complete! Comparison plot saved.")


# Example 3: LIME Explanations
print("\n\n" + "=" * 80)
print("Example 3: Generating LIME Explanations")
print("=" * 80)

from src.explainability import ModelExplainer

print("\n1. Creating explainer...")
explainer = ModelExplainer(
    model=ml_models.get_model("random_forest"),
    vectorizer=preprocessor.vectorizer,
    class_names=["Real", "Fake"],
)

print("\n2. Explaining a sample prediction...")
sample_text = X_test[0]
print(f"Sample text: {sample_text[:100]}...")

lime_exp = explainer.explain_with_lime(sample_text, num_features=8)

print("\n3. Getting top features...")
lime_features = explainer.get_lime_top_features(lime_exp, top_n=5)
print("\nTop 5 LIME Features:")
print(lime_features.to_string(index=False))

explainer.visualize_lime_explanation(
    lime_exp, save_path="results/plots/example_lime.png"
)

print("\n✅ Example 3 Complete! LIME explanation saved.")


# Example 4: Feature Importance
print("\n\n" + "=" * 80)
print("Example 4: Analyzing Feature Importance")
print("=" * 80)

from src.utils import get_feature_importance, plot_feature_importance

print("\n1. Getting feature importance from Random Forest...")
rf_model = ml_models.get_model("random_forest")
importance_df = get_feature_importance(rf_model, preprocessor.vectorizer, top_n=15)

print("\n2. Top 10 Important Features:")
print(importance_df.head(10).to_string(index=False))

print("\n3. Plotting feature importance...")
plot_feature_importance(
    importance_df,
    "Random Forest",
    save_path="results/plots/example_feature_importance.png",
)

print("\n✅ Example 4 Complete! Feature importance plot saved.")


# Example 5: Visualization
print("\n\n" + "=" * 80)
print("Example 5: Creating Visualizations")
print("=" * 80)

from src.visualization import (
    plot_word_cloud,
    plot_text_length_distribution,
    plot_multiple_confusion_matrices,
)

print("\n1. Creating word clouds...")
plot_word_cloud(
    df["text"].tolist(),
    df["label"].values,
    label_value=0,
    title="Word Cloud - Real News",
    save_path="results/plots/example_wordcloud_real.png",
)

plot_word_cloud(
    df["text"].tolist(),
    df["label"].values,
    label_value=1,
    title="Word Cloud - Fake News",
    save_path="results/plots/example_wordcloud_fake.png",
)

print("\n2. Plotting text length distribution...")
plot_text_length_distribution(df, save_path="results/plots/example_text_length.png")

print("\n3. Creating confusion matrices...")
cm_data = {}
for model_name in ["logistic_regression", "random_forest", "svm"]:
    y_pred = ml_models.predict(model_name, X_test_tfidf)
    cm_data[model_name] = (y_test, y_pred)

plot_multiple_confusion_matrices(
    cm_data, save_path="results/plots/example_confusion_matrices.png"
)

print("\n✅ Example 5 Complete! All visualizations saved.")


# Example 6: Saving and Loading Models
print("\n\n" + "=" * 80)
print("Example 6: Saving and Loading Models")
print("=" * 80)

from src.utils import save_model, load_model

print("\n1. Saving trained model...")
save_model(rf_model, "models/example_rf_model.pkl")

print("\n2. Loading model...")
loaded_model = load_model("models/example_rf_model.pkl")

print("\n3. Making predictions with loaded model...")
y_pred_loaded = loaded_model.predict(X_test_tfidf)
accuracy = (y_pred_loaded == y_test).mean()
print(f"Loaded model accuracy: {accuracy:.4f}")

print("\n✅ Example 6 Complete! Model saved and loaded successfully.")


# Example 7: Custom Prediction Function
print("\n\n" + "=" * 80)
print("Example 7: Creating Custom Prediction Function")
print("=" * 80)


def predict_news(text, model, preprocessor):
    """
    Predict if a news article is fake or real

    Args:
        text: News article text
        model: Trained model
        preprocessor: DataPreprocessor instance

    Returns:
        tuple: (prediction, probability)
    """
    # Preprocess
    cleaned_text = preprocessor.preprocess_text(text)

    # Vectorize
    text_tfidf = preprocessor.vectorizer.transform([cleaned_text])

    # Predict
    prediction = model.predict(text_tfidf)[0]
    probability = model.predict_proba(text_tfidf)[0]

    label = "FAKE" if prediction == 1 else "REAL"
    confidence = probability[prediction] * 100

    return label, confidence


# Test the function
print("\n1. Testing custom prediction function...")
test_texts = [
    "Scientists discover new method for renewable energy production",
    "You won't believe what this celebrity said! Shocking revelation!",
    "Government announces new policy on education funding",
]

print("\n2. Making predictions:")
for i, text in enumerate(test_texts, 1):
    label, confidence = predict_news(text, rf_model, preprocessor)
    print(f"\nSample {i}:")
    print(f"Text: {text}")
    print(f"Prediction: {label} (Confidence: {confidence:.2f}%)")

print("\n✅ Example 7 Complete! Custom prediction function works.")


# Example 8: Batch Processing
print("\n\n" + "=" * 80)
print("Example 8: Batch Processing Multiple Articles")
print("=" * 80)

import pandas as pd


def batch_predict(texts, model, preprocessor):
    """
    Predict multiple texts at once

    Args:
        texts: List of text strings
        model: Trained model
        preprocessor: DataPreprocessor instance

    Returns:
        DataFrame with predictions
    """
    results = []

    for text in texts:
        label, confidence = predict_news(text, model, preprocessor)
        results.append(
            {
                "text": text[:50] + "..." if len(text) > 50 else text,
                "prediction": label,
                "confidence": f"{confidence:.2f}%",
            }
        )

    return pd.DataFrame(results)


print("\n1. Processing batch of articles...")
batch_texts = [
    "Research shows positive correlation between exercise and health",
    "Secret government conspiracy revealed by anonymous source!",
    "Local school wins state championship in mathematics competition",
    "Miracle cure for all diseases discovered but hidden by big pharma",
    "Economic indicators suggest steady growth in manufacturing sector",
]

results_df = batch_predict(batch_texts, rf_model, preprocessor)

print("\n2. Batch Prediction Results:")
print(results_df.to_string(index=False))

print("\n✅ Example 8 Complete! Batch processing successful.")


# Summary
print("\n\n" + "=" * 80)
print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
print("=" * 80)
print("\n📁 Output files saved to:")
print("   - results/plots/example_*.png")
print("   - models/example_rf_model.pkl")
print("\n📊 You can now:")
print("   1. Check the generated plots in results/plots/")
print("   2. Use the saved model for predictions")
print("   3. Modify these examples for your own use case")
print("   4. Explore the full API in docs/API.md")
print("\n" + "=" * 80)


if __name__ == "__main__":
    print("\n✨ Script execution complete!")
    print("Run this script to see all examples in action:")
    print("   python examples.py")
