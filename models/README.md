# Models Directory

This directory contains saved trained models.

## Structure

```
models/
├── logistic_regression.pkl    # Trained Logistic Regression model
├── random_forest.pkl          # Trained Random Forest model
├── svm.pkl                    # Trained SVM model
├── lstm.pt                    # Trained LSTM model (PyTorch)
└── bert/                      # Fine-tuned BERT model
```

## Model Files

Models are saved in pickle format (`.pkl`) for scikit-learn models and PyTorch format (`.pt`) for deep learning models.

### Loading Models

```python
from src.utils import load_model

# Load scikit-learn model
model = load_model('models/random_forest.pkl')

# Make predictions
predictions = model.predict(X_test_tfidf)
```

### Model Sizes

Approximate file sizes:
- Logistic Regression: 1-5 MB
- Random Forest: 10-50 MB
- SVM: 5-20 MB
- LSTM: 10-30 MB
- BERT: 400-500 MB

## Notes

- Models are trained on TF-IDF features for ML models
- Models include the learned parameters and can be used for inference
- Re-train models periodically with new data for best performance
- Keep the corresponding vectorizer file (`tfidf_vectorizer.pkl`) with ML models

## Version Control

Models are excluded from git by default (see `.gitignore`). 

To share models:
1. Upload to cloud storage
2. Use model versioning tools (MLflow, DVC)
3. Document model version and training date
