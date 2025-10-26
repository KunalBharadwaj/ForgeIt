# Results Directory

This directory contains all outputs from model training and evaluation.

## Structure

```
results/
├── plots/                      # All visualization outputs
│   ├── confusion_matrices_comparison.png
│   ├── roc_curves_comparison.png
│   ├── model_comparison.png
│   ├── wordcloud_real.png
│   ├── wordcloud_fake.png
│   ├── text_length_distribution.png
│   ├── lime_explanation_*.png
│   ├── shap_summary.png
│   └── *_feature_importance.png
│
└── reports/                    # Performance metrics and reports
    ├── performance_report.txt
    ├── logistic_regression_results.json
    ├── random_forest_results.json
    └── svm_results.json
```

## Plot Types

### Model Performance
- **Confusion Matrices**: Shows true vs predicted labels
- **ROC Curves**: Receiver Operating Characteristic with AUC scores
- **Model Comparison**: Bar charts comparing all models

### Data Exploration
- **Word Clouds**: Most frequent words in real/fake news
- **Text Length Distribution**: Histogram and box plots of text lengths

### Explainability
- **LIME Explanations**: Word-level feature importance
- **SHAP Summary**: Global feature importance
- **Feature Importance**: Most important features per model

## Report Files

### JSON Format
Contains detailed metrics:
```json
{
  "model_name": "Random Forest",
  "accuracy": 0.9412,
  "precision": 0.9356,
  "recall": 0.9468,
  "f1_score": 0.9412,
  "confusion_matrix": [[...], [...]],
  "classification_report": "..."
}
```

### TXT Format
Human-readable performance summary with all models.

## Accessing Results

### Programmatically

```python
from src.utils import load_results

# Load results
results = load_results('results/reports/random_forest_results.json')
print(f"Accuracy: {results['accuracy']:.4f}")
```

### Viewing Plots

Open PNG files directly or display in notebook:

```python
from IPython.display import Image, display

display(Image('results/plots/model_comparison.png'))
```

## Notes

- Plots are saved at 300 DPI for publication quality
- All results include timestamps in filenames
- Results are excluded from git by default
- Re-run experiments to update results
