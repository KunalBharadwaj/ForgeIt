# Quick Start Guide

## 🚀 Getting Started with Fake News Detection

This guide will help you get the project running in 5 minutes!

---

## Prerequisites

- Python 3.8 or higher
- pip package manager
- 4GB RAM minimum (8GB recommended)
- (Optional) GPU for deep learning models

---

## Installation Steps

### 1. Navigate to Project Directory

```bash
cd /home/kunal/Desktop/AIProject
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Linux/Mac
```

### 3. Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet')"
```

This will install:
- NumPy, Pandas, scikit-learn (ML libraries)
- PyTorch, Transformers (Deep Learning)
- LIME, SHAP (Explainability)
- Matplotlib, Seaborn (Visualization)
- Jupyter (Interactive notebooks)

---

## Quick Test Run

### Option 1: Run with Sample Dataset (Recommended for First Time)

```bash
# Create sample dataset and train all models
python main.py --create-sample --models all --explain
```

This will:
1. Create a sample dataset (2000 articles)
2. Train Logistic Regression, Random Forest, and SVM
3. Evaluate models with metrics and plots
4. Generate LIME & SHAP explanations
5. Save results to `results/` directory

**Expected runtime**: 5-10 minutes

### Option 2: Use Your Own Dataset

```bash
# Train with your dataset
python main.py --data path/to/your/news.csv --models all --explain
```

Your CSV should have columns:
- `text` (or `content`, `article`): The news article text
- `label`: Binary label (0=Real, 1=Fake)

### Option 3: Interactive Jupyter Notebook

```bash
# Launch Jupyter notebook
jupyter notebook notebooks/demo.ipynb
```

Then run cells sequentially to see:
- Data exploration and visualization
- Model training and evaluation
- Interactive LIME & SHAP explanations

---

## Understanding the Output

After running, you'll find:

```
AIProject/
├── data/
│   ├── processed/              # Preprocessed data
│   └── raw/                    # Original dataset
├── models/                     # Saved trained models (.pkl files)
└── results/
    ├── plots/                  # All visualizations
    │   ├── confusion_matrices_comparison.png
    │   ├── roc_curves_comparison.png
    │   ├── model_comparison.png
    │   ├── lime_explanation_*.png
    │   ├── shap_summary.png
    │   └── feature_importance_*.png
    └── reports/                # Performance metrics (.json, .txt)
        └── performance_report.txt
```

---

## Command Line Options

```bash
python main.py [OPTIONS]

Options:
  --data PATH           Path to your CSV dataset
  --models {lr,rf,svm,all}   
                        Models to train:
                        lr  = Logistic Regression
                        rf  = Random Forest
                        svm = Support Vector Machine
                        all = Train all models
  --test-size FLOAT     Test set size (default: 0.2)
  --val-size FLOAT      Validation set size (default: 0.1)
  --explain             Generate LIME & SHAP explanations
  --tune                Perform hyperparameter tuning
  --create-sample       Create sample dataset for testing
  --output-dir PATH     Output directory (default: results)
```

### Examples

**Train only Logistic Regression:**
```bash
python main.py --data data/raw/news.csv --models lr
```

**Train all models with hyperparameter tuning:**
```bash
python main.py --data data/raw/news.csv --models all --tune --explain
```

**Custom train/test split:**
```bash
python main.py --data data/raw/news.csv --models all --test-size 0.3
```

---

## Using Python API

You can also use the modules directly in your Python code:

```python
from src.data_preprocessing import DataPreprocessor
from src.ml_models import FakeNewsMLModels
from src.explainability import ModelExplainer

# Load and preprocess data
preprocessor = DataPreprocessor()
df = preprocessor.load_data('data/raw/news.csv')
df = preprocessor.preprocess_dataframe(df)

# Train model
ml_models = FakeNewsMLModels()
ml_models.create_random_forest()
ml_models.train_model('random_forest', X_train, y_train)

# Generate explanations
explainer = ModelExplainer(model, vectorizer)
lime_exp = explainer.explain_with_lime(text)
lime_exp.show_in_notebook()
```

---

## Downloading Real Datasets

### Kaggle Fake News Dataset

1. Install Kaggle CLI:
```bash
pip install kaggle
```

2. Setup API credentials:
   - Go to https://www.kaggle.com/account
   - Create API token
   - Save `kaggle.json` to `~/.kaggle/`

3. Download dataset:
```bash
kaggle competitions download -c fake-news
unzip fake-news.zip -d data/raw/
```

### ISOT Fake News Dataset

1. Visit: https://www.uvic.ca/engineering/ece/isot/datasets/fake-news/
2. Download True.csv and Fake.csv
3. Place in `data/raw/` directory

### Manual Dataset Format

Create a CSV with these columns:

```csv
text,label
"Article text here...",0
"Another article...",1
```

Where:
- Label 0 = Real/True news
- Label 1 = Fake/False news

---

## Troubleshooting

### Issue: "Import error" or "Module not found"

**Solution**: Make sure virtual environment is activated and dependencies are installed:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Issue: "NLTK data not found"

**Solution**: Download NLTK data:
```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet')"
```

### Issue: "Out of memory" during training

**Solution**: Reduce dataset size or max_features:
```python
preprocessor = DataPreprocessor(max_features=2000)  # Instead of 5000
```

### Issue: SHAP explanations are slow

**Solution**: This is normal. SHAP is computationally intensive. Reduce sample size:
```python
shap_values, explainer = explainer.explain_with_shap(
    texts[:50],  # Use fewer samples
    background_samples=25  # Reduce background samples
)
```

### Issue: Jupyter notebook kernel crashes

**Solution**: Restart kernel and run cells one by one. Avoid running all cells at once for the first time.

---

## Next Steps

Once you have the basic setup working:

1. **Explore the Jupyter Notebook**: `notebooks/demo.ipynb`
   - Interactive visualizations
   - Step-by-step explanations
   - Experiment with different parameters

2. **Try Different Models**:
   - Tune hyperparameters with `--tune` flag
   - Experiment with ensemble methods
   - Try deep learning models (LSTM, BERT)

3. **Analyze Explanations**:
   - Compare LIME vs SHAP
   - Identify common fake news patterns
   - Test on different types of articles

4. **Deploy Your Model**:
   - Create a Flask API
   - Build a web interface
   - Integrate with real-time news feeds

---

## Getting Help

- **Check Documentation**: See `README.md` for detailed information
- **Read Module Docstrings**: Each Python file has detailed documentation
- **Review Examples**: Check `notebooks/demo.ipynb` for working examples
- **Common Issues**: See "Troubleshooting" section above

---

## Performance Expectations

On a typical laptop with sample dataset (2000 articles):

| Task | Time |
|------|------|
| Data preprocessing | 1-2 min |
| Train Logistic Regression | 5-10 sec |
| Train Random Forest | 30-60 sec |
| Train SVM | 1-2 min |
| LIME explanations (5 samples) | 30-60 sec |
| SHAP explanations (50 samples) | 3-5 min |

**Total runtime**: ~5-10 minutes for complete pipeline

---

## Tips for Best Results

1. **Start Small**: Use sample dataset first to verify everything works
2. **Monitor Memory**: Large datasets may require more RAM
3. **Use GPU**: For deep learning models, GPU significantly speeds up training
4. **Preprocess Once**: Save preprocessed data to avoid re-processing
5. **Batch Explanations**: Generate explanations for representative samples, not entire dataset
6. **Version Control**: Track your experiments and results
7. **Document Findings**: Keep notes on what works and what doesn't

---

## Project Structure Overview

```
src/
├── data_preprocessing.py    # Data loading, cleaning, TF-IDF
├── ml_models.py            # Logistic Regression, RF, SVM
├── dl_models.py            # LSTM, BERT models
├── explainability.py       # LIME & SHAP implementations
├── visualization.py        # All plotting functions
└── utils.py                # Helper functions

main.py                     # Command-line interface
notebooks/demo.ipynb        # Interactive demonstration
```

---

## Congratulations! 🎉

You now have a fully functional explainable fake news detection system!

Start by running:
```bash
python main.py --create-sample --models all --explain
```

Then explore `results/plots/` to see your model's performance and explanations.

Happy detecting! 🔍
