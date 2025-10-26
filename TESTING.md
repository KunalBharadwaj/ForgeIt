# Testing Guide

## How to Test the Fake News Detection Project

This guide explains how to test all components of the project.

---

## Quick Test (5 minutes)

The fastest way to verify everything works:

```bash
# 1. Navigate to project
cd /home/kunal/Desktop/AIProject

# 2. Activate virtual environment (if created)
source venv/bin/activate

# 3. Run with sample data
python main.py --create-sample --models lr --explain
```

**Expected Output:**
- Sample dataset created (2000 articles)
- Logistic Regression trained
- Test accuracy ~90%+
- Plots saved to `results/plots/`
- LIME explanations generated

---

## Component Testing

### 1. Data Preprocessing

```bash
python src/data_preprocessing.py --create-sample
```

**Verifies:**
- Sample dataset creation
- Text cleaning
- TF-IDF vectorization
- Train/test splitting

**Expected Output:**
```
Creating sample dataset with 1000 samples...
Label distribution:
0    500
1    500
```

### 2. ML Models

```python
python -c "
from src.ml_models import FakeNewsMLModels
ml = FakeNewsMLModels()
ml.create_logistic_regression()
ml.create_random_forest()
ml.create_svm()
print('✓ All ML models created successfully')
"
```

**Verifies:**
- Model initialization
- No import errors

### 3. Explainability

```python
python -c "
import lime
import shap
print(f'✓ LIME version: {lime.__version__}')
print(f'✓ SHAP version: {shap.__version__}')
"
```

**Verifies:**
- LIME and SHAP installed correctly

### 4. Visualization

```python
python -c "
import matplotlib
import seaborn
from wordcloud import WordCloud
print('✓ All visualization libraries working')
"
```

**Verifies:**
- Plotting libraries functional

---

## Full Pipeline Test

Run complete pipeline with all models:

```bash
python main.py --create-sample --models all --explain --output-dir test_results
```

**This tests:**
1. ✅ Data creation and loading
2. ✅ Text preprocessing
3. ✅ TF-IDF vectorization
4. ✅ Train/validation/test splitting
5. ✅ Training 3 ML models
6. ✅ Model evaluation
7. ✅ Confusion matrices
8. ✅ ROC curves
9. ✅ Model comparison
10. ✅ LIME explanations
11. ✅ SHAP explanations
12. ✅ Feature importance
13. ✅ Results saving

**Expected Runtime:** 5-10 minutes

**Check Output:**
```bash
ls test_results/plots/          # Should have 10+ PNG files
ls test_results/reports/        # Should have JSON and TXT files
ls models/                      # Should have 3 PKL files
```

---

## Interactive Notebook Test

```bash
jupyter notebook notebooks/demo.ipynb
```

**Test by:**
1. Run all cells sequentially
2. Verify no errors
3. Check visualizations appear
4. Confirm metrics are reasonable

**Key Cells to Check:**
- Cell 2: Dataset creation
- Cell 5: Data preprocessing
- Cell 8: Model training
- Cell 12: Model evaluation
- Cell 16: LIME explanations
- Cell 20: SHAP explanations

---

## Example Script Test

```bash
python examples.py
```

**This tests 8 usage examples:**
1. Basic training and prediction
2. Multiple model comparison
3. LIME explanations
4. Feature importance
5. Visualizations
6. Model saving/loading
7. Custom prediction function
8. Batch processing

**Expected Output:**
- All 8 examples complete successfully
- Example plots in `results/plots/`
- Model saved to `models/`

---

## Unit Tests

### Test Data Preprocessing

```python
python -c "
from src.data_preprocessing import DataPreprocessor

# Test text cleaning
prep = DataPreprocessor()
text = 'Check out http://example.com! #FakeNews @user'
cleaned = prep.clean_text(text)
assert 'http' not in cleaned
assert '#' not in cleaned
assert '@' not in cleaned
print('✓ Text cleaning works')

# Test tokenization
tokens = prep.tokenize_and_lemmatize('running runs ran')
print(f'✓ Tokenization works: {tokens}')
"
```

### Test Model Training

```python
python -c "
from src.data_preprocessing import create_sample_dataset, DataPreprocessor
from src.ml_models import FakeNewsMLModels

# Create small dataset
df = create_sample_dataset('data/raw/test.csv', n_samples=100)

# Preprocess
prep = DataPreprocessor(max_features=100)
df = prep.preprocess_dataframe(df)
X_train, X_val, X_test, y_train, y_val, y_test = prep.prepare_train_test_split(df)
X_train_tfidf, X_val_tfidf, X_test_tfidf = prep.create_tfidf_features(X_train, X_val, X_test)

# Train
ml = FakeNewsMLModels()
ml.create_logistic_regression()
ml.train_model('logistic_regression', X_train_tfidf, y_train)

# Predict
y_pred = ml.predict('logistic_regression', X_test_tfidf)
accuracy = (y_pred == y_test).mean()
print(f'✓ Model training works. Accuracy: {accuracy:.2f}')
assert accuracy > 0.5  # Should be better than random
"
```

### Test Explainability

```python
python -c "
from src.data_preprocessing import create_sample_dataset, DataPreprocessor
from src.ml_models import FakeNewsMLModels
from src.explainability import ModelExplainer

# Quick setup
df = create_sample_dataset('data/raw/test.csv', n_samples=100)
prep = DataPreprocessor(max_features=100)
df = prep.preprocess_dataframe(df)
X_train, X_val, X_test, y_train, y_val, y_test = prep.prepare_train_test_split(df)
X_train_tfidf, X_val_tfidf, X_test_tfidf = prep.create_tfidf_features(X_train, X_val, X_test)

ml = FakeNewsMLModels()
ml.create_logistic_regression()
ml.train_model('logistic_regression', X_train_tfidf, y_train)

# Test LIME
explainer = ModelExplainer(ml.get_model('logistic_regression'), prep.vectorizer)
exp = explainer.explain_with_lime(X_test[0], num_features=5)
features = explainer.get_lime_top_features(exp, top_n=5)
print(f'✓ LIME works. Generated {len(features)} features')
assert len(features) > 0
"
```

---

## Performance Benchmarks

Expected performance on sample dataset (2000 articles):

| Component | Time | Memory |
|-----------|------|--------|
| Data loading | <1s | <100MB |
| Preprocessing | 30-60s | <200MB |
| LR training | 5-10s | <500MB |
| RF training | 30-60s | <1GB |
| SVM training | 60-120s | <1GB |
| Evaluation | <5s | <200MB |
| LIME (5 samples) | 30-60s | <500MB |
| SHAP (50 samples) | 3-5min | <1GB |

**Hardware:** Typical laptop (Intel i5, 8GB RAM)

---

## Common Issues and Solutions

### Issue: Import errors

```bash
# Solution: Install dependencies
pip install -r requirements.txt
```

### Issue: NLTK data not found

```bash
# Solution: Download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet')"
```

### Issue: Out of memory

```python
# Solution: Reduce dataset size or max_features
preprocessor = DataPreprocessor(max_features=1000)  # Instead of 5000
df_small = df.sample(500)  # Use smaller sample
```

### Issue: Matplotlib backend error

```bash
# Solution: Use non-interactive backend
export MPLBACKEND=Agg
```

### Issue: Permission denied

```bash
# Solution: Check file permissions
chmod +x main.py examples.py
```

---

## Validation Checklist

Use this checklist to verify project completeness:

**Data Processing:**
- [ ] Can load CSV files
- [ ] Text cleaning works
- [ ] TF-IDF vectorization succeeds
- [ ] Train/test split is correct

**Models:**
- [ ] Logistic Regression trains
- [ ] Random Forest trains
- [ ] SVM trains
- [ ] Models can make predictions
- [ ] Predictions have correct format

**Evaluation:**
- [ ] Accuracy calculated correctly
- [ ] Confusion matrix generated
- [ ] ROC curve plotted
- [ ] Metrics saved to JSON

**Explainability:**
- [ ] LIME explanations generate
- [ ] LIME plots display
- [ ] SHAP values calculate
- [ ] SHAP plots display

**Outputs:**
- [ ] Plots saved to results/plots/
- [ ] Reports saved to results/reports/
- [ ] Models saved to models/
- [ ] No error messages

---

## Continuous Testing

For development, create automated tests:

```python
# tests/test_preprocessing.py
import pytest
from src.data_preprocessing import DataPreprocessor

def test_text_cleaning():
    prep = DataPreprocessor()
    text = "Visit http://example.com! #news"
    cleaned = prep.clean_text(text)
    assert "http" not in cleaned
    assert "#" not in cleaned

def test_tokenization():
    prep = DataPreprocessor()
    tokens = prep.tokenize_and_lemmatize("running runs")
    assert len(tokens) > 0
    assert all(isinstance(t, str) for t in tokens)

# Run with: pytest tests/
```

---

## Success Criteria

Your project is working correctly if:

1. ✅ All imports work without errors
2. ✅ Sample dataset generates successfully
3. ✅ All 3 ML models train without errors
4. ✅ Test accuracy > 85% on sample data
5. ✅ Confusion matrices display correctly
6. ✅ ROC curves show AUC > 0.90
7. ✅ LIME explanations generate
8. ✅ SHAP explanations complete
9. ✅ All plots save to results/
10. ✅ Models save and load correctly

---

## Next Steps After Testing

Once all tests pass:

1. **Try real datasets**: Download and test with actual fake news data
2. **Tune hyperparameters**: Experiment with model parameters
3. **Add features**: Incorporate metadata, sources, etc.
4. **Deploy**: Create API or web interface
5. **Monitor**: Track performance over time

---

## Getting Help

If tests fail:
1. Check error messages carefully
2. Review QUICKSTART.md troubleshooting section
3. Verify all dependencies installed
4. Ensure Python 3.8+ is being used
5. Check file paths are correct

---

## Automated Test Script

Save this as `run_tests.sh`:

```bash
#!/bin/bash

echo "Running Fake News Detection Tests..."
echo "======================================"

echo -e "\n1. Testing imports..."
python -c "from src import *; print('✓ Imports successful')"

echo -e "\n2. Testing data preprocessing..."
python src/data_preprocessing.py --create-sample

echo -e "\n3. Testing ML models..."
python -c "from src.ml_models import FakeNewsMLModels; ml = FakeNewsMLModels(); ml.create_logistic_regression(); print('✓ ML models OK')"

echo -e "\n4. Testing full pipeline..."
python main.py --create-sample --models lr

echo -e "\n5. Testing examples..."
python examples.py

echo -e "\n======================================"
echo "All tests completed!"
```

Run with: `bash run_tests.sh`

---

**Happy Testing! 🧪**
