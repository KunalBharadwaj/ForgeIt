# API Documentation

## Explainable Fake News Detection - Module Documentation

This document provides detailed API documentation for all modules in the project.

---

## Table of Contents

1. [Data Preprocessing](#data-preprocessing)
2. [ML Models](#ml-models)
3. [Deep Learning Models](#deep-learning-models)
4. [Explainability](#explainability)
5. [Visualization](#visualization)
6. [Utilities](#utilities)

---

## Data Preprocessing

### `DataPreprocessor`

Main class for data loading, cleaning, and preprocessing.

#### Constructor

```python
DataPreprocessor(max_features=5000, max_len=500)
```

**Parameters:**
- `max_features` (int): Maximum number of TF-IDF features
- `max_len` (int): Maximum sequence length for text

#### Methods

##### `load_data(file_path, text_column=None, label_column='label')`

Load dataset from CSV file.

**Parameters:**
- `file_path` (str): Path to CSV file
- `text_column` (str, optional): Name of text column (auto-detected if None)
- `label_column` (str): Name of label column

**Returns:**
- `pd.DataFrame`: Loaded dataframe

**Example:**
```python
preprocessor = DataPreprocessor()
df = preprocessor.load_data('data/raw/news.csv')
```

##### `clean_text(text)`

Clean a single text string.

**Parameters:**
- `text` (str): Input text

**Returns:**
- `str`: Cleaned text

**Operations performed:**
- Convert to lowercase
- Remove URLs, emails, mentions, hashtags
- Remove special characters and digits
- Remove extra whitespace

##### `preprocess_dataframe(df, save_path=None)`

Preprocess entire dataframe.

**Parameters:**
- `df` (pd.DataFrame): Input dataframe
- `save_path` (str, optional): Path to save processed data

**Returns:**
- `pd.DataFrame`: Processed dataframe with 'processed_text' column

##### `prepare_train_test_split(df, test_size=0.2, val_size=0.1, random_state=42)`

Split data into train, validation, and test sets.

**Parameters:**
- `df` (pd.DataFrame): Preprocessed dataframe
- `test_size` (float): Fraction for test set
- `val_size` (float): Fraction of training for validation
- `random_state` (int): Random seed

**Returns:**
- `tuple`: (X_train, X_val, X_test, y_train, y_val, y_test)

##### `create_tfidf_features(X_train, X_val, X_test, save_path=None)`

Create TF-IDF features for ML models.

**Parameters:**
- `X_train` (array): Training texts
- `X_val` (array): Validation texts
- `X_test` (array): Test texts
- `save_path` (str, optional): Path to save vectorizer

**Returns:**
- `tuple`: (X_train_tfidf, X_val_tfidf, X_test_tfidf)

### Helper Functions

##### `create_sample_dataset(save_path='data/raw/sample_news.csv', n_samples=1000)`

Create a sample dataset for testing.

**Parameters:**
- `save_path` (str): Path to save dataset
- `n_samples` (int): Number of samples to generate

**Returns:**
- `pd.DataFrame`: Generated dataset

---

## ML Models

### `FakeNewsMLModels`

Collection of traditional machine learning models.

#### Constructor

```python
FakeNewsMLModels(random_state=42)
```

**Parameters:**
- `random_state` (int): Random seed for reproducibility

#### Methods

##### `create_logistic_regression(**kwargs)`

Create Logistic Regression model.

**Parameters:**
- `**kwargs`: Additional parameters for LogisticRegression

**Returns:**
- `LogisticRegression`: Configured model

**Example:**
```python
ml_models = FakeNewsMLModels()
model = ml_models.create_logistic_regression(C=1.0, max_iter=1000)
```

##### `create_random_forest(**kwargs)`

Create Random Forest model.

**Parameters:**
- `**kwargs`: Additional parameters for RandomForestClassifier

**Default Parameters:**
- `n_estimators=100`
- `max_depth=20`
- `min_samples_split=5`

**Returns:**
- `RandomForestClassifier`: Configured model

##### `create_svm(**kwargs)`

Create SVM model.

**Parameters:**
- `**kwargs`: Additional parameters for SVC

**Default Parameters:**
- `kernel='rbf'`
- `C=1.0`
- `probability=True`

**Returns:**
- `SVC`: Configured model

##### `train_model(model_name, X_train, y_train, X_val=None, y_val=None)`

Train a specific model.

**Parameters:**
- `model_name` (str): Name of model to train
- `X_train` (array): Training features
- `y_train` (array): Training labels
- `X_val` (array, optional): Validation features
- `y_val` (array, optional): Validation labels

**Returns:**
- Trained model

**Example:**
```python
ml_models.train_model('random_forest', X_train_tfidf, y_train, X_val_tfidf, y_val)
```

##### `predict(model_name, X)`

Make predictions.

**Parameters:**
- `model_name` (str): Name of model
- `X` (array): Features

**Returns:**
- `np.ndarray`: Predictions

##### `predict_proba(model_name, X)`

Get prediction probabilities.

**Parameters:**
- `model_name` (str): Name of model
- `X` (array): Features

**Returns:**
- `np.ndarray`: Prediction probabilities

##### `hyperparameter_tuning(model_name, X_train, y_train, param_grid, cv=5)`

Perform hyperparameter tuning using GridSearchCV.

**Parameters:**
- `model_name` (str): Name of model
- `X_train` (array): Training features
- `y_train` (array): Training labels
- `param_grid` (dict): Parameter grid for GridSearchCV
- `cv` (int): Number of cross-validation folds

**Returns:**
- Best model after tuning

**Example:**
```python
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30]
}
ml_models.hyperparameter_tuning('random_forest', X_train, y_train, param_grid)
```

---

## Deep Learning Models

### `FakeNewsDeepLearning`

Deep learning models for fake news detection.

#### Constructor

```python
FakeNewsDeepLearning(device=None)
```

**Parameters:**
- `device` (str, optional): Device to use ('cuda' or 'cpu'). Auto-detected if None.

#### Methods

##### `create_lstm_model(vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2, dropout=0.5)`

Create LSTM model.

**Parameters:**
- `vocab_size` (int): Size of vocabulary
- `embedding_dim` (int): Dimension of word embeddings
- `hidden_dim` (int): Dimension of LSTM hidden state
- `num_layers` (int): Number of LSTM layers
- `dropout` (float): Dropout rate

**Returns:**
- `LSTMClassifier`: LSTM model

##### `create_bert_model(model_name='bert-base-uncased')`

Create BERT model.

**Parameters:**
- `model_name` (str): Name of pre-trained BERT model

**Returns:**
- `BertForSequenceClassification`: BERT model

##### `train_lstm(model, train_loader, val_loader=None, epochs=10, learning_rate=0.001)`

Train LSTM model.

**Parameters:**
- `model`: LSTM model to train
- `train_loader` (DataLoader): Training data loader
- `val_loader` (DataLoader, optional): Validation data loader
- `epochs` (int): Number of training epochs
- `learning_rate` (float): Learning rate

**Returns:**
- `dict`: Training history

##### `train_bert(model, train_loader, val_loader=None, epochs=3, learning_rate=2e-5)`

Fine-tune BERT model.

**Parameters:**
- `model`: BERT model to train
- `train_loader` (DataLoader): Training data loader
- `val_loader` (DataLoader, optional): Validation data loader
- `epochs` (int): Number of epochs (default 3 for BERT)
- `learning_rate` (float): Learning rate (2e-5 recommended for BERT)

**Returns:**
- `dict`: Training history

##### `predict(model, data_loader)`

Make predictions using trained model.

**Parameters:**
- `model`: Trained model
- `data_loader` (DataLoader): Data loader

**Returns:**
- `tuple`: (predictions, probabilities)

---

## Explainability

### `ModelExplainer`

Explainability wrapper supporting LIME and SHAP.

#### Constructor

```python
ModelExplainer(model, vectorizer=None, class_names=None)
```

**Parameters:**
- `model`: Trained model (ML or DL)
- `vectorizer`: Text vectorizer (required for ML models)
- `class_names` (list): Names of output classes (default: ['Real', 'Fake'])

#### LIME Methods

##### `explain_with_lime(text, num_features=10, num_samples=5000)`

Generate LIME explanation for a text.

**Parameters:**
- `text` (str): Input text to explain
- `num_features` (int): Number of features in explanation
- `num_samples` (int): Number of samples for LIME

**Returns:**
- LIME explanation object

**Example:**
```python
explainer = ModelExplainer(model, vectorizer)
lime_exp = explainer.explain_with_lime(text, num_features=10)
lime_exp.show_in_notebook()
```

##### `visualize_lime_explanation(explanation, save_path=None, figsize=(10, 6))`

Visualize LIME explanation.

**Parameters:**
- `explanation`: LIME explanation object
- `save_path` (str, optional): Path to save plot
- `figsize` (tuple): Figure size

##### `get_lime_top_features(explanation, top_n=10)`

Get top features from LIME explanation.

**Parameters:**
- `explanation`: LIME explanation object
- `top_n` (int): Number of top features

**Returns:**
- `pd.DataFrame`: Features and weights

#### SHAP Methods

##### `explain_with_shap(texts, background_samples=100)`

Generate SHAP explanations.

**Parameters:**
- `texts` (list): List of input texts
- `background_samples` (int): Number of background samples

**Returns:**
- `tuple`: (shap_values, explainer)

**Example:**
```python
shap_values, shap_explainer = explainer.explain_with_shap(texts[:100])
```

##### `visualize_shap_summary(shap_values, X, feature_names=None, save_path=None, max_display=20)`

Create SHAP summary plot.

**Parameters:**
- `shap_values`: SHAP values array
- `X`: Feature matrix
- `feature_names` (list, optional): Names of features
- `save_path` (str, optional): Path to save plot
- `max_display` (int): Maximum features to display

##### `visualize_shap_force_plot(shap_values, explainer, instance_idx=0, save_path=None)`

Create SHAP force plot for single instance.

**Parameters:**
- `shap_values`: SHAP values
- `explainer`: SHAP explainer object
- `instance_idx` (int): Index of instance
- `save_path` (str, optional): Path to save plot

##### `get_shap_top_features(shap_values, feature_names=None, instance_idx=0, top_n=10)`

Get top features from SHAP values.

**Parameters:**
- `shap_values`: SHAP values array
- `feature_names` (list, optional): Names of features
- `instance_idx` (int): Index of instance
- `top_n` (int): Number of top features

**Returns:**
- `pd.DataFrame`: Features and SHAP values

---

## Visualization

### Functions

##### `plot_training_history(history, model_name, save_path=None, figsize=(12, 5))`

Plot training history (loss and accuracy).

**Parameters:**
- `history` (dict): Training history with 'train_loss', 'val_loss', etc.
- `model_name` (str): Name of model
- `save_path` (str, optional): Path to save plot
- `figsize` (tuple): Figure size

##### `plot_word_cloud(texts, labels, label_value, title, save_path=None, figsize=(12, 8))`

Generate word cloud for specific label.

**Parameters:**
- `texts` (list): List of texts
- `labels` (array): Array of labels
- `label_value` (int): Label to filter (0 or 1)
- `title` (str): Plot title
- `save_path` (str, optional): Path to save plot
- `figsize` (tuple): Figure size

##### `plot_multiple_roc_curves(results_dict, save_path=None, figsize=(10, 8))`

Plot ROC curves for multiple models.

**Parameters:**
- `results_dict` (dict): {model_name: (y_true, y_proba)} pairs
- `save_path` (str, optional): Path to save plot
- `figsize` (tuple): Figure size

##### `plot_multiple_confusion_matrices(results_dict, save_path=None, figsize=(15, 5))`

Plot confusion matrices for multiple models.

**Parameters:**
- `results_dict` (dict): {model_name: (y_true, y_pred)} pairs
- `save_path` (str, optional): Path to save plot
- `figsize` (tuple): Figure size

##### `create_performance_report(results_list, save_path=None)`

Create comprehensive performance report.

**Parameters:**
- `results_list` (list): List of result dictionaries
- `save_path` (str, optional): Path to save report

**Returns:**
- `str`: Report text

---

## Utilities

### Functions

##### `evaluate_model(y_true, y_pred, model_name="Model")`

Evaluate model performance.

**Parameters:**
- `y_true` (array): True labels
- `y_pred` (array): Predicted labels
- `model_name` (str): Name of model

**Returns:**
- `dict`: Metrics (accuracy, precision, recall, f1_score, etc.)

**Example:**
```python
results = evaluate_model(y_test, y_pred, "Random Forest")
print(f"Accuracy: {results['accuracy']:.4f}")
```

##### `save_model(model, filepath)`

Save trained model to disk.

**Parameters:**
- `model`: Model object
- `filepath` (str): Path to save model

##### `load_model(filepath)`

Load trained model from disk.

**Parameters:**
- `filepath` (str): Path to saved model

**Returns:**
- Loaded model

##### `save_results(results, filepath)`

Save evaluation results to JSON.

**Parameters:**
- `results` (dict): Results dictionary
- `filepath` (str): Path to save results

##### `get_feature_importance(model, vectorizer, top_n=20)`

Get top important features from model.

**Parameters:**
- `model`: Trained model with feature_importances_ or coef_
- `vectorizer`: Fitted vectorizer
- `top_n` (int): Number of top features

**Returns:**
- `pd.DataFrame`: Features and importance scores

##### `plot_confusion_matrix(cm, model_name, save_path=None, figsize=(8, 6))`

Plot confusion matrix.

**Parameters:**
- `cm` (array): Confusion matrix
- `model_name` (str): Name of model
- `save_path` (str, optional): Path to save plot
- `figsize` (tuple): Figure size

##### `plot_roc_curve(y_true, y_proba, model_name, save_path=None, figsize=(8, 6))`

Plot ROC curve.

**Parameters:**
- `y_true` (array): True labels
- `y_proba` (array): Prediction probabilities
- `model_name` (str): Name of model
- `save_path` (str, optional): Path to save plot
- `figsize` (tuple): Figure size

**Returns:**
- `float`: ROC AUC score

---

## Complete Usage Example

```python
# 1. Import modules
from src.data_preprocessing import DataPreprocessor
from src.ml_models import FakeNewsMLModels
from src.explainability import ModelExplainer
from src.utils import evaluate_model, plot_confusion_matrix
from src.visualization import plot_word_cloud

# 2. Load and preprocess data
preprocessor = DataPreprocessor(max_features=5000)
df = preprocessor.load_data('data/raw/news.csv')
df = preprocessor.preprocess_dataframe(df)

# 3. Split data
X_train, X_val, X_test, y_train, y_val, y_test = \
    preprocessor.prepare_train_test_split(df)

# 4. Create TF-IDF features
X_train_tfidf, X_val_tfidf, X_test_tfidf = \
    preprocessor.create_tfidf_features(X_train, X_val, X_test)

# 5. Train models
ml_models = FakeNewsMLModels()
ml_models.create_random_forest()
ml_models.train_model('random_forest', X_train_tfidf, y_train)

# 6. Evaluate
y_pred = ml_models.predict('random_forest', X_test_tfidf)
results = evaluate_model(y_test, y_pred, 'Random Forest')

# 7. Generate explanations
explainer = ModelExplainer(
    ml_models.get_model('random_forest'),
    preprocessor.vectorizer
)
lime_exp = explainer.explain_with_lime(X_test[0])
lime_exp.show_in_notebook()
```

---

For more examples, see:
- `main.py` for command-line usage
- `notebooks/demo.ipynb` for interactive examples
- Individual module files for detailed docstrings
