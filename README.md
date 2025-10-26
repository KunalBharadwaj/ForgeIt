# Explainable Fake News Detection using Machine Learning and Deep Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive implementation of fake news detection using both traditional Machine Learning and Deep Learning approaches, with explainability powered by LIME (Local Interpretable Model-agnostic Explanations) and SHAP (SHapley Additive exPlanations).

## 🎯 Project Overview

This project addresses the critical problem of fake news detection by:
- Implementing multiple ML models (Logistic Regression, Random Forest, SVM)
- Implementing Deep Learning models (LSTM, BERT)
- Providing **explainable AI (XAI)** using LIME and SHAP
- Visualizing model predictions and explanations
- Comparing model performance and interpretability

## 🚀 Features

- **Multiple Model Architectures**
  - Traditional ML: Logistic Regression, Random Forest, SVM
  - Deep Learning: LSTM, BERT (Transformer-based)
  
- **Explainability Techniques**
  - LIME: Local interpretable model-agnostic explanations
  - SHAP: Unified approach to explain predictions
  
- **Comprehensive Analysis**
  - Performance metrics (Accuracy, Precision, Recall, F1-Score)
  - Confusion matrices
  - ROC curves
  - Feature importance visualization
  - Word-level and phrase-level explanations

## 📁 Project Structure

```
AIProject/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── data/
│   ├── raw/                    # Raw dataset files
│   ├── processed/              # Preprocessed data
│   └── README.md
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py   # Data loading and preprocessing
│   ├── ml_models.py            # Traditional ML models
│   ├── dl_models.py            # Deep Learning models
│   ├── explainability.py       # LIME & SHAP implementations
│   ├── visualization.py        # Plotting and visualization
│   └── utils.py                # Utility functions
├── notebooks/
│   └── demo.ipynb              # Interactive demonstration
├── models/                     # Saved model files
├── results/                    # Outputs, plots, reports
│   ├── plots/
│   └── reports/
└── main.py                     # Main training script
```

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-capable GPU for deep learning models

### Setup

1. Clone the repository:
```bash
cd /home/kunal/Desktop/AIProject
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download NLTK data:
```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet')"
```

## 📊 Dataset

This project uses fake news datasets. You can use:
- **Kaggle Fake News Dataset**: [Link](https://www.kaggle.com/c/fake-news/data)
- **LIAR Dataset**: Political statements with truth ratings
- **FakeNewsNet**: Social media fake news dataset

Place your dataset in the `data/raw/` directory.

Expected CSV format:
- `text` or `title`: The news article text
- `label`: Binary classification (0=Real, 1=Fake) or (Real/Fake)

## 🎮 Usage

### Training Models

Run the main training script:
```bash
python main.py --data data/raw/news.csv --models all --explain
```

Options:
- `--data`: Path to dataset
- `--models`: Choose models (lr, rf, svm, lstm, bert, all)
- `--explain`: Enable LIME & SHAP explanations
- `--test-size`: Train-test split ratio (default: 0.2)

### Interactive Demo

Launch the Jupyter notebook for interactive exploration:
```bash
jupyter notebook notebooks/demo.ipynb
```

### Explaining Individual Predictions

```python
from src.explainability import explain_prediction

# Explain a prediction using LIME
explanation = explain_prediction(
    model=trained_model,
    text="Your news article text here",
    method="lime"
)

# Visualize the explanation
explanation.show_in_notebook()
```

## 📈 Models Implemented

### Traditional Machine Learning

1. **Logistic Regression**
   - Fast, interpretable baseline
   - TF-IDF feature extraction
   
2. **Random Forest**
   - Ensemble method
   - Feature importance analysis
   
3. **Support Vector Machine (SVM)**
   - Effective for high-dimensional text data
   - RBF and linear kernels

### Deep Learning

4. **LSTM (Long Short-Term Memory)**
   - Sequential text processing
   - Word embeddings (Word2Vec/GloVe)
   
5. **BERT (Bidirectional Encoder Representations from Transformers)**
   - State-of-the-art NLP model
   - Fine-tuned for fake news classification

## 🔍 Explainability Methods

### LIME (Local Interpretable Model-agnostic Explanations)

LIME explains individual predictions by:
- Perturbing input text
- Building local linear approximations
- Highlighting influential words/phrases

**Advantages:**
- Model-agnostic
- Human-interpretable
- Works with any classifier

### SHAP (SHapley Additive exPlanations)

SHAP provides unified explanations based on game theory:
- Consistent feature attribution
- Shapley values for each word
- Global and local explanations

**Advantages:**
- Theoretically grounded
- Consistent and accurate
- Multiple visualization types

## 📊 Results

Example performance metrics:

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 92.3% | 91.8% | 92.7% | 92.2% |
| Random Forest | 94.1% | 93.5% | 94.6% | 94.0% |
| SVM | 93.7% | 93.2% | 94.1% | 93.6% |
| LSTM | 95.2% | 94.8% | 95.6% | 95.2% |
| BERT | 97.4% | 97.1% | 97.7% | 97.4% |

*Note: Results may vary based on dataset and hyperparameters*

## 🖼️ Visualizations

The project generates various visualizations:
- Confusion matrices
- ROC curves and AUC scores
- Word importance heatmaps (LIME)
- SHAP force plots
- SHAP summary plots
- Feature importance charts

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 References

- Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?" Explaining the predictions of any classifier. KDD.
- Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. NIPS.
- Devlin, J., et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.
- Shu, K., et al. (2017). Fake News Detection on Social Media: A Data Mining Perspective.

## 👥 Authors

- Kunal Bharadwaj - [GitHub](https://github.com/KunalBharadwaj)

## 🙏 Acknowledgments

- LIME and SHAP libraries for explainability frameworks
- Hugging Face for transformer models
- Kaggle for fake news datasets
- scikit-learn for ML implementations

## 📧 Contact

For questions or suggestions, please open an issue or contact [bharadwajkunal172@gmail.com]

---

**Note**: This project is for educational and research purposes. Always verify news from multiple credible sources.
