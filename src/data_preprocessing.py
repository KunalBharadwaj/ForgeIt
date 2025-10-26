"""
Data Preprocessing Module for Fake News Detection
Handles loading, cleaning, and preprocessing of text data
"""

import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os
from typing import Tuple, List, Optional
import warnings

warnings.filterwarnings("ignore")

# Download required NLTK data
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")
try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")


class DataPreprocessor:
    """
    Preprocesses text data for fake news detection
    """

    def __init__(self, max_features: int = 5000, max_len: int = 500):
        """
        Initialize the preprocessor

        Args:
            max_features: Maximum number of features for TF-IDF
            max_len: Maximum length for text sequences
        """
        self.max_features = max_features
        self.max_len = max_len
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()
        self.vectorizer = TfidfVectorizer(max_features=max_features)

    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess a single text string

        Args:
            text: Input text string

        Returns:
            Cleaned text string
        """
        if pd.isna(text):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)

        # Remove email addresses
        text = re.sub(r"\S+@\S+", "", text)

        # Remove mentions and hashtags
        text = re.sub(r"@\w+|#\w+", "", text)

        # Remove special characters and digits
        text = re.sub(r"[^a-zA-Z\s]", "", text)

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def tokenize_and_lemmatize(self, text: str) -> List[str]:
        """
        Tokenize and lemmatize text

        Args:
            text: Input text string

        Returns:
            List of processed tokens
        """
        # Tokenize
        tokens = word_tokenize(text)

        # Remove stopwords and lemmatize
        tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.stop_words and len(token) > 2
        ]

        return tokens

    def preprocess_text(self, text: str) -> str:
        """
        Complete preprocessing pipeline for a single text

        Args:
            text: Input text string

        Returns:
            Processed text string
        """
        # Clean
        text = self.clean_text(text)

        # Tokenize and lemmatize
        tokens = self.tokenize_and_lemmatize(text)

        # Rejoin tokens
        return " ".join(tokens)

    def load_data(
        self, file_path: str, text_column: str = None, label_column: str = "label"
    ) -> pd.DataFrame:
        """
        Load dataset from CSV file

        Args:
            file_path: Path to the CSV file
            text_column: Name of the text column (auto-detected if None)
            label_column: Name of the label column

        Returns:
            DataFrame with loaded data
        """
        print(f"Loading data from {file_path}...")
        df = pd.read_csv(file_path)

        # Auto-detect text column if not specified
        if text_column is None:
            possible_names = [
                "text",
                "content",
                "article",
                "title",
                "news",
                "statement",
            ]
            for name in possible_names:
                if name in df.columns:
                    text_column = name
                    break

            if text_column is None:
                raise ValueError(
                    f"Could not auto-detect text column. Available columns: {df.columns.tolist()}"
                )

        print(f"Using text column: '{text_column}', label column: '{label_column}'")

        # Keep only necessary columns
        df = df[[text_column, label_column]].copy()
        df.columns = ["text", "label"]

        # Handle missing values
        df = df.dropna()

        # Convert labels to binary (0 and 1)
        if df["label"].dtype == "object":
            label_map = {
                "fake": 1,
                "false": 1,
                "unreliable": 1,
                "bias": 1,
                "real": 0,
                "true": 0,
                "reliable": 0,
            }
            df["label"] = df["label"].str.lower().map(label_map)
            df = df.dropna()  # Drop any unmapped labels

        df["label"] = df["label"].astype(int)

        print(f"Loaded {len(df)} samples")
        print(f"Label distribution:\n{df['label'].value_counts()}")

        return df

    def preprocess_dataframe(
        self, df: pd.DataFrame, save_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Preprocess entire dataframe

        Args:
            df: Input dataframe with 'text' and 'label' columns
            save_path: Optional path to save processed data

        Returns:
            Processed dataframe
        """
        print("Preprocessing text data...")
        df["processed_text"] = df["text"].apply(self.preprocess_text)

        # Remove empty texts
        df = df[df["processed_text"].str.len() > 0]

        print(f"Preprocessing complete. {len(df)} samples remaining.")

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            df.to_csv(save_path, index=False)
            print(f"Saved preprocessed data to {save_path}")

        return df

    def prepare_train_test_split(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
    ) -> Tuple:
        """
        Split data into train, validation, and test sets

        Args:
            df: Preprocessed dataframe
            test_size: Fraction of data for testing
            val_size: Fraction of training data for validation
            random_state: Random seed

        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        print("Splitting data into train/val/test sets...")

        # First split: train+val vs test
        X = df["processed_text"].values
        y = df["label"].values

        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Second split: train vs val
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val,
            y_train_val,
            test_size=val_size,
            random_state=random_state,
            stratify=y_train_val,
        )

        print(f"Train set: {len(X_train)} samples")
        print(f"Validation set: {len(X_val)} samples")
        print(f"Test set: {len(X_test)} samples")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def create_tfidf_features(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray,
        save_path: Optional[str] = None,
    ) -> Tuple:
        """
        Create TF-IDF features for ML models

        Args:
            X_train: Training texts
            X_val: Validation texts
            X_test: Test texts
            save_path: Optional path to save vectorizer

        Returns:
            Tuple of (X_train_tfidf, X_val_tfidf, X_test_tfidf)
        """
        print("Creating TF-IDF features...")

        # Fit on training data
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_val_tfidf = self.vectorizer.transform(X_val)
        X_test_tfidf = self.vectorizer.transform(X_test)

        print(f"TF-IDF feature shape: {X_train_tfidf.shape}")

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "wb") as f:
                pickle.dump(self.vectorizer, f)
            print(f"Saved TF-IDF vectorizer to {save_path}")

        return X_train_tfidf, X_val_tfidf, X_test_tfidf


def create_sample_dataset(
    save_path: str = "data/raw/sample_news.csv", n_samples: int = 1000
):
    """
    Create a sample dataset for testing purposes

    Args:
        save_path: Path to save the sample dataset
        n_samples: Number of samples to generate
    """
    print(f"Creating sample dataset with {n_samples} samples...")

    fake_templates = [
        "Breaking: {subject} causes {effect} in {location}. Experts shocked!",
        "You won't believe what {celebrity} just revealed about {topic}!",
        "Secret {organization} plan to {action} exposed by whistleblower",
        "{subject} linked to {disease}! Government hiding the truth!",
        "Scientists discover {discovery} that {effect}! Media silent!",
    ]

    real_templates = [
        "Study shows {subject} may affect {outcome} according to research",
        "{organization} announces new policy regarding {topic}",
        "Economic report indicates {trend} in {sector} sector",
        "{location} officials discuss {topic} at annual conference",
        "Research team publishes findings on {subject} in scientific journal",
    ]

    subjects = [
        "vaccine",
        "climate change",
        "technology",
        "economy",
        "education",
        "healthcare",
        "social media",
        "artificial intelligence",
    ]
    effects = [
        "major changes",
        "unexpected results",
        "serious concerns",
        "dramatic improvements",
        "significant impacts",
    ]
    locations = [
        "major cities",
        "rural areas",
        "coastal regions",
        "worldwide",
        "developing nations",
    ]

    data = []

    for i in range(n_samples):
        if i % 2 == 0:  # Fake news
            template = np.random.choice(fake_templates)
            text = template.format(
                subject=np.random.choice(subjects),
                effect=np.random.choice(effects),
                location=np.random.choice(locations),
                celebrity=f"Celebrity{np.random.randint(1, 20)}",
                topic=np.random.choice(subjects),
                organization=f"Organization{np.random.randint(1, 10)}",
                action=np.random.choice(["control", "manipulate", "hide", "censor"]),
                disease=np.random.choice(["cancer", "disease", "illness"]),
                discovery=np.random.choice(
                    ["miracle cure", "hidden truth", "secret technology"]
                ),
            )
            label = 1
        else:  # Real news
            template = np.random.choice(real_templates)
            text = template.format(
                subject=np.random.choice(subjects),
                outcome=np.random.choice(["outcomes", "results", "trends"]),
                organization=f"Institution{np.random.randint(1, 10)}",
                topic=np.random.choice(subjects),
                trend=np.random.choice(["growth", "decline", "stability"]),
                sector=np.random.choice(["technology", "finance", "healthcare"]),
                location=np.random.choice(locations),
            )
            label = 0

        data.append({"text": text, "label": label})

    df = pd.DataFrame(data)

    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Sample dataset created and saved to {save_path}")
    print(f"Label distribution:\n{df['label'].value_counts()}")

    return df


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess fake news dataset")
    parser.add_argument("--input", type=str, help="Input CSV file path")
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/",
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--create-sample", action="store_true", help="Create sample dataset"
    )

    args = parser.parse_args()

    if args.create_sample:
        create_sample_dataset()

    if args.input:
        preprocessor = DataPreprocessor()
        df = preprocessor.load_data(args.input)
        df = preprocessor.preprocess_dataframe(
            df, save_path=os.path.join(args.output, "preprocessed_data.csv")
        )

        X_train, X_val, X_test, y_train, y_val, y_test = (
            preprocessor.prepare_train_test_split(df)
        )

        X_train_tfidf, X_val_tfidf, X_test_tfidf = preprocessor.create_tfidf_features(
            X_train,
            X_val,
            X_test,
            save_path=os.path.join(args.output, "tfidf_vectorizer.pkl"),
        )

        print("\nPreprocessing complete!")
