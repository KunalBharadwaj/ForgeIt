"""
Deep Learning Models for Fake News Detection
Implements LSTM and BERT-based models
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
from typing import Tuple, List, Optional
import warnings

warnings.filterwarnings("ignore")


class TextDataset(Dataset):
    """Custom Dataset for text data"""

    def __init__(
        self,
        texts: List[str],
        labels: np.ndarray,
        tokenizer=None,
        max_length: int = 512,
    ):
        """
        Initialize dataset

        Args:
            texts: List of text strings
            labels: Array of labels
            tokenizer: Tokenizer for BERT (optional)
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        if self.tokenizer:
            encoding = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt",
            )

            return {
                "input_ids": encoding["input_ids"].flatten(),
                "attention_mask": encoding["attention_mask"].flatten(),
                "label": torch.tensor(label, dtype=torch.long),
            }
        else:
            return {"text": text, "label": torch.tensor(label, dtype=torch.long)}


class LSTMClassifier(nn.Module):
    """LSTM-based fake news classifier"""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.5,
    ):
        """
        Initialize LSTM model

        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_dim: Dimension of LSTM hidden state
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super(LSTMClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 2)  # *2 for bidirectional

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)

        # Concatenate last hidden states from both directions
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        hidden = self.dropout(hidden)
        output = self.fc(hidden)

        return output


class FakeNewsDeepLearning:
    """
    Deep Learning models for fake news detection
    """

    def __init__(self, device: str = None):
        """
        Initialize DL models

        Args:
            device: Device to use ('cuda' or 'cpu')
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")
        self.models = {}

    def create_lstm_model(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.5,
    ) -> LSTMClassifier:
        """
        Create LSTM model

        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_dim: Dimension of LSTM hidden state
            num_layers: Number of LSTM layers
            dropout: Dropout rate

        Returns:
            LSTM model
        """
        model = LSTMClassifier(
            vocab_size, embedding_dim, hidden_dim, num_layers, dropout
        )
        model = model.to(self.device)
        self.models["lstm"] = model
        print(
            f"LSTM model created with {sum(p.numel() for p in model.parameters())} parameters"
        )
        return model

    def create_bert_model(
        self, model_name: str = "bert-base-uncased"
    ) -> BertForSequenceClassification:
        """
        Create BERT model

        Args:
            model_name: Name of pre-trained BERT model

        Returns:
            BERT model
        """
        print(f"Loading BERT model: {model_name}...")
        model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,
            output_attentions=False,
            output_hidden_states=False,
        )
        model = model.to(self.device)
        self.models["bert"] = model
        print(
            f"BERT model loaded with {sum(p.numel() for p in model.parameters())} parameters"
        )
        return model

    def train_lstm(
        self,
        model,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
        epochs: int = 10,
        learning_rate: float = 0.001,
    ) -> dict:
        """
        Train LSTM model

        Args:
            model: LSTM model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            learning_rate: Learning rate

        Returns:
            Training history
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        print(f"\nTraining LSTM for {epochs} epochs...")

        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch in train_loader:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["label"].to(self.device)

                optimizer.zero_grad()
                outputs = model(input_ids)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

            train_loss = train_loss / len(train_loader)
            train_acc = train_correct / train_total
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)

            # Validation
            if val_loader:
                model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0

                with torch.no_grad():
                    for batch in val_loader:
                        input_ids = batch["input_ids"].to(self.device)
                        labels = batch["label"].to(self.device)

                        outputs = model(input_ids)
                        loss = criterion(outputs, labels)

                        val_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()

                val_loss = val_loss / len(val_loader)
                val_acc = val_correct / val_total
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)

                print(
                    f"Epoch {epoch + 1}/{epochs} - "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
                )
            else:
                print(
                    f"Epoch {epoch + 1}/{epochs} - "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}"
                )

        print("LSTM training complete!")
        return history

    def train_bert(
        self,
        model,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
        epochs: int = 3,
        learning_rate: float = 2e-5,
    ) -> dict:
        """
        Train BERT model

        Args:
            model: BERT model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            learning_rate: Learning rate

        Returns:
            Training history
        """
        optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)

        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=total_steps
        )

        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        print(f"\nFine-tuning BERT for {epochs} epochs...")

        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch in train_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                optimizer.zero_grad()

                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

                loss = outputs.loss
                logits = outputs.logits

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                train_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

            train_loss = train_loss / len(train_loader)
            train_acc = train_correct / train_total
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)

            # Validation
            if val_loader:
                model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0

                with torch.no_grad():
                    for batch in val_loader:
                        input_ids = batch["input_ids"].to(self.device)
                        attention_mask = batch["attention_mask"].to(self.device)
                        labels = batch["label"].to(self.device)

                        outputs = model(
                            input_ids, attention_mask=attention_mask, labels=labels
                        )

                        loss = outputs.loss
                        logits = outputs.logits

                        val_loss += loss.item()
                        _, predicted = torch.max(logits, 1)
                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()

                val_loss = val_loss / len(val_loader)
                val_acc = val_correct / val_total
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)

                print(
                    f"Epoch {epoch + 1}/{epochs} - "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
                )
            else:
                print(
                    f"Epoch {epoch + 1}/{epochs} - "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}"
                )

        print("BERT training complete!")
        return history

    def predict(self, model, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions using a trained model

        Args:
            model: Trained model
            data_loader: Data loader

        Returns:
            Tuple of (predictions, probabilities)
        """
        model.eval()
        all_predictions = []
        all_probabilities = []

        with torch.no_grad():
            for batch in data_loader:
                if "input_ids" in batch:
                    input_ids = batch["input_ids"].to(self.device)

                    if hasattr(model, "bert"):  # BERT model
                        attention_mask = batch["attention_mask"].to(self.device)
                        outputs = model(input_ids, attention_mask=attention_mask)
                        logits = outputs.logits
                    else:  # LSTM model
                        logits = model(input_ids)

                    probabilities = torch.softmax(logits, dim=1)
                    _, predictions = torch.max(logits, 1)

                    all_predictions.extend(predictions.cpu().numpy())
                    all_probabilities.extend(probabilities.cpu().numpy())

        return np.array(all_predictions), np.array(all_probabilities)

    def save_model(self, model_name: str, filepath: str):
        """
        Save model to disk

        Args:
            model_name: Name of the model
            filepath: Path to save the model
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")

        model = self.models[model_name]
        torch.save(model.state_dict(), filepath)
        print(f"{model_name} model saved to {filepath}")

    def load_model(self, model_name: str, filepath: str):
        """
        Load model from disk

        Args:
            model_name: Name of the model
            filepath: Path to the saved model
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Create it first.")

        model = self.models[model_name]
        model.load_state_dict(torch.load(filepath, map_location=self.device))
        print(f"{model_name} model loaded from {filepath}")


if __name__ == "__main__":
    print("Deep Learning Models module loaded successfully!")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
