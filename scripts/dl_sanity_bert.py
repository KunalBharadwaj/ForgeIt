# Tiny BERT sanity fine-tune. Optional. Requires: transformers datasets torch accelerate
# Usage:
#   pip install transformers datasets torch accelerate
#   python scripts/dl_sanity_bert.py --subset 1000 --epochs 1
import argparse, numpy as np, pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/raw/news.csv")
ap.add_argument("--subset", type=int, default=1000)
ap.add_argument("--epochs", type=float, default=1.0)
ap.add_argument("--model", default="distilbert-base-uncased")
args = ap.parse_args()

df = pd.read_csv(args.data)[["text","label"]].dropna().sample(args.subset, random_state=42)
df["label_id"] = df["label"].str.lower().map({"fake":0,"real":1})
ds = Dataset.from_pandas(df[["text","label_id"]])

tok = AutoTokenizer.from_pretrained(args.model)
def tok_fn(batch): return tok(batch["text"], truncation=True, padding="max_length", max_length=256)
tds = ds.train_test_split(test_size=0.2, seed=42).map(tok_fn, batched=True)
tds = tds.remove_columns([c for c in tds["train"].column_names if c not in ["input_ids","attention_mask","label_id"]])
tds = tds.rename_column("label_id","labels")

model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)
args_tr = TrainingArguments(output_dir="results/bert_sanity", per_device_train_batch_size=8,
                            per_device_eval_batch_size=8, num_train_epochs=args.epochs,
                            evaluation_strategy="epoch", logging_steps=50, save_strategy="no")
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {"accuracy": float((preds == p.label_ids).mean())}

trainer = Trainer(model=model, args=args_tr, train_dataset=tds["train"], eval_dataset=tds["test"],
                  tokenizer=tok, compute_metrics=compute_metrics)
trainer.train(); print(trainer.evaluate())
