import os, argparse, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
import matplotlib.pyplot as plt, seaborn as sns
from lime.lime_text import LimeTextExplainer
import joblib

def ensure_dirs():
    for p in ["results", "results/plots", "results/reports", "models"]:
        os.makedirs(p, exist_ok=True)

def load_csv(path):
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    text_col = cols.get('text') or cols.get('title')
    label_col = cols.get('label')
    if not text_col or not label_col:
        raise ValueError("CSV must have columns: text,label (labels like real/fake or 0/1)")
    df = df[[text_col, label_col]].rename(columns={text_col:'text', label_col:'label'})
    df = df.dropna()
    return df

def plot_confusion(cm, classes, outpath):
    plt.figure(figsize=(4.5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted'); plt.ylabel('True'); plt.tight_layout()
    plt.savefig(outpath, dpi=150); plt.close()

def plot_roc(y_true, y_prob, outpath):
    if y_prob is None:
        return
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(4.5,4))
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0,1],[0,1],'--')
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.legend(); plt.tight_layout()
    plt.savefig(outpath, dpi=150); plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="CSV with columns text,label")
    ap.add_argument("--test_size", type=float, default=0.33)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lime_samples", type=int, default=2)
    args = ap.parse_args()

    ensure_dirs()
    df = load_csv(args.data)

    # Encode labels (handles strings like real/fake)
    le = LabelEncoder()
    y = le.fit_transform(df['label'])
    classes = list(le.classes_)

    X_train, X_test, y_train, y_test = train_test_split(
        df['text'].values, y, test_size=args.test_size, random_state=args.seed, stratify=y if len(np.unique(y))>1 else None
    )

    # Balanced class weights improve robustness on imbalanced data
    if len(np.unique(y_train)) > 1:
        weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
        cw = {i: w for i, w in enumerate(weights)}
    else:
        cw = None

    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words='english')),
        ('clf', LogisticRegression(max_iter=500, class_weight=cw))
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    # Probability for positive class in binary case
    y_prob = None
    if len(classes) == 2:
        try:
            y_prob = pipe.predict_proba(X_test)[:,1]
        except Exception:
            y_prob = None

    report = classification_report(y_test, y_pred, target_names=classes, digits=4, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    # Save model and artifacts
    joblib.dump(pipe, "models/logreg_tfidf.joblib")
    with open("results/reports/performance_report.txt", "w") as f:
        f.write("Classes: " + ", ".join(classes) + "\n\n")
        f.write(report + "\n")
        f.write("Confusion Matrix:\n" + np.array2string(cm) + "\n")

    plot_confusion(cm, classes, "results/plots/confusion_matrix.png")
    if len(classes)==2 and y_prob is not None:
        # Bin y_true as 0/1 (LabelEncoder already did this)
        plot_roc(y_test, y_prob, "results/plots/roc.png")

    # LIME explanations
    if len(X_test) > 0:
        explainer = LimeTextExplainer(class_names=classes)
        rng = np.random.default_rng(args.seed)
        sample_idxs = rng.choice(len(X_test), size=min(args.lime_samples, len(X_test)), replace=False)
        for idx in sample_idxs:
            exp = explainer.explain_instance(X_test[idx], pipe.predict_proba, num_features=8)
            exp.save_to_file(f"results/reports/lime_example_{idx}.html")

    print("\n=== DONE ===")
    print("Model: models/logreg_tfidf.joblib")
    print("Report: results/reports/performance_report.txt")
    print("Plots:  results/plots/confusion_matrix.png", end="")
    if len(classes)==2 and y_prob is not None:
        print(", results/plots/roc.png")
    else:
        print()
    print("LIME:   results/reports/lime_example_*.html\n")

if __name__ == "__main__":
    main()
