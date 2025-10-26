"""
Flask Web API for Fake News Detection
Run with: python app.py
"""

from flask import Flask, request, jsonify, render_template_string
import pickle
from src.data_preprocessing import DataPreprocessor
from src.utils import load_model

app = Flask(__name__)

# Load model and vectorizer at startup
preprocessor = DataPreprocessor()
with open("data/processed/tfidf_vectorizer.pkl", "rb") as f:
    preprocessor.vectorizer = pickle.load(f)
model = load_model("models/random_forest.pkl")

# HTML template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Fake News Detector</title>
    <style>
        body { font-family: Arial; max-width: 800px; margin: 50px auto; padding: 20px; }
        textarea { width: 100%; height: 200px; padding: 10px; }
        button { padding: 10px 20px; background: #007bff; color: white; border: none; cursor: pointer; }
        .result { margin-top: 20px; padding: 20px; border-radius: 5px; }
        .fake { background: #f8d7da; border: 1px solid #f5c6cb; }
        .real { background: #d4edda; border: 1px solid #c3e6cb; }
    </style>
</head>
<body>
    <h1>🔍 Fake News Detector</h1>
    <p>Enter a news article to check if it's real or fake:</p>
    
    <form id="newsForm">
        <textarea id="newsText" placeholder="Paste news article here..."></textarea>
        <br><br>
        <button type="submit">Analyze</button>
    </form>
    
    <div id="result"></div>
    
    <script>
        document.getElementById('newsForm').onsubmit = async (e) => {
            e.preventDefault();
            const text = document.getElementById('newsText').value;
            
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({text: text})
            });
            
            const data = await response.json();
            const resultDiv = document.getElementById('result');
            const className = data.prediction === 'FAKE' ? 'fake' : 'real';
            
            resultDiv.className = 'result ' + className;
            resultDiv.innerHTML = `
                <h2>Result: ${data.prediction}</h2>
                <p>Confidence: ${data.confidence}%</p>
                <p><small>Model: ${data.model}</small></p>
            `;
        };
    </script>
</body>
</html>
"""


@app.route("/")
def home():
    return render_template_string(HTML_TEMPLATE)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Preprocess
    cleaned = preprocessor.preprocess_text(text)
    vectorized = preprocessor.vectorizer.transform([cleaned])

    # Predict
    prediction = model.predict(vectorized)[0]
    probability = model.predict_proba(vectorized)[0]

    result = {
        "prediction": "FAKE" if prediction == 1 else "REAL",
        "confidence": round(probability[prediction] * 100, 2),
        "model": "Random Forest",
        "probabilities": {
            "real": round(probability[0] * 100, 2),
            "fake": round(probability[1] * 100, 2),
        },
    }

    return jsonify(result)


@app.route("/health")
def health():
    return jsonify({"status": "healthy", "model": "loaded"})


if __name__ == "__main__":
    print("🚀 Starting Fake News Detection API...")
    print("📍 Open: http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
