from flask import Flask, render_template_string, request
import joblib

app = Flask(__name__)

# Load the pre-trained model ONCE when the server boots up (Instant load)
MODEL_PATH = "model2.joblib"
model = joblib.load(MODEL_PATH)


def predict_explain(text):
    """Simple prediction and explanation wrapper using the loaded model pipeline."""
    # Predict class
    prediction = model.predict([text])[0]
    probabilities = model.predict_proba([text])[0]
    classes = model.classes_
    
    # Get confidence score and runner up
    max_idx = probabilities.argmax()
    confidence = probabilities[max_idx]
    
    # Sort for runner-up
    sorted_indices = probabilities.argsort()[::-1]
    runner_up = classes[sorted_indices[1]] if len(classes) > 1 else "N/A"
    runner_up_conf = probabilities[sorted_indices[1]] if len(classes) > 1 else 0.0

    # Basic threshold check for unsure verdict
    verdict = prediction
    if confidence < 0.45:
        verdict = f"UNSURE ({prediction})"

    # Extract top contributing keywords if it's a linear model (like LogisticRegression)
    key_words = []
    try:
        vectorizer = model.named_steps['tfidf']
        classifier = model.named_steps['clf']
        
        # Get class index
        class_idx = list(classes).index(prediction)
        feature_names = vectorizer.get_feature_names_out()
        
        # Get coefficients for this class
        if hasattr(classifier, "coef_"):
            coefs = classifier.coef_[class_idx]
            text_tokens = vectorizer.build_tokenizer()(text.lower())
            
            # Find tokens present in text with highest weights
            token_weights = {token: coefs[vectorizer.vocabulary_[token]] 
                             for token in text_tokens if token in vectorizer.vocabulary_}
            
            key_words = sorted(token_weights, key=token_weights.get, reverse=True)[:4]
    except Exception:
        key_words = []

    return {
        "verdict": verdict,
        "confidence": round(float(confidence), 2),
        "runner_up": f"{runner_up} ({round(float(runner_up_conf), 2)})",
        "key_words": key_words
    }

# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Cyberbullying Type Detector</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; max-width: 650px; margin: 40px auto; padding: 0 20px; line-height: 1.5; color: #333; }
        textarea { width: 100%; height: 120px; padding: 10px; box-sizing: border-box; border: 1px solid #ccc; border-radius: 4px; font-size: 15px; }
        button { background-color: #0066cc; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; font-size: 15px; margin-top: 10px; }
        button:hover { background-color: #0052a3; }
        .card { border: 1px solid #e0e0e0; background: #fafafa; border-radius: 6px; padding: 15px 20px; margin-top: 25px; }
        .warning { background-color: #fff3cd; border: 1px solid #ffeeba; color: #856404; padding: 10px 15px; border-radius: 4px; margin-top: 15px; }
        .chip { display: inline-block; background: #e1f0ff; color: #0066cc; padding: 3px 8px; border-radius: 12px; margin: 2px; font-weight: 500; font-size: 14px; }
    </style>
</head>
<body>
    <h2>Cyberbullying Type Detector</h2>
    
    <form method="POST" action="/">
        <textarea name="comment" placeholder="Paste a comment or tweet here...">{{ text }}</textarea>
        <button type="submit">Analyse</button>
    </form>

    {% if result %}
    <div class="card">
        <h3>Verdict: {{ result.verdict }}</h3>
        <p><strong>Confidence:</strong> {{ result.confidence }} &nbsp;|&nbsp; <strong>Runner-up:</strong> {{ result.runner_up }}</p>
        
        <p><strong>Words that pushed the model:</strong></p>
        <div>
            {% for word in result.key_words %}
                <span class="chip">{{ word }}</span>
            {% else %}
                <em>None identified</em>
            {% endfor %}
        </div>

        {% if result.verdict.startswith("UNSURE") %}
            <div class="warning">
                ⚠️ Not enough evidence, so the model declines to decide.
            </div>
        {% endif %}
    </div>
    {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    text = ""
    if request.method == "POST":
        text = request.form.get("comment", "").strip()
        if text:
            result = predict_explain(text)
            
    return render_template_string(HTML_TEMPLATE, text=text, result=result)

if __name__ == "__main__":
    app.run(debug=True, port=5000)