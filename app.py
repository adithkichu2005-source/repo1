"""
Flask web application for Cognitive Bias Prediction.
"""

from flask import Flask, request, render_template, jsonify
from cognitive_bias_predictor import load_model, predict, predict_batch

app = Flask(__name__)

# Load the model once at startup
model = load_model()


@app.route("/")
def index():
    """Render the home page with the input form."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_route():
    """Accept form POST and render results page."""
    text = request.form.get("text", "").strip()
    if not text:
        return render_template("index.html", error="Please enter some text.")

    result = predict(text, model=model)
    return render_template(
        "results.html",
        text=text,
        bias=result["bias"],
        confidence=result["confidence"],
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """API endpoint: accepts JSON, returns prediction as JSON."""
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Invalid JSON body."}), 400

    # Single prediction
    if "text" in data:
        text = str(data["text"]).strip()
        if not text:
            return jsonify({"error": "The 'text' field must not be empty."}), 400
        result = predict(text, model=model)
        return jsonify({"text": text, "bias": result["bias"], "confidence": result["confidence"]})

    # Batch prediction
    if "texts" in data:
        texts = data["texts"]
        if not isinstance(texts, list) or len(texts) == 0:
            return jsonify({"error": "'texts' must be a non-empty list."}), 400
        texts = [str(t).strip() for t in texts]
        if any(t == "" for t in texts):
            return jsonify({"error": "All texts in 'texts' must be non-empty."}), 400
        results = predict_batch(texts, model=model)
        return jsonify({"predictions": results})

    return jsonify({"error": "Request body must contain 'text' or 'texts'."}), 400


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found."}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error."}), 500


if __name__ == "__main__":
    app.run(debug=False)
