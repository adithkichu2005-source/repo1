# Cognitive Bias Predictor – Flask Web Application

A Flask-based web application that predicts cognitive biases in text using a trained machine learning model.

## Supported Bias Types

| Bias | Description |
|------|-------------|
| `overgeneralization` | Drawing broad conclusions from a single event |
| `catastrophizing` | Expecting the worst-case outcome |
| `mind_reading` | Assuming you know what others are thinking |
| `black_and_white_thinking` | Seeing things in all-or-nothing terms |
| `emotional_reasoning` | Treating feelings as facts |

## Project Structure

```
├── app.py                      # Flask application
├── cognitive_bias_predictor.py # Model training and inference
├── cognitive_bias_dataset.csv  # Training dataset
├── cognitive_bias_model.pkl    # Saved model (generated on first run)
├── templates/
│   ├── index.html              # Home page with input form
│   └── results.html            # Results display page
├── requirements.txt
└── README.md
```

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# (Optional) Train the model explicitly
python cognitive_bias_predictor.py

# Start the Flask app (model is trained automatically if not found)
python app.py
```

Then open <http://localhost:5000> in your browser.

## API Reference

### Single prediction

```
POST /api/predict
Content-Type: application/json

{ "text": "I always mess everything up!" }
```

**Response**
```json
{
  "text": "I always mess everything up!",
  "bias": "overgeneralization",
  "confidence": 0.8234
}
```

### Batch prediction

```
POST /api/predict
Content-Type: application/json

{ "texts": ["I always fail.", "Everyone hates me."] }
```

**Response**
```json
{
  "predictions": [
    { "text": "I always fail.", "bias": "overgeneralization", "confidence": 0.75 },
    { "text": "Everyone hates me.", "bias": "mind_reading", "confidence": 0.81 }
  ]
}
```

## Error Handling

All API errors return JSON with an `"error"` key and an appropriate HTTP status code.
