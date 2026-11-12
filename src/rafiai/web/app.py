
from flask import Flask, render_template, request, jsonify

from rafiai.model_storage import model_exists, load_model
from rafiai.predict import predict_sentiment
from rafiai.feedback_storage import save_feedback


app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        feedback_saved=request.args.get("feedback_saved") == "true"
    )


@app.route("/predict", methods=["POST"])
def predict():
    user_text = request.form.get("text", "").strip()

    if not user_text:
        return render_template(
            "index.html",
            error="Please enter a sentence."
        )

    if not model_exists():
        return render_template(
            "index.html",
            error="No trained model found. Run Development Mode first."
        )

    saved_data = load_model()

    model = saved_data["model"]
    vectorizer = saved_data["vectorizer"]

    prediction, probabilities = predict_sentiment(
        model,
        vectorizer,
        user_text
    )

    if prediction is None:
        return render_template(
            "index.html",
            error="RafiAI does not recognize the words in this sentence."
        )

    return render_template(
        "prediction.html",
        text=user_text,
        prediction=prediction,
        probabilities=probabilities
    )


@app.route("/predict-feedback", methods=["POST"])
def predict_feedback():
    user_text = request.form.get("text", "").strip()

    if not user_text:
        return jsonify({
            "error": "Please enter a sentence."
        }), 400

    if not model_exists():
        return jsonify({
            "error": "No trained model found. Run Development Mode first."
        }), 400

    saved_data = load_model()

    model = saved_data["model"]
    vectorizer = saved_data["vectorizer"]

    prediction, probabilities = predict_sentiment(
        model,
        vectorizer,
        user_text
    )

    # The model does not recognize any words in the sentence.
    if prediction is None:
        return jsonify({
            "unknown": True
        })

    # The model successfully recognizes the sentence.
    return jsonify({
        "unknown": False,
        "prediction": str(prediction),
        "probabilities": {
            str(sentiment): float(probability)
            for sentiment, probability in probabilities.items()
        }
    })


@app.route("/feedback", methods=["POST"])
def feedback():
    user_text = request.form.get("text", "").strip()

    correct_label = request.form.get(
        "correct_label", ""
    ).strip().lower()

    valid_labels = ["positive", "negative"]

    if not user_text or correct_label not in valid_labels:
        return jsonify({
            "error": "Please enter a sentence and select a valid sentiment."
        }), 400

    predicted_label = "unknown"

    if model_exists():
        saved_data = load_model()

        model = saved_data["model"]
        vectorizer = saved_data["vectorizer"]

        prediction, _ = predict_sentiment(
            model,
            vectorizer,
            user_text
        )

        if prediction is not None:
            predicted_label = str(prediction).lower()

    save_feedback(
        user_text,
        predicted_label,
        correct_label
    )

    return jsonify({
        "message": "Your feedback has been saved successfully."
    })


def run_web():
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
