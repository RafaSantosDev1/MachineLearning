
import csv
import hmac
import secrets
import threading

from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from pathlib import Path

from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    session
)

from rafiai.model_storage import model_exists, load_model
from rafiai.predict import predict_sentiment
from rafiai.feedback_storage import (
    save_feedback,
    get_pending_feedback,
    clear_feedback
)
from rafiai.development import run_development


app = Flask(__name__)

app.secret_key = secrets.token_urlsafe(32)

app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax"
)


training_status = {
    "running": False,
    "output": "",
    "error": None
}

training_lock = threading.Lock()


def get_project_root():
    return Path(__file__).resolve().parents[3]


def get_admin_credentials():
    credentials_path = get_project_root() / "admin.txt"

    if not credentials_path.exists():
        raise FileNotFoundError("admin.txt was not found.")

    credentials = {}

    with credentials_path.open("r", encoding="utf-8") as file:
        for line in file:
            if "=" not in line:
                continue

            key, value = line.strip().split("=", 1)
            credentials[key.strip()] = value.strip()

    username = credentials.get("username")
    password = credentials.get("password")

    if not username or not password:
        raise ValueError("Invalid admin.txt configuration.")

    return username, password


def is_admin():
    return session.get("is_admin") is True


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

    if prediction is None:
        return jsonify({
            "unknown": True
        })

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


# ADMIN AUTHENTICATION

@app.route("/admin/login", methods=["POST"])
def admin_login():
    username = request.form.get("username", "").strip()
    password = request.form.get("password", "")

    try:
        correct_username, correct_password = get_admin_credentials()

    except (FileNotFoundError, ValueError):
        return jsonify({
            "error": "Admin credentials are not configured correctly."
        }), 500

    username_valid = hmac.compare_digest(
        username,
        correct_username
    )

    password_valid = hmac.compare_digest(
        password,
        correct_password
    )

    if not username_valid or not password_valid:
        return jsonify({
            "error": "Invalid username or password."
        }), 401

    session.clear()
    session["is_admin"] = True

    return jsonify({
        "message": "Admin login successful.",
        "pending_count": len(get_pending_feedback())
    })


@app.route("/admin/logout", methods=["POST"])
def admin_logout():
    session.clear()

    return jsonify({
        "message": "Admin session closed."
    })


# IMPORT FEEDBACK INTO TRAINING CSV

@app.route("/admin/import", methods=["POST"])
def admin_import():
    if not is_admin():
        return jsonify({
            "error": "Admin authentication required."
        }), 401

    project_root = get_project_root()
    csv_path = project_root / "data" / "training.csv"

    if not csv_path.exists():
        return jsonify({
            "error": "training.csv was not found."
        }), 404

    pending_feedback = get_pending_feedback()

    if not pending_feedback:
        return jsonify({
            "message": "There are no new examples to import.",
            "imported_count": 0,
            "skipped_count": 0,
            "pending_count": 0
        })

    try:
        with csv_path.open(
            "r",
            encoding="utf-8-sig",
            newline=""
        ) as file:
            reader = csv.DictReader(file)

            if not reader.fieldnames:
                raise ValueError("The CSV has no header.")

            if not {"text", "label"}.issubset(reader.fieldnames):
                raise ValueError(
                    "training.csv must contain text and label columns."
                )

            existing_rows = list(reader)
            fieldnames = reader.fieldnames

        existing_examples = {
            (
                row["text"].strip(),
                row["label"].strip().lower()
            )
            for row in existing_rows
        }

        new_rows = []
        skipped_count = 0

        for item in pending_feedback:
            text = item["text"].strip()
            label = item["correct_label"].strip().lower()

            if not text or label not in ("positive", "negative"):
                continue

            example = (text, label)

            if example in existing_examples:
                skipped_count += 1
                continue

            new_rows.append({
                "text": text,
                "label": label
            })

            existing_examples.add(example)

        if new_rows:
            needs_newline = False

            with csv_path.open("rb") as file:
                file.seek(0, 2)
                file_size = file.tell()

                if file_size > 0:
                    file.seek(-1, 2)
                    needs_newline = file.read(1) not in (b"\n", b"\r")

            with csv_path.open(
                "a",
                encoding="utf-8",
                newline=""
            ) as file:

                if needs_newline:
                    file.write("\n")

                writer = csv.DictWriter(
                    file,
                    fieldnames=fieldnames,
                    quoting=csv.QUOTE_ALL
                )

                writer.writerows(new_rows)

        clear_feedback()

        return jsonify({
            "message": "Knowledge import completed successfully.",
            "imported_count": len(new_rows),
            "skipped_count": skipped_count,
            "pending_count": 0
        })

    except (OSError, csv.Error, ValueError, KeyError) as error:
        return jsonify({
            "error": str(error)
        }), 500


# DEVELOPMENT MODE

def execute_development():
    output = StringIO()

    try:
        with redirect_stdout(output), redirect_stderr(output):
            run_development()

        training_status["output"] = output.getvalue()
        training_status["error"] = None

    except Exception as error:
        training_status["output"] = output.getvalue()
        training_status["error"] = str(error)

    finally:
        training_status["running"] = False


@app.route("/admin/development", methods=["POST"])
def admin_development():
    if not is_admin():
        return jsonify({
            "error": "Admin authentication required."
        }), 401

    with training_lock:
        if training_status["running"]:
            return jsonify({
                "error": "Development Mode is already running."
            }), 409

        training_status["running"] = True
        training_status["output"] = ""
        training_status["error"] = None

    thread = threading.Thread(
        target=execute_development,
        daemon=True
    )

    thread.start()

    return jsonify({
        "message": "Development Mode started successfully."
    })


@app.route("/admin/development/status", methods=["GET"])
def admin_development_status():
    if not is_admin():
        return jsonify({
            "error": "Admin authentication required."
        }), 401

    return jsonify({
        "running": training_status["running"],
        "output": training_status["output"],
        "error": training_status["error"]
    })


def run_web():
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
