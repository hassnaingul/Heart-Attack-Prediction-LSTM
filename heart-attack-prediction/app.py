"""Flask application for the Heart Attack Risk Prediction System.

Local development (Windows, macOS, Linux):
    python app.py            -> http://127.0.0.1:5000

Production (Render / any Gunicorn host):
    gunicorn "app:create_app()" -c gunicorn.conf.py
"""
from __future__ import annotations

import logging
import os
import secrets
from pathlib import Path

from flask import (Flask, Response, abort, flash, jsonify, redirect,
                   render_template, request, url_for)
from werkzeug.exceptions import HTTPException

import app_lstm
from app_lstm import CLASSIFICATION_THRESHOLD, load_artifacts, predict_heart_attack_risk_lstm
from utils import helpers

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = BASE_DIR / "instance" / "predictions.db"


def create_app(test_config: dict | None = None) -> Flask:
    app = Flask(__name__)
    app.config.update(
        # Set SECRET_KEY in the environment in production. A random key is
        # generated otherwise (sessions reset on restart).
        SECRET_KEY=os.environ.get("SECRET_KEY") or secrets.token_hex(32),
        DATABASE=os.environ.get("DATABASE_PATH") or str(DEFAULT_DB_PATH),
        MAX_CONTENT_LENGTH=64 * 1024,  # reject oversized form posts
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=os.environ.get("SESSION_COOKIE_SECURE", "0") == "1",
    )
    if test_config:
        app.config.update(test_config)

    logging.basicConfig(level=logging.INFO)
    app.jinja_env.filters["datetime"] = helpers.format_datetime

    db_path = app.config["DATABASE"]
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    helpers.init_db(db_path)

    # Load the existing LSTM model and preprocessor ONCE at startup.
    try:
        load_artifacts()
    except Exception:
        app.logger.exception("Could not load model artifacts (lstm_model.keras / preprocessor.pkl)")
        raise

    # ------------------------------------------------------------------ #
    # Security & template globals
    # ------------------------------------------------------------------ #
    @app.before_request
    def csrf_protect():
        if request.method == "POST":
            expected = helpers.get_csrf_token()
            sent = request.form.get("csrf_token", "")
            if not secrets.compare_digest(expected, sent):
                abort(400, description="Your session security token is missing or expired. "
                                       "Please reload the page and try again.")

    @app.context_processor
    def inject_globals():
        return {
            "csrf_token": helpers.get_csrf_token,
            "threshold": CLASSIFICATION_THRESHOLD,
            "disclaimer": helpers.DISCLAIMER,
        }

    @app.after_request
    def security_headers(response):
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "same-origin")
        return response

    # ------------------------------------------------------------------ #
    # Pages
    # ------------------------------------------------------------------ #
    @app.get("/")
    def index():
        stats = helpers.get_stats(db_path, recent_limit=3)
        return render_template("index.html", stats=stats, model_name=helpers.MODEL_NAME)

    @app.get("/dashboard")
    def dashboard():
        stats = helpers.get_stats(db_path, recent_limit=5)
        chart_data = {k: stats[k] for k in ("positive", "negative", "labels", "bins")}
        return render_template("dashboard.html", stats=stats, chart_data=chart_data,
                               model=helpers.get_model_info())

    @app.get("/patient-info")
    def patient_info():
        return render_template("patient_info.html", sections=helpers.form_sections(),
                               values={}, errors=[])

    @app.route("/predict", methods=["GET", "POST"])
    def predict():
        if request.method == "GET":
            latest = helpers.list_predictions(db_path, limit=1)
            return render_template("prediction.html", model=helpers.get_model_info(),
                                   latest=latest[0] if latest else None,
                                   model_ready=app_lstm.is_loaded())

        values, errors = helpers.validate_patient_form(request.form)
        if errors:
            return render_template("patient_info.html", sections=helpers.form_sections(),
                                   values=request.form, errors=errors), 400

        try:
            result = predict_heart_attack_risk_lstm(values)
        except Exception:
            app.logger.exception("Prediction failed")
            return render_template(
                "patient_info.html", sections=helpers.form_sections(), values=request.form,
                errors=["The prediction service could not process these values. "
                        "Please review the inputs and try again."],
            ), 500

        record_id = helpers.save_prediction(db_path, values, result)
        flash("Prediction completed and saved to your history.", "success")
        return redirect(url_for("result", record_id=record_id))

    @app.get("/result/<int:record_id>")
    def result(record_id: int):
        record = helpers.get_prediction(db_path, record_id)
        if record is None:
            abort(404, description="That prediction could not be found.")
        return render_template("result.html", record=record,
                               groups=helpers.describe_inputs(record["inputs"]))

    @app.get("/result/<int:record_id>/report")
    def download_report(record_id: int):
        record = helpers.get_prediction(db_path, record_id)
        if record is None:
            abort(404, description="That prediction could not be found.")
        filename = f"prediction_report_{int(record_id)}.txt"  # id is an int: safe filename
        return Response(
            helpers.build_report_text(record),
            mimetype="text/plain",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.get("/history")
    def history():
        return render_template("history.html", records=helpers.list_predictions(db_path))

    @app.post("/history/<int:record_id>/delete")
    def delete_record(record_id: int):
        if helpers.delete_prediction(db_path, record_id):
            flash("Prediction record deleted.", "success")
        else:
            flash("That record no longer exists.", "warning")
        return redirect(url_for("history"))

    @app.get("/model-info")
    def model_info():
        return render_template("model_info.html", model=helpers.get_model_info())

    @app.get("/about")
    def about():
        return render_template("about.html")

    @app.get("/health")
    def health():
        return jsonify(status="ok", model_loaded=app_lstm.is_loaded(), model=helpers.MODEL_NAME)

    # ------------------------------------------------------------------ #
    # Errors
    # ------------------------------------------------------------------ #
    @app.errorhandler(HTTPException)
    def handle_http_error(err: HTTPException):
        if request.path == "/health":
            return jsonify(status="error", code=err.code), err.code
        return render_template("error.html", code=err.code, name=err.name,
                               message=err.description), err.code

    @app.errorhandler(Exception)
    def handle_unexpected(err: Exception):
        app.logger.exception("Unhandled error: %s", err)
        return render_template("error.html", code=500, name="Internal Server Error",
                               message="Something went wrong on our side. Please try again."), 500

    return app


if __name__ == "__main__":
    application = create_app()
    port = int(os.environ.get("PORT", "5000"))
    application.run(host="127.0.0.1", port=port, debug=os.environ.get("FLASK_DEBUG") == "1")
