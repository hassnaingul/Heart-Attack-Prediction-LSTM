"""LSTM inference module for the Heart Attack Risk Prediction System.

The prediction logic is taken unchanged from the original ``app_lstm.py``:

* the same 25 raw input features, in the same order,
* the same ``preprocessor.pkl`` pipeline (StandardScaler on 20 numeric columns,
  OneHotEncoder(drop="first") on 5 categorical columns -> 48 features),
* the same reshape to a 3-D LSTM tensor ``(samples, 1, 48)``,
* the same classification threshold of 0.5.

Changes made for the web application (model logic is NOT changed):

* Artifact paths are resolved relative to this file, so the app runs from any
  working directory (required on Render and for Windows development).
* The preprocessor and model are loaded once through ``load_artifacts()``
  instead of at import time. Import is cheap and the load can be reported
  and tested.
* ``model.predict`` is wrapped in a lock. Keras inference is not guaranteed to
  be thread-safe when Gunicorn runs several threads.
* ``compile=False`` is passed to ``load_model`` because only inference is
  performed; this avoids needing the training loss/optimizer configuration.
"""
from __future__ import annotations

import os
import pickle
import threading
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent
PREPROCESSOR_PATH = BASE_DIR / "preprocessor.pkl"
MODEL_PATH = BASE_DIR / "lstm_model.keras"

CLASSIFICATION_THRESHOLD = 0.5

# Exact feature order used by the original app_lstm.py.
EXPECTED_FEATURES = [
    "Age", "Cholesterol", "Heart Rate", "Diabetes", "Family History",
    "Smoking", "Obesity", "Alcohol Consumption", "Exercise Hours Per Week",
    "Previous Heart Problems", "Medication Use", "Stress Level",
    "Sedentary Hours Per Day", "Income", "BMI", "Triglycerides",
    "Physical Activity Days Per Week", "Sleep Hours Per Day",
    "Systolic BP", "Diastolic BP", "Sex", "Diet", "Country",
    "Continent", "Hemisphere",
]

_artifacts: dict = {"preprocessor": None, "model": None}
_load_lock = threading.Lock()
_predict_lock = threading.Lock()


def is_loaded() -> bool:
    """Return True when the preprocessor and model are in memory."""
    return _artifacts["model"] is not None


def load_artifacts() -> dict:
    """Load the preprocessor and LSTM model once per process and return them."""
    if _artifacts["model"] is not None:
        return _artifacts
    with _load_lock:
        if _artifacts["model"] is None:
            import tensorflow as tf  # deferred: heavy import

            with open(PREPROCESSOR_PATH, "rb") as f:
                preprocessor = pickle.load(f)
            model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            _artifacts["preprocessor"] = preprocessor
            _artifacts["model"] = model
    return _artifacts


def predict_heart_attack_risk_lstm(raw_patient_data):
    """Run the existing preprocessing pipeline and LSTM model on one patient.

    ``raw_patient_data`` is a dict (or list of dicts) keyed by the raw feature
    names. ``Blood Pressure`` as ``"130/85"`` is still accepted, exactly as in
    the original implementation.
    """
    artifacts = load_artifacts()
    preprocessor = artifacts["preprocessor"]
    lstm_model = artifacts["model"]

    if isinstance(raw_patient_data, dict):
        df_input = pd.DataFrame([raw_patient_data])
    else:
        df_input = pd.DataFrame(raw_patient_data)

    # Process Blood Pressure (original logic)
    if "Blood Pressure" in df_input.columns and "Systolic BP" not in df_input.columns:
        df_input[["Systolic BP", "Diastolic BP"]] = (
            df_input["Blood Pressure"].str.split("/", expand=True).astype(int)
        )
        df_input = df_input.drop(columns=["Blood Pressure"])

    df_input = df_input[EXPECTED_FEATURES]

    # Apply scaler and encoder (existing preprocessor.pkl)
    processed_features = preprocessor.transform(df_input)

    # Reshape to 3-D LSTM input: [samples, time_steps, features]
    processed_features_3d = np.reshape(
        processed_features, (processed_features.shape[0], 1, processed_features.shape[1])
    )

    with _predict_lock:
        prediction_prob = lstm_model.predict(processed_features_3d, verbose=0).flatten()[0]
    prediction_class = int(prediction_prob > CLASSIFICATION_THRESHOLD)

    return {
        "predicted_class": prediction_class,
        "probability_of_risk": float(prediction_prob),
    }


if __name__ == "__main__":
    test_patient = {
        "Age": 50,
        "Sex": "Male",
        "Cholesterol": 220,
        "Blood Pressure": "130/85",
        "Heart Rate": 75,
        "Diabetes": 0,
        "Family History": 1,
        "Smoking": 0,
        "Obesity": 0,
        "Alcohol Consumption": 1,
        "Exercise Hours Per Week": 4.5,
        "Diet": "Healthy",
        "Previous Heart Problems": 0,
        "Medication Use": 0,
        "Stress Level": 4,
        "Sedentary Hours Per Day": 5.0,
        "Income": 75000,
        "BMI": 24.5,
        "Triglycerides": 150,
        "Physical Activity Days Per Week": 3,
        "Sleep Hours Per Day": 7,
        "Country": "Canada",
        "Continent": "North America",
        "Hemisphere": "Northern Hemisphere",
    }
    print("LSTM Model Prediction Result:", predict_heart_attack_risk_lstm(test_patient))
