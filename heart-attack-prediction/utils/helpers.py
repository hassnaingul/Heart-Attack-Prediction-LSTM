"""Shared helpers: form definition, validation, SQLite storage, reports, CSRF.

Plausibility limits in FIELDS are input guards for the web form. They are NOT
the training-data ranges of the model, which were not supplied with the project.
"""
from __future__ import annotations

import json
import math
import secrets
import sqlite3
from contextlib import closing
from datetime import datetime
from typing import Any, Iterable, Mapping

from flask import session

from app_lstm import CLASSIFICATION_THRESHOLD, MODEL_PATH, load_artifacts

MODEL_NAME = "LSTM"
MODEL_TYPE = "Deep Learning Classification (binary, sigmoid output)"
DISCLAIMER = (
    "This is an experimental AI prediction produced by a research model. It is "
    "NOT a medical diagnosis and must not be used to make clinical decisions. "
    "Please consult a qualified healthcare professional for any concern about "
    "heart health."
)

SEXES = ["Female", "Male"]
DIETS = ["Average", "Healthy", "Unhealthy"]
CONTINENTS = ["Africa", "Asia", "Australia", "Europe", "North America", "South America"]
HEMISPHERES = ["Northern Hemisphere", "Southern Hemisphere"]
COUNTRIES = [
    "Argentina", "Australia", "Brazil", "Canada", "China", "Colombia", "France",
    "Germany", "India", "Italy", "Japan", "New Zealand", "Nigeria", "South Africa",
    "South Korea", "Spain", "Thailand", "United Kingdom", "United States", "Vietnam",
]

SECTION_DEFS = [
    ("personal", "Personal Information", "bi-person-vcard",
     "Demographic details used as model inputs."),
    ("medical", "Medical Information", "bi-heart-pulse",
     "Clinical measurements and medical history."),
    ("lifestyle", "Lifestyle Information", "bi-activity",
     "Daily habits and routines."),
    ("environment", "Health and Environmental Information", "bi-globe-americas",
     "Location and environmental context used by the model."),
]


def _f(name, model, label, section, kind, help_text,
       min_=None, max_=None, unit="", options=None):
    return {
        "name": name, "model": model, "label": label, "section": section,
        "kind": kind, "help": help_text, "min": min_, "max": max_,
        "unit": unit, "options": options or [],
    }


# 24 patient items; Blood Pressure is entered as two fields -> 25 form fields = 25 model inputs.
FIELDS = [
    _f("age", "Age", "Age", "personal", "int", "Patient age in years.", 18, 100, "years"),
    _f("sex", "Sex", "Sex", "personal", "select", "Biological sex, as encoded for the model.", options=SEXES),
    _f("income", "Income", "Annual income", "personal", "float",
       "Annual income in the same currency the model was trained on.", 0, 10_000_000, "per year"),

    _f("cholesterol", "Cholesterol", "Cholesterol", "medical", "float", "Total cholesterol level.", 50, 500, "mg/dL"),
    _f("systolic_bp", "Systolic BP", "Systolic blood pressure", "medical", "int",
       "Upper blood pressure value. Must be higher than diastolic.", 70, 250, "mmHg"),
    _f("diastolic_bp", "Diastolic BP", "Diastolic blood pressure", "medical", "int",
       "Lower blood pressure value.", 40, 150, "mmHg"),
    _f("heart_rate", "Heart Rate", "Resting heart rate", "medical", "int", "Resting heart rate.", 30, 200, "bpm"),
    _f("triglycerides", "Triglycerides", "Triglycerides", "medical", "float", "Triglyceride level.", 20, 2000, "mg/dL"),
    _f("bmi", "BMI", "Body Mass Index (BMI)", "medical", "float",
       "Weight in kg divided by height in metres squared.", 10, 70, "kg/m²"),
    _f("obesity", "Obesity", "Obesity", "medical", "binary", "Clinically classified as obese."),
    _f("diabetes", "Diabetes", "Diabetes", "medical", "binary", "Has a diabetes diagnosis."),
    _f("family_history", "Family History", "Family history of heart disease", "medical", "binary",
       "A close relative has had heart disease."),
    _f("previous_heart_problems", "Previous Heart Problems", "Previous heart problems", "medical", "binary",
       "Any prior cardiac event or diagnosis."),
    _f("medication_use", "Medication Use", "Regular medication use", "medical", "binary",
       "Currently takes regular prescribed medication."),

    _f("smoking", "Smoking", "Smoking", "lifestyle", "binary", "Currently smokes."),
    _f("alcohol", "Alcohol Consumption", "Regular alcohol consumption", "lifestyle", "binary",
       "Drinks alcohol regularly."),
    _f("exercise_hours", "Exercise Hours Per Week", "Exercise per week", "lifestyle", "float",
       "Total exercise time per week.", 0, 40, "hours/week"),
    _f("physical_activity_days", "Physical Activity Days Per Week", "Active days per week", "lifestyle", "int",
       "Number of days with physical activity.", 0, 7, "days/week"),
    _f("sedentary_hours", "Sedentary Hours Per Day", "Sedentary time per day", "lifestyle", "float",
       "Hours spent sitting or inactive per day.", 0, 24, "hours/day"),
    _f("sleep_hours", "Sleep Hours Per Day", "Sleep per day", "lifestyle", "float",
       "Average hours of sleep per day.", 0, 24, "hours/day"),
    _f("diet", "Diet", "Diet", "lifestyle", "select", "Overall diet quality.", options=DIETS),
    _f("stress_level", "Stress Level", "Stress level", "lifestyle", "int",
       "Self-reported stress: 1 = low, 10 = high.", 1, 10),

    _f("country", "Country", "Country", "environment", "select", "Patient's country.", options=COUNTRIES),
    _f("continent", "Continent", "Continent", "environment", "select",
       "Continent of the patient's location.", options=CONTINENTS),
    _f("hemisphere", "Hemisphere", "Hemisphere", "environment", "select",
       "Hemisphere of the patient's location.", options=HEMISPHERES),
]

FIELD_BY_NAME = {f["name"]: f for f in FIELDS}
FIELD_BY_MODEL = {f["model"]: f for f in FIELDS}


def form_sections() -> list[dict[str, Any]]:
    """Return the form grouped into the four required sections."""
    return [
        {"key": key, "title": title, "icon": icon, "desc": desc,
         "fields": [f for f in FIELDS if f["section"] == key]}
        for key, title, icon, desc in SECTION_DEFS
    ]


# --------------------------------------------------------------------------- #
# CSRF
# --------------------------------------------------------------------------- #
def get_csrf_token() -> str:
    token = session.get("_csrf_token")
    if not token:
        token = secrets.token_urlsafe(32)
        session["_csrf_token"] = token
    return token


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def validate_patient_form(form: Mapping[str, str]) -> tuple[dict[str, Any], list[str]]:
    """Validate submitted values. Returns (model_inputs, errors).

    ``model_inputs`` is keyed by the ORIGINAL model feature names
    (e.g. "Systolic BP", "Sex") so it can be passed straight to the model.
    """
    values: dict[str, Any] = {}
    errors: list[str] = []

    for f in FIELDS:
        raw = (form.get(f["name"]) or "").strip()
        label = f["label"]
        if raw == "":
            errors.append(f"{label} is required.")
            continue

        if f["kind"] == "select":
            if raw not in f["options"]:
                errors.append(f"{label}: please choose a valid option.")
                continue
            values[f["model"]] = raw
        elif f["kind"] == "binary":
            if raw not in ("0", "1"):
                errors.append(f"{label}: please answer Yes or No.")
                continue
            values[f["model"]] = int(raw)
        else:
            try:
                number = float(raw)
            except ValueError:
                errors.append(f"{label} must be a number.")
                continue
            if not math.isfinite(number):
                errors.append(f"{label} must be a finite number.")
                continue
            if number < f["min"] or number > f["max"]:
                unit = f" {f['unit']}" if f["unit"] else ""
                errors.append(f"{label} must be between {f['min']:g} and {f['max']:g}{unit}.")
                continue
            if f["kind"] == "int":
                if not number.is_integer():
                    errors.append(f"{label} must be a whole number.")
                    continue
                values[f["model"]] = int(number)
            else:
                values[f["model"]] = number

    if "Systolic BP" in values and "Diastolic BP" in values:
        if values["Systolic BP"] <= values["Diastolic BP"]:
            errors.append("Systolic blood pressure must be higher than diastolic blood pressure.")

    return values, errors


# --------------------------------------------------------------------------- #
# Display helpers
# --------------------------------------------------------------------------- #
def format_datetime(value: str | None) -> str:
    if not value:
        return "—"
    try:
        return datetime.fromisoformat(value).strftime("%d %b %Y, %I:%M %p %Z").strip()
    except ValueError:
        return value


def _display(field: dict[str, Any], value: Any) -> str:
    if field["kind"] == "binary":
        return "Yes" if int(value) == 1 else "No"
    if field["kind"] == "select":
        return str(value)
    if field["kind"] == "int":
        text = f"{int(value):,}"
    else:
        text = f"{float(value):,.2f}".rstrip("0").rstrip(".")
    return f"{text} {field['unit']}".strip()


def describe_inputs(inputs: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Group stored model inputs by form section for display."""
    groups = []
    for section in form_sections():
        rows = [
            {"label": f["label"], "value": _display(f, inputs[f["model"]])}
            for f in section["fields"]
            if f["model"] in inputs
        ]
        groups.append({"title": section["title"], "icon": section["icon"], "rows": rows})
    return groups


def risk_label(predicted_class: int) -> str:
    return ("Elevated model-indicated risk" if int(predicted_class) == 1
            else "No elevated model-indicated risk")


# --------------------------------------------------------------------------- #
# SQLite storage (local file; path comes from app config, never from users)
# --------------------------------------------------------------------------- #
SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    model_name TEXT NOT NULL,
    predicted_class INTEGER NOT NULL,
    probability REAL NOT NULL,
    risk_label TEXT NOT NULL,
    inputs_json TEXT NOT NULL
)
"""


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str) -> None:
    with closing(_connect(db_path)) as conn, conn:
        conn.execute(SCHEMA)


def _row_to_record(row: sqlite3.Row) -> dict[str, Any]:
    record = dict(row)
    record["inputs"] = json.loads(record.pop("inputs_json"))
    record["created_display"] = format_datetime(record["created_at"])
    return record


def save_prediction(db_path: str, inputs: Mapping[str, Any], result: Mapping[str, Any]) -> int:
    created_at = datetime.now().astimezone().isoformat(timespec="seconds")
    predicted_class = int(result["predicted_class"])
    with closing(_connect(db_path)) as conn, conn:
        cur = conn.execute(
            "INSERT INTO predictions (created_at, model_name, predicted_class, probability, risk_label, inputs_json) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (created_at, MODEL_NAME, predicted_class, float(result["probability_of_risk"]),
             risk_label(predicted_class), json.dumps(dict(inputs))),
        )
        return int(cur.lastrowid)


def get_prediction(db_path: str, record_id: int) -> dict[str, Any] | None:
    with closing(_connect(db_path)) as conn:
        row = conn.execute("SELECT * FROM predictions WHERE id = ?", (record_id,)).fetchone()
    return _row_to_record(row) if row else None


def list_predictions(db_path: str, limit: int | None = None) -> list[dict[str, Any]]:
    sql = "SELECT * FROM predictions ORDER BY id DESC"
    params: tuple = ()
    if limit is not None:
        sql += " LIMIT ?"
        params = (int(limit),)
    with closing(_connect(db_path)) as conn:
        rows = conn.execute(sql, params).fetchall()
    return [_row_to_record(r) for r in rows]


def delete_prediction(db_path: str, record_id: int) -> bool:
    with closing(_connect(db_path)) as conn, conn:
        cur = conn.execute("DELETE FROM predictions WHERE id = ?", (record_id,))
        return cur.rowcount > 0


def get_stats(db_path: str, recent_limit: int = 5) -> dict[str, Any]:
    records = list_predictions(db_path)
    total = len(records)
    positive = sum(1 for r in records if r["predicted_class"] == 1)
    bins = [0] * 10
    for r in records:
        bins[min(int(r["probability"] * 10), 9)] += 1
    return {
        "total": total,
        "positive": positive,
        "negative": total - positive,
        "labels": [f"{i / 10:.1f}–{(i + 1) / 10:.1f}" for i in range(10)],
        "bins": bins,
        "recent": records[:recent_limit],
    }


# --------------------------------------------------------------------------- #
# Model metadata (read from the loaded artifacts, not hard-coded numbers)
# --------------------------------------------------------------------------- #
def get_model_info() -> dict[str, Any]:
    artifacts = load_artifacts()
    model = artifacts["model"]
    pre = artifacts["preprocessor"]

    layers = []
    for layer in model.layers:
        try:
            out_shape = str(tuple(layer.output.shape))
        except Exception:  # pragma: no cover - defensive
            out_shape = "—"
        activation = getattr(getattr(layer, "activation", None), "__name__", None)
        layers.append({
            "name": layer.name,
            "type": type(layer).__name__,
            "output_shape": out_shape,
            "activation": activation,
            "params": int(layer.count_params()),
        })

    numeric_cols = pre.transformers_[0][2]
    categorical_cols = pre.transformers_[1][2]
    ohe = pre.named_transformers_["cat"]
    processed_width = len(pre.get_feature_names_out())

    try:
        model_path_size = MODEL_PATH.stat().st_size
    except OSError:  # pragma: no cover
        model_path_size = None

    return {
        "name": MODEL_NAME,
        "type": MODEL_TYPE,
        "file": MODEL_PATH.name,
        "file_size_kb": round(model_path_size / 1024, 1) if model_path_size else None,
        "input_shape": str(tuple(model.input_shape)),
        "output_shape": str(tuple(model.output_shape)),
        "output_activation": layers[-1]["activation"] if layers else None,
        "layers": layers,
        "total_params": int(model.count_params()),
        "raw_features": len(FIELDS),  # 25 form fields (Blood Pressure = systolic + diastolic)
        "model_features": len(pre.feature_names_in_),
        "processed_features": processed_width,
        "numeric_features": len(numeric_cols),
        "categorical_features": len(categorical_cols),
        "categorical_names": list(categorical_cols),
        "ohe_drop": str(ohe.drop),
        "threshold": CLASSIFICATION_THRESHOLD,
    }


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def build_report_text(record: Mapping[str, Any]) -> str:
    rule = "=" * 64
    lines = [
        rule,
        "HEART ATTACK RISK PREDICTION SYSTEM - PREDICTION REPORT",
        rule,
        f"Report ID              : {record['id']}",
        f"Prediction date / time : {record['created_display']}",
        f"Model                  : {MODEL_NAME} - Deep Learning Classification (binary)",
        "",
        "MODEL OUTPUT",
        "-" * 64,
        f"Output probability     : {record['probability']:.4f}  (model output score, 0-1 scale)",
        f"Decision threshold     : {CLASSIFICATION_THRESHOLD}",
        f"Classification         : {record['risk_label']} (class {record['predicted_class']})",
        "Note: the score is not a clinically validated risk percentage.",
        "",
        "PATIENT INPUTS",
        "-" * 64,
    ]
    for group in describe_inputs(record["inputs"]):
        lines.append(f"[{group['title']}]")
        for row in group["rows"]:
            lines.append(f"  {row['label']:<38}{row['value']}")
        lines.append("")
    lines += [
        rule,
        "IMPORTANT DISCLAIMER",
        rule,
        DISCLAIMER,
        "",
    ]
    return "\n".join(lines)
