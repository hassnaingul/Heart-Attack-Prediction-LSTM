"""Tests for the Heart Attack Risk Prediction System.

Run from the project root:
    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import app_lstm  # noqa: E402
from app import create_app  # noqa: E402
from utils import helpers  # noqa: E402

# Test patient copied from the ORIGINAL app_lstm.py (__main__ block).
ORIGINAL_TEST_PATIENT = {
    "Age": 50, "Sex": "Male", "Cholesterol": 220, "Blood Pressure": "130/85",
    "Heart Rate": 75, "Diabetes": 0, "Family History": 1, "Smoking": 0, "Obesity": 0,
    "Alcohol Consumption": 1, "Exercise Hours Per Week": 4.5, "Diet": "Healthy",
    "Previous Heart Problems": 0, "Medication Use": 0, "Stress Level": 4,
    "Sedentary Hours Per Day": 5.0, "Income": 75000, "BMI": 24.5, "Triglycerides": 150,
    "Physical Activity Days Per Week": 3, "Sleep Hours Per Day": 7, "Country": "Canada",
    "Continent": "North America", "Hemisphere": "Northern Hemisphere",
}
# Output of the ORIGINAL implementation for the patient above (verified during Phase 1).
REFERENCE_PROBABILITY = 0.35555168986320496

# The same patient, as the web form would submit it.
FORM_VALUES = {
    "age": "50", "sex": "Male", "income": "75000",
    "cholesterol": "220", "systolic_bp": "130", "diastolic_bp": "85", "heart_rate": "75",
    "triglycerides": "150", "bmi": "24.5", "obesity": "0", "diabetes": "0",
    "family_history": "1", "previous_heart_problems": "0", "medication_use": "0",
    "smoking": "0", "alcohol": "1", "exercise_hours": "4.5", "physical_activity_days": "3",
    "sedentary_hours": "5", "sleep_hours": "7", "diet": "Healthy", "stress_level": "4",
    "country": "Canada", "continent": "North America", "hemisphere": "Northern Hemisphere",
}


class ModelPipelineTests(unittest.TestCase):
    """Checks that the existing model and preprocessing are used unchanged."""

    def test_output_matches_original_implementation(self):
        result = app_lstm.predict_heart_attack_risk_lstm(ORIGINAL_TEST_PATIENT)
        self.assertAlmostEqual(result["probability_of_risk"], REFERENCE_PROBABILITY, places=6)
        self.assertEqual(result["predicted_class"], 0)

    def test_form_values_give_same_prediction_as_original(self):
        values, errors = helpers.validate_patient_form(FORM_VALUES)
        self.assertEqual(errors, [])
        result = app_lstm.predict_heart_attack_risk_lstm(values)
        self.assertAlmostEqual(result["probability_of_risk"], REFERENCE_PROBABILITY, places=6)

    def test_feature_contract(self):
        self.assertEqual(len(app_lstm.EXPECTED_FEATURES), 25)
        expected = set(ORIGINAL_TEST_PATIENT) - {"Blood Pressure"} | {"Systolic BP", "Diastolic BP"}
        self.assertEqual(set(app_lstm.EXPECTED_FEATURES), expected)
        values, _ = helpers.validate_patient_form(FORM_VALUES)
        self.assertEqual(set(values), expected)

    def test_model_input_shape_and_threshold(self):
        info = helpers.get_model_info()
        self.assertEqual(info["input_shape"], "(None, 1, 48)")
        self.assertEqual(info["output_activation"], "sigmoid")
        self.assertEqual(info["processed_features"], 48)
        self.assertEqual(app_lstm.CLASSIFICATION_THRESHOLD, 0.5)

    def test_blood_pressure_string_still_supported(self):
        result = app_lstm.predict_heart_attack_risk_lstm(ORIGINAL_TEST_PATIENT)
        self.assertIn("predicted_class", result)


class ValidationTests(unittest.TestCase):
    def test_valid_form_has_no_errors(self):
        values, errors = helpers.validate_patient_form(FORM_VALUES)
        self.assertEqual(errors, [])
        self.assertEqual(values["Systolic BP"], 130)
        self.assertEqual(values["Sex"], "Male")

    def test_missing_field_is_reported(self):
        form = dict(FORM_VALUES)
        del form["cholesterol"]
        _, errors = helpers.validate_patient_form(form)
        self.assertTrue(any("Cholesterol is required" in e for e in errors))

    def test_out_of_range_value_is_rejected(self):
        form = dict(FORM_VALUES, age="150")
        _, errors = helpers.validate_patient_form(form)
        self.assertTrue(any("Age must be between" in e for e in errors))

    def test_blood_pressure_order_is_checked(self):
        form = dict(FORM_VALUES, systolic_bp="80", diastolic_bp="90")
        _, errors = helpers.validate_patient_form(form)
        self.assertTrue(any("Systolic blood pressure must be higher" in e for e in errors))

    def test_integer_fields_reject_fractions(self):
        form = dict(FORM_VALUES, stress_level="4.5")
        _, errors = helpers.validate_patient_form(form)
        self.assertTrue(any("whole number" in e for e in errors))

    def test_unknown_category_is_rejected(self):
        form = dict(FORM_VALUES, country="Narnia")
        _, errors = helpers.validate_patient_form(form)
        self.assertTrue(any("valid option" in e for e in errors))

    def test_non_numeric_value_is_rejected(self):
        form = dict(FORM_VALUES, bmi="abc")
        _, errors = helpers.validate_patient_form(form)
        self.assertTrue(any("must be a number" in e for e in errors))

    def test_form_has_25_fields_covering_25_model_inputs(self):
        # 24 patient items; Blood Pressure is two fields (systolic, diastolic).
        self.assertEqual(len(helpers.FIELDS), 25)
        self.assertEqual(len({f["model"] for f in helpers.FIELDS}), 25)


class WebAppTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self._tmp.name) / "test.db")
        self.app = create_app({"TESTING": True, "DATABASE": self.db_path})
        self.client = self.app.test_client()
        self.token = "test-token"
        with self.client.session_transaction() as sess:
            sess["_csrf_token"] = self.token

    def tearDown(self):
        self._tmp.cleanup()

    def post_form(self, data, token=None):
        payload = dict(data)
        payload["csrf_token"] = self.token if token is None else token
        return self.client.post("/predict", data=payload)

    def test_pages_render(self):
        for path in ["/", "/dashboard", "/patient-info", "/predict", "/history", "/model-info", "/about"]:
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertEqual(response.status_code, 200)
                self.assertIn(b"Heart Attack Risk Prediction", response.data)

    def test_health_endpoint(self):
        data = json.loads(self.client.get("/health").data)
        self.assertEqual(data["status"], "ok")
        self.assertTrue(data["model_loaded"])

    def test_form_contains_all_sections_and_fields(self):
        html = self.client.get("/patient-info").get_data(as_text=True)
        for title in ["Personal Information", "Medical Information", "Lifestyle Information",
                      "Health and Environmental Information"]:
            self.assertIn(title, html)
        for field in helpers.FIELDS:
            self.assertIn(f'name="{field["name"]}"', html)

    def test_post_without_csrf_token_is_rejected(self):
        response = self.post_form(FORM_VALUES, token="wrong")
        self.assertEqual(response.status_code, 400)

    def test_invalid_submission_shows_errors(self):
        response = self.post_form(dict(FORM_VALUES, systolic_bp="100", diastolic_bp="110"))
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"Systolic blood pressure must be higher", response.data)

    def test_full_prediction_flow(self):
        response = self.post_form(FORM_VALUES)
        self.assertEqual(response.status_code, 302)
        self.assertTrue(response.headers["Location"].endswith("/result/1"))

        result_html = self.client.get("/result/1").get_data(as_text=True)
        self.assertIn("0.3556", result_html)
        self.assertIn("not a medical diagnosis", result_html.lower())
        self.assertIn("No elevated model-indicated risk", result_html)
        self.assertIn("Not a clinically validated risk percentage", result_html)

        report = self.client.get("/result/1/report")
        self.assertEqual(report.status_code, 200)
        self.assertIn("attachment", report.headers["Content-Disposition"])
        self.assertIn("prediction_report_1.txt", report.headers["Content-Disposition"])
        text = report.get_data(as_text=True)
        self.assertIn("Output probability     : 0.3556", text)
        self.assertIn("IMPORTANT DISCLAIMER", text)

    def test_dashboard_and_history_reflect_saved_predictions(self):
        self.post_form(FORM_VALUES)
        html = self.client.get("/dashboard").get_data(as_text=True)
        match = re.search(r'id="chart-data">(.*?)</script>', html, re.S)
        self.assertIsNotNone(match)
        chart = json.loads(match.group(1))
        self.assertEqual(chart["negative"], 1)
        self.assertEqual(chart["positive"], 0)
        self.assertEqual(sum(chart["bins"]), 1)
        self.assertIn(b"0.356", self.client.get("/history").data)
        self.assertIn(b"Total predictions", self.client.get("/dashboard").data)

    def test_delete_requires_token_and_removes_record(self):
        self.post_form(FORM_VALUES)
        bad = self.client.post("/history/1/delete", data={"csrf_token": "nope"})
        self.assertEqual(bad.status_code, 400)
        ok = self.client.post("/history/1/delete", data={"csrf_token": self.token})
        self.assertEqual(ok.status_code, 302)
        self.assertEqual(self.client.get("/result/1").status_code, 404)

    def test_unknown_result_returns_404(self):
        self.assertEqual(self.client.get("/result/999").status_code, 404)
        self.assertEqual(self.client.get("/result/999/report").status_code, 404)

    def test_non_integer_record_id_is_not_routed(self):
        self.assertEqual(self.client.get("/result/abc").status_code, 404)


if __name__ == "__main__":
    unittest.main(verbosity=2)
