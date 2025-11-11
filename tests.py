# tests.py
import pytest
import pandas as pd
import numpy as np
import torch

from app.main import predict_failure, LogInput
from app.feature_extraction import encode_text, encode_metadata, combine_features
from app.model import load_model, load_encoder

model = load_model()
encoder = load_encoder()

def make_log(message="Test failed: AssertionError"):
    return {
        "message": message,
        "pipeline_id": "p1",
        "stage_name": "test",
        "job_name": "unit_tests",
        "task_name": "test_login",
        "branch": "main",
        "user": "alice"
    }

def test_pipeline_shapes():
    sample_log = pd.DataFrame([make_log()])
    text_features = encode_text(sample_log['message'].tolist())
    meta_features, _ = encode_metadata(sample_log)
    X = combine_features(text_features, meta_features)

    assert X.shape[0] == 1, "Should have 1 row"
    assert isinstance(X, np.ndarray), "X must be a numpy array"

    X_tensor = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        logits = model(X_tensor)
    assert logits.shape[0] == 1, "Output batch size should be 1"

@pytest.mark.parametrize("log_dict", [
    {"message": "Build failed: missing dependency", "pipeline_id":"p2", "stage_name":"build",
     "job_name":"compile","task_name":"compile_all","branch":"dev","user":"bob"},
    {"message": "Deployment failed: timeout error", "pipeline_id":"p1", "stage_name":"deploy",
     "job_name":"deploy_prod","task_name":"deploy_service","branch":"main","user":"alice"},
])
def test_sample_predictions(log_dict):
    log_input = LogInput(**log_dict)
    result = predict_failure(log_input)
    assert "prediction" in result, "Prediction key missing"
    assert isinstance(result["prediction"], int), "Prediction must be an integer"
    assert "suggestion" in result, "Suggestion key missing"

def test_prediction_confidence():
    log_input = LogInput(**make_log())
    result = predict_failure(log_input)
    assert "prediction" in result
    assert "suggestion" in result

def test_empty_unknown_inputs():
    log_dict = {
        "message": "",
        "pipeline_id": "unknown_pipeline",
        "stage_name": "unknown_stage",
        "job_name": "unknown_job",
        "task_name": "unknown_task",
        "branch": "unknown_branch",
        "user": "unknown_user"
    }
    log_input = LogInput(**log_dict)
    result = predict_failure(log_input)
    assert "prediction" in result, "Prediction should exist even for unknown inputs"

def test_regression_example():
    log_input = LogInput(**make_log())
    result = predict_failure(log_input)
    assert isinstance(result["prediction"], int)
    assert "suggestion" in result
