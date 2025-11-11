from fastapi import FastAPI
from pydantic import BaseModel
import torch
import pandas as pd
import numpy as np

from .utils import load_data, get_fix
from .feature_extraction import encode_text, encode_metadata, combine_features
from .model import train_model, load_model, save_encoder, load_encoder

# ------------------------------
# FASTAPI APP
# ------------------------------
app = FastAPI(title="Smart CI/CD Bot", version="1.0")

# ------------------------------
# TRAIN OR LOAD MODEL
# ------------------------------
try:
    model = load_model()
    encoder = load_encoder()
    print("✅ Model and encoder loaded from artifacts")
except Exception as e:
    print(f"⚠️ Artifacts missing, training new model: {e}")
    features, labels = load_data()
    text_features = encode_text(features['message'].tolist())
    meta_features, encoder = encode_metadata(features)
    X = combine_features(text_features, meta_features)
    y = labels.values
    model = train_model(X, y, epochs=5)
    save_encoder(encoder)

# ------------------------------
# INPUT SCHEMA
# ------------------------------
class LogInput(BaseModel):
    message: str
    pipeline_id: str
    stage_name: str
    job_name: str
    task_name: str
    branch: str
    user: str

# ------------------------------
# ENDPOINTS
# ------------------------------
@app.get("/")
def root():
    return {"message": "🚀 Smart CI/CD Bot is running!"}


@app.post("/predict")
def predict_failure(log: LogInput, confidence_threshold: float = 0.7):
    try:
        df = pd.DataFrame([log.model_dump()])  # Pydantic v2-safe

        # Handle skipped/running first
        log_lower = log.message.lower()
        if "skipped" in log_lower:
            suggestion = "Task was skipped due to pipeline conditions."
            return {"prediction": None, "probabilities": None, "suggestion": suggestion}
        elif "running" in log_lower:
            suggestion = "Task is currently in progress."
            return {"prediction": None, "probabilities": None, "suggestion": suggestion}

        # Encode features
        text_features = encode_text(df['message'].tolist())
        meta_features, _ = encode_metadata(df, encoder=encoder)
        X = combine_features(text_features, meta_features)
        X_tensor = torch.tensor(X, dtype=torch.float32)

        # Model prediction
        with torch.no_grad():
            logits = model(X_tensor)
            probs = torch.softmax(logits, dim=1).numpy()[0]
            pred_idx = int(np.argmax(probs))
            top_confidence = float(probs[pred_idx])

        # Generate context-aware suggestion
        suggestion = get_fix(pred_idx, log_message=log.message)

        # Low-confidence warning
        if top_confidence < confidence_threshold:
            suggestion += " ⚠️ Model confidence is low. Check logs carefully."

        return {
            "prediction": pred_idx,
            "probabilities": probs.tolist(),
            "confidence": top_confidence,
            "suggestion": suggestion
        }

    except Exception as e:
        return {"error": str(e)}
