import torch
import pandas as pd
from app.model import load_model, load_encoder
from app.feature_extraction import encode_text, encode_metadata, combine_features
from app.utils import get_fix

# ------------------------------
# LOAD MODEL & ENCODER
# ------------------------------
model = load_model()
encoder = load_encoder()
print("✅ Model and encoder loaded from artifacts")

# ------------------------------
# SANITY CHECK LOGS
# ------------------------------
sample_logs = [
    "Build failed: missing dependency",
    "Deployment succeeded",
    "Test failed: assertion error",
    "Environment setup inconsistent",
    "Version mismatch in build tools",
    ""  # empty log
]

print("\n=== SANITY CHECKS ===")
for log in sample_logs:
    df = pd.DataFrame([{
        'message': log,
        'pipeline_id': 'p1',
        'stage_name': 'build',
        'job_name': 'job1',
        'task_name': 'task1',
        'branch': 'main',
        'user': 'dev'
    }])

    # Encode features
    text_features = encode_text(df['message'].tolist())
    meta_features, _ = encode_metadata(df, encoder=encoder)
    X = combine_features(text_features, meta_features)
    X_tensor = torch.tensor(X, dtype=torch.float32)

    # Predict
    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.softmax(logits, dim=1).numpy()[0]
        pred_idx = int(probs.argmax())

    suggestion = get_fix(pred_idx, log_message=log)

    print(f"Log: '{log}' -> Prediction: {{'prediction': {pred_idx}, 'probabilities': {probs.tolist()}, 'suggestion': '{suggestion}'}}")

# ------------------------------
# OPTIONAL: Evaluate test split
# ------------------------------
from app.utils import load_data

features, labels = load_data()
test_df = features.head(5)
test_labels = labels.head(5).tolist()

print("\n=== TEST SPLIT EVALUATION (first 5 rows) ===")
for i, row in test_df.iterrows():
    log = row['message']
    df = pd.DataFrame([row])
    text_features = encode_text(df['message'].tolist())
    meta_features, _ = encode_metadata(df, encoder=encoder)
    X = combine_features(text_features, meta_features)
    X_tensor = torch.tensor(X, dtype=torch.float32)

    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.softmax(logits, dim=1).numpy()[0]
        pred_idx = int(probs.argmax())

    suggestion = get_fix(pred_idx, log_message=log)
    print(f"Log: '{log}' -> True: {test_labels[i]}, Pred: {pred_idx}, Suggestion: {suggestion}")
