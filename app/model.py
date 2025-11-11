import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import joblib
import os

# ------------------------------
# MODEL DEFINITION
# ------------------------------
class CICDClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=7):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        return self.network(x)

# ------------------------------
# TRAIN MODEL
# ------------------------------
def train_model(X, y, epochs=10, lr=0.001, hidden_dim=128, output_dim=None, model_path="artifacts/cicd_model.pth"):
    if isinstance(X, np.ndarray):
        X = torch.tensor(X, dtype=torch.float32)
    if isinstance(y, np.ndarray):
        y = torch.tensor(y, dtype=torch.long)

    input_dim = X.shape[1]
    if output_dim is None:
        output_dim = len(set(y.tolist()))

    model = CICDClassifier(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 2 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "output_dim": output_dim
    }, model_path)
    print(f"✅ Model saved at {model_path}")
    return model

# ------------------------------
# LOAD MODEL
# ------------------------------
def load_model(model_path="artifacts/cicd_model.pth"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    checkpoint = torch.load(model_path, map_location="cpu")
    model = CICDClassifier(
        input_dim=checkpoint["input_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        output_dim=checkpoint["output_dim"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

# ------------------------------
# SAVE / LOAD ENCODER
# ------------------------------
def save_encoder(encoder, path="artifacts/encoder.pkl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(encoder, path)
    print(f"✅ Encoder saved at {path}")

def load_encoder(path="artifacts/encoder.pkl"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Encoder file not found: {path}")
    return joblib.load(path)
