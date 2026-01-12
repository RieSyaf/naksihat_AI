#CODE FOR BODY FAT PERCENTAGE MODEL

import os
import torch
import torch.nn as nn
import joblib
import numpy as np

# --- CONFIG ---
base_path = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(base_path, 'models') #add models to base path dir


MODEL_PATH = os.path.join(models_dir, "naksihat_bf%_model.pth")
SCALER_PATH = os.path.join(models_dir, "scaler.pkl")

# --- DEFINE MODEL CLASS ---
class BodyFatModel(nn.Module):
    def __init__(self, input_count):
        super().__init__()
        self.fc1 = nn.Linear(input_count, 128)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 64)
        self.act2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.1)
        self.out = nn.Linear(64, 1)

    def forward(self, x):
        x = self.drop1(self.act1(self.fc1(x)))
        x = self.drop2(self.act2(self.fc2(x)))
        return self.out(x)

# --- LOAD RESOURCES ---
model = None
scaler = None

try:
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
        model = BodyFatModel(input_count=15)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        model.eval()
        print("✅ PyTorch Engine: Online")
    else:
        print(f"⚠️ PyTorch Engine: Missing files at {MODEL_PATH}")
except Exception as e:
    print(f"⚠️ PyTorch Engine Error: {e}")

# --- PREDICTION FUNCTION ---
def get_bodyfat_prediction(features_array):
    """
    Expects a numpy array of shape (1, 15) with the exact column order.
    Returns: body_fat_percentage (float)
    """
    if model is None or scaler is None:
        raise Exception("Body Fat Model is not loaded.")
    
    with torch.no_grad():
        features_scaled = scaler.transform(features_array)
        inputs = torch.FloatTensor(features_scaled)
        prediction = model(inputs)
    
    return prediction.item()