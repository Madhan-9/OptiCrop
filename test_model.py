import os
import joblib
import pandas as pd

# Resolve paths relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "crop_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "model", "label_encoder.pkl")

# Load model
# Debug info
print("MODEL_PATH:", MODEL_PATH)
print("ENCODER_PATH:", ENCODER_PATH)
print("MODEL exists:", os.path.exists(MODEL_PATH))
print("ENCODER exists:", os.path.exists(ENCODER_PATH))

try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully")
except Exception as e:
    print("Failed to load model:", e)
    raise

# Load label encoder
try:
    encoder = joblib.load(ENCODER_PATH)
    print("Encoder loaded successfully")
except Exception as e:
    print("Failed to load encoder:", e)
    raise

# Sample input
sample = pd.DataFrame([{
    "N": 90,
    "P": 42,
    "K": 43,
    "temperature": 20.87,
    "humidity": 82.00,
    "ph": 6.50,
    "rainfall": 202.93
}])

# Predict
prediction = model.predict(sample)

# Convert numeric prediction back to crop name
crop = encoder.inverse_transform(prediction)

print("Recommended Crop:", crop[0])