from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np

# Import the engines (Make sure files are named sklearn_engine.py and pytorch_engine.py)
from app.wdr_engine import get_diet_workout_prediction
from app.bfp_engine import get_bodyfat_prediction

app = FastAPI(title='Naksihat AI API', version='2.0')

# --- DATA MODELS ---

# Input for Diet/Workout (Matches your sklearn logic)
class DietWorkoutInput(BaseModel):
    age: int
    gender: str
    height: float
    weight: float
    activeness: str
    goal: str

# Input for Body Fat (Matches your PyTorch logic)
class BodyFatInput(BaseModel):
    sex: str
    age: float
    weight_kg: float
    height_cm: float
    neck_cm: float
    chest_cm: float
    abdomen_cm: float
    hip_cm: float
    thigh_cm: float
    knee_cm: float
    ankle_cm: float
    biceps_cm: float
    forearm_cm: float
    wrist_cm: float

@app.get("/")
def home():
    return {"message": "Naksihat AI API is Running with Dual Engines 🚀"}

# --- ROUTE 1: DIET & WORKOUT ---
@app.post("/predict-plan")
def predict_regimen(data: DietWorkoutInput):
    try:
        # Convert Pydantic model to dictionary for the engine
        data_dict = data.dict()
        
        # Call the sklearn engine
        result = get_diet_workout_prediction(data_dict)
        
        return result
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- ROUTE 2: BODY FAT PERCENTAGE ---
@app.post("/predict-bfp")
def predict_bfp(stats: BodyFatInput):
    try:
        # Preprocess Inputs for PyTorch
        sex_val = 0 if stats.sex.lower() == "male" else 1
        height_m = stats.height_cm / 100
        bmi = stats.weight_kg / (height_m ** 2)

        # Create Numpy Array (Strict Order: Sex, Age, Weight, Height, Neck... BMI)
        # MUST MATCH your training columns exactly
        features = np.array([[
            sex_val, stats.age, stats.weight_kg, stats.height_cm,
            stats.neck_cm, stats.chest_cm, stats.abdomen_cm, stats.hip_cm,
            stats.thigh_cm, stats.knee_cm, stats.ankle_cm, stats.biceps_cm,
            stats.forearm_cm, stats.wrist_cm, bmi
        ]])

        # Call the PyTorch engine
        result = get_bodyfat_prediction(features)
        
        return {
            "body_fat_percentage": round(result, 2),
            "bmi": round(bmi, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))