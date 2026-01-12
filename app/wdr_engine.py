#CODE FOR WORKOUT AND DIET RECOMMENDATION MODEL

from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os




# this is like making the path naming convention consistent and enable to run this server in any terminal iic
base_path = os.path.dirname(os.path.abspath(__file__)) #store this file directory into base_path
models_dir = os.path.join(base_path, 'models') #add models to base path dir

#load the saved models and encoders
try :
    model_workout = joblib.load(os.path.join(models_dir, 'workout_model.pkl'))
    model_diet = joblib.load(os.path.join(models_dir, 'diet_model.pkl'))
    encoders = joblib.load(os.path.join(models_dir, 'encoders.pkl'))
except Exception as e:
    print(f"❌ Error loading models: {e}")




def get_diet_workout_prediction(data_dict):
    """
    Takes a dictionary of user inputs, processes them, and returns predictions.
    Input format: {'age': 25, 'gender': 'Male', ...}
    """
    if not model_workout or not model_diet or not encoders:
        raise Exception("Sklearn models are not loaded.")

    try:
        # 1. Calculate BMI
        # Ensure height is in cm as per your logic
        bmi = round(data_dict['weight'] / ((data_dict['height'] / 100) ** 2), 1)

        # 2. Create DataFrame (Exact order from training)
        input_data = pd.DataFrame([{
            'age': data_dict['age'],
            'gender': data_dict['gender'],
            'height': data_dict['height'],
            'weight': data_dict['weight'],
            'activeness': data_dict['activeness'],
            'goal': data_dict['goal'],
            'bmi': bmi
        }])

        # 3. Encode Inputs
        for col in ['gender', 'activeness', 'goal']:
            encoder = encoders[col]
            # Check validity
            val = input_data[col].iloc[0]
            if val not in encoder.classes_:
                raise ValueError(f"Invalid value '{val}' for {col}. Expected: {list(encoder.classes_)}")
            
            input_data[col] = encoder.transform(input_data[col])

        # 4. Predict
        workout_pred = model_workout.predict(input_data)
        diet_pred = model_diet.predict(input_data)

        # 5. Decode Results (Numbers -> Text)
        workout_res = encoders['workout_plan'].inverse_transform(workout_pred)[0]
        diet_res = encoders['diet_plan'].inverse_transform(diet_pred)[0]

        return {
            "bmi": bmi,
            "workout_plan": workout_res,
            "diet_plan": diet_res
        }

    except Exception as e:
        # Re-raise the error so main.py can catch it
        raise e



