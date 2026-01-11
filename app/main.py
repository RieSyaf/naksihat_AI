from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os

#initialize app, create the server
app = FastAPI(title = 'Naksihat AI API', version = '1.0')


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


# for valid user input
class UserInput(BaseModel):
    age: int
    gender : str
    height : float
    weight : float
    activeness : str
    goal : str

# the route for prediction
@app.post('/predict')
def predict_regimen(data: UserInput):
    try:
        bmi = round(data.weight / ((data.height / 100) ** 2), 1)

        #since that to use the model, we passed dataframe structure, then we need to structure the input to a dataframe
        # with the same exact order of column that we use for training
        input_data = pd.DataFrame([{ #we use this combinantion of brackets because we need a single row ([]) with dictionary ({})
            
            'age' : data.age,
            'gender': data.gender,
            'height': data.height,
            'weight': data.weight,
            'activeness': data.activeness,
            'goal': data.goal,
            'bmi': bmi

        }])

        # now we encode(change) the label(value) for gender,activeness and goal
        for col in ['gender', 'activeness', 'goal']:
            encoder = encoders[col]

            #check if the value user give exist in the encorder or not, if not raise error
            if input_data[col].iloc[0] not in encoder.classes_ :
                raise HTTPException(status_code = 400, detail = f'Invalid value {input_data[col].iloc[0]} for {col}')
            input_data[col] = encoder.transform(input_data[col])

        #make predictions:
        workout_pred = model_workout.predict(input_data)
        diet_pred  = model_diet.predict(input_data)

        #decode (change back) the result of workout and diet model which is numbers to text
        workout_res = encoders['workout_plan'].inverse_transform(workout_pred)[0]
        diet_res = encoders['diet_plan'].inverse_transform(diet_pred)[0]

        #return the result in json array
        return {
            "bmi": bmi,
            "workout_plan": workout_res,
            "diet_plan": diet_res
        }


    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
#route for home
@app.get('/')
def home():
    return {"message": "naksihat AI is running! Go to /docs to test the API."}


