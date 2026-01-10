import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

# Ensure the output directory exists
os.makedirs('app/models', exist_ok=True)

np.random.seed(5)
# num of samples
ns = 1500

# fabricated data to train the model
data = {
    
    'age' : np.random.randint(18, 65, ns), #randomly choose age between 18-65
    'gender' : np.random.choice(['Male', 'Female'], ns), #randomly choose gender
    'height' : np.random.normal(170, 10, ns).round(0),      #in cm
    'weight' : np.random.normal(75, 15, ns).round(0),      #in kg
    'activeness' : np.random.choice(['Sedentary', 'Lightly Active', 'Moderate Active', 'Very Active'], ns), #randomly choose activity level
    'goal' : np.random.choice(['Weight Loss', 'Muscle Gain', 'Maintenance'], ns) #randomly choose goal

}

df = pd.DataFrame(data)

df['bmi'] = (df['weight'] / ((df['height'] / 100)**2)).round(1)#calculate bmi

def workout(row):
    if row['goal'] == 'Muscle Gain': return 'Weight Training'
    elif row['goal'] == 'Weight Loss' and row['bmi'] > 30: return 'Light Cardio'
    elif row['goal'] == 'Weight Loss': return 'Moderate to High Cardio'
    else: return 'Yoga'

def diet(row):
    if row['goal'] == 'Muscle Gain': return 'High Protein'
    elif row['goal'] == 'Weight Loss': return 'Caloric Deficit'
    else: return 'Balanced Macros'

# apply the workout and diet func to the df
df['workout_plan'] = df.apply(workout, axis=1)
df['diet_plan'] = df.apply(diet, axis=1)

print(f'Dataset Generated : {df.shape}')

# copy df to another df where we can modify freely
df_train = df.copy()

# label encoding eg turns male and female to 0 and 1
le_gender = LabelEncoder()
le_goal = LabelEncoder()
le_activeness = LabelEncoder()
le_diet_plan = LabelEncoder()
le_workout_plan = LabelEncoder()

df_train['gender'] = le_gender.fit_transform(df['gender'])
df_train['goal'] = le_goal.fit_transform(df['goal'])
df_train['activeness'] = le_activeness.fit_transform(df['activeness'])
df_train['diet_plan'] = le_diet_plan.fit_transform(df['diet_plan'])
df_train['workout_plan'] = le_workout_plan.fit_transform(df['workout_plan'])

print('Data encoded success')

# variable X for features/inputs
X = df_train[['age', 'gender', 'height', 'weight', 'activeness', 'goal', 'bmi']]

# variable y for targets/outputs for workout and diet plan
y_workout = df_train['workout_plan']
y_diet = df_train['diet_plan']

# split the dataset : use y_work_train and y_diet_train as the objective for X_train for the
# train part and then use y_work_test and y_diet_test as objective for X_test
X_train, X_test, y_workout_train, y_workout_test, y_diet_train, y_diet_test = train_test_split(
    X, y_workout, y_diet, test_size=0.2, random_state=5
)

# Training the workout and diet model
print('Training Workout Model...')
model_workout = RandomForestClassifier(n_estimators=100, random_state=5)
model_workout.fit(X_train, y_workout_train)

print('Training Diet Model...')
model_diet = RandomForestClassifier(n_estimators=100, random_state=5)
model_diet.fit(X_train, y_diet_train)

# Evaluation
acc_workout = accuracy_score(y_workout_test, model_workout.predict(X_test))
acc_diet = accuracy_score(y_diet_test, model_diet.predict(X_test))

print('\nModel Results')
print(f"Workout Model Accuracy: {acc_workout*100:.2f}%")
print(f"Diet Model Accuracy: {acc_diet*100:.2f}%")

# --- SAVING ARTIFACTS ---
print("\nSaving models and encoders...")

# Save the models
joblib.dump(model_workout, 'app/models/workout_model.pkl')
joblib.dump(model_diet, 'app/models/diet_model.pkl')

# Save the encoders (We bundle them into a dictionary for easier loading)
encoders = {
    'gender': le_gender,
    'goal': le_goal,
    'activeness': le_activeness,
    'workout_plan': le_workout_plan,
    'diet_plan': le_diet_plan
}
joblib.dump(encoders, 'app/models/encoders.pkl')

print("✅ All files saved to app/models/")