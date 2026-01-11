import streamlit as st
import requests

# 1. Setup the Page
st.set_page_config(page_title="NAKSIHAT-AI", page_icon="💪")
st.title("🔥 NAKSIHAT-AI 🔥", text_alignment='center')
st.write("Enter your details below to get a custom AI-generated regimen.")

# 2. Input Form
with st.form("user_input"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=80, value=25)
        gender = st.selectbox("Gender", ["Male", "Female"])
        height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=175.0)
    
    with col2:
        weight = st.number_input("Weight (kg)", min_value=40.0, max_value=200.0, value=75.0)
        activeness = st.selectbox("Activity Level", ["Sedentary", "Lightly Active", "Moderate Active", "Very Active"])
        goal = st.selectbox("Goal", ["Weight Loss", "Muscle Gain", "Maintenance"])
    
    submitted = st.form_submit_button("Generate Plan")

# 3. Logic: When button is clicked...
if submitted:
    # Prepare the data dictionary (must match FastAPI Input Model exactly)
    api_url = "https://ironpath-api-371790036126.asia-southeast1.run.app/predict"
    payload = {
        "age": age,
        "gender": gender,
        "height": height,
        "weight": weight,
        "activeness": activeness,
        "goal": goal
    }
    
    try:
        # Send data to your running API
        response = requests.post(api_url, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            
            # Display Results
            st.success("✅ Plan Generated Successfully!")
            
            # Metrics
            st.metric(label="Your BMI", value=result['bmi'])
            
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**🏋️ Workout Plan:**\n\n{result['workout_plan']}")
            with col2:
                st.info(f"**🥗 Diet Plan:**\n\n{result['diet_plan']}")
        else:
            st.error(f"Error: {response.text}")
            
    except requests.exceptions.ConnectionError:
        st.error("🚨 Connection Error! Is your FastAPI backend running?")