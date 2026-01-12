import streamlit as st
import requests
import os

# --- CONFIGURATION ---
st.set_page_config(page_title="💪🏼 NAKSIHAT AI", layout="wide")

# API URL (Connects to your FastAPI Backend)
API_URL = os.getenv("API_URL", "https://naksihat-api-371790036126.asia-southeast1.run.app")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    div.stButton > button:first-child {
        height: 3em;
        width: 100%;
        font-weight: bold;
        font-size: 18px;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
    }
    .main-header {
        font-size: 2.5rem;
        color: #4CAF50;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

def custom_info_box(text, font_size="24px", align="center"):
    st.markdown(
        f"""
        <div style="
            background-color: #111184; 
            padding: 0px; 
            border-radius: 10px; 
            border-left: 0px;
            text-align: {align};
            padding: 5px;
            margin-bottom: 15px;
        ">
            <span style="font-size: {font_size}; font-weight: bold; color: #83EEFF; font-family: sans-serif;  font-weight: 900; letter-spacing: 3px;">
                {text}
            </span>
        </div>
        """, 
        unsafe_allow_html=True
    )

def custom_info_box2(text, font_size="24px", align="center"):
    st.markdown(
        f"""
        <div style="
            background-color: #06402B; 
            padding: 0px; 
            border-radius: 10px; 
            border-left: 0px;
            text-align: {align};
            padding: 5px;
            margin-bottom: 15px;
        ">
            <span style="font-size: {font_size}; font-weight: bold; color: #0FFF50; font-family: sans-serif;  font-weight: 900; letter-spacing: 3px;">
                {text}
            </span>
        </div>
        """, 
        unsafe_allow_html=True
    )

# --- NAVIGATION STATE ---
if 'page' not in st.session_state:
    st.session_state.page = 'home'

def go_home():
    st.session_state.page = 'home'

def go_bfp():
    st.session_state.page = 'bfp'

def go_plan():
    st.session_state.page = 'plan'

# ==========================
# 🏠 HOME PAGE
# ==========================
if st.session_state.page == 'home':
    st.markdown("<h1 class='main-header'>🔥 NAKSIHAT-AI 🔥</h1>", unsafe_allow_html=True)
    st.markdown("### Choose your health journey:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        custom_info_box("🏋️‍♂️ Get Fit", font_size="32px", align="center")
        st.markdown("Get personalized workout routines and meal plans tailored to your goal.")
        st.button("Generate Diet & Workout Plan", on_click=go_plan)
        
    with col2:
        custom_info_box2("🧬 Analyze Body", font_size="32px", align="center")
        st.markdown("Calculate medical-grade Body Fat % using our PyTorch AI scan.")
        st.button("Calculate Body Fat %", on_click=go_bfp)

# ==========================
# 🧬 BODY FAT PAGE (INTEGRATED)
# ==========================
elif st.session_state.page == 'bfp':
    st.button("← Back to Home", on_click=go_home)
    st.title("🧬 Body Fat Percentage Analyzer")
    st.markdown("Enter your measurements below. The AI uses these to predict your body fat percentage.")

    with st.form("bfp_form"):
        
        # --- SECTION 1: GENERAL INFO ---
        st.subheader("General Information")
        col1, col2 = st.columns(2)

        with col1:
            sex = st.selectbox("Sex", ["Male", "Female"]) # Changed to full words for backend compatibility
            age = st.number_input("Age", 1.0, 120.0, 25.0)

        with col2:
            weight_kg = st.number_input("Weight (kg)", 10.0, 300.0, 70.0)
            height_cm = st.number_input("Height (cm)", 50.0, 250.0, 175.0)

        # --- SECTION 2: BODY MEASUREMENTS ---
        st.subheader("Measurements (cm)")
        m_col1, m_col2, m_col3 = st.columns(3)

        with m_col1:
            neck_cm = st.number_input("Neck", 20.0, 60.0, 38.0)
            chest_cm = st.number_input("Chest", 50.0, 150.0, 100.0)
            abdomen_cm = st.number_input("Abdomen (Waist)", 50.0, 150.0, 85.0)
            hip_cm = st.number_input("Hip", 50.0, 150.0, 95.0)

        with m_col2:
            thigh_cm = st.number_input("Thigh", 20.0, 100.0, 55.0)
            knee_cm = st.number_input("Knee", 10.0, 60.0, 38.0)
            ankle_cm = st.number_input("Ankle", 10.0, 40.0, 22.0)

        with m_col3:
            biceps_cm = st.number_input("Biceps", 10.0, 70.0, 33.0)
            forearm_cm = st.number_input("Forearm", 10.0, 50.0, 28.0)
            wrist_cm = st.number_input("Wrist", 10.0, 30.0, 17.0)

        # Submit Button
        submitted = st.form_submit_button("Analyze Body Composition")

    # --- HANDLE SUBMISSION ---
    if submitted:
        # Prepare Payload (Maps your variables to backend requirements)
        payload = {
            "sex": sex,
            "age": age,
            "weight_kg": weight_kg,
            "height_cm": height_cm,
            "neck_cm": neck_cm,
            "chest_cm": chest_cm,
            "abdomen_cm": abdomen_cm,
            "hip_cm": hip_cm,
            "thigh_cm": thigh_cm,
            "knee_cm": knee_cm,
            "ankle_cm": ankle_cm,
            "biceps_cm": biceps_cm,
            "forearm_cm": forearm_cm,
            "wrist_cm": wrist_cm
        }

        try:
            with st.spinner("Connecting to PyTorch Neural Network..."):
                res = requests.post(f"{API_URL}/predict-bfp", json=payload)
            
            if res.status_code == 200:
                data = res.json()
                bfp = data['body_fat_percentage']
                bmi = data['bmi']

                st.success("Analysis Complete! ✅")
                
                # Display Results
                r1, r2 = st.columns(2)
                r1.metric("Body Fat Percentage", f"{bfp}%")
                r2.metric("BMI Score", f"{bmi}")
                
                # Visual Bar
                st.progress(min(bfp/50, 1.0))
                
                # Health Context
                if bfp < 6: 
                    st.error("⚠️ **Essential Fat Only** -> Your body fat is at a critical level; consult a professional to ensure hormonal health.")

                elif bfp < 14: 
                    st.info("🏃 **Athlete / Lean** -> Peak performance range: You are highly lean with excellent muscle definition.")

                elif bfp < 25: 
                    st.success("✅ **Fitness / Average** -> Healthy and sustainable: You maintain a balanced body composition for an active lifestyle.")

                else: 
                    st.warning("⚠️ **Obesity**  ->  Elevated health risk: Focus on a consistent caloric deficit and strength training.")

            else:
                st.error(f"Server Error: {res.text}")
        except Exception as e:
            st.error(f"Connection Failed. Is the backend running? \nError: {e}")

# ==========================
# 🏋️‍♂️ DIET & WORKOUT PAGE
# ==========================
elif st.session_state.page == 'plan':
    st.button("← Back to Home", on_click=go_home)
    st.title("🏋️‍♂️ Workout & Diet Planner")
    
    with st.form("plan_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", 10, 100, 25)
            gender = st.selectbox("Gender", ["Male", "Female"])
            height = st.number_input("Height (cm)", 100.0, 250.0, 175.0)
        with col2:
            weight = st.number_input("Weight (kg)", 30.0, 200.0, 70.0)
            activeness = st.selectbox("Activity Level", ["Sedentary", "Lightly Active", "Moderately Active", "Very Active"])
            goal = st.selectbox("Goal", ["Weight Loss", "Muscle Gain", "Maintenance"])
        
        if st.form_submit_button("Generate Plan"):
            payload = {
                "age": age, "gender": gender, "height": height,
                "weight": weight, "activeness": activeness, "goal": goal
            }
            try:
                with st.spinner("Generating regimen..."):
                    res = requests.post(f"{API_URL}/predict-plan", json=payload)
                
                if res.status_code == 200:
                    data = res.json()
                    st.divider()
                    st.subheader("📋 Your Custom Plan")
                    
                    custom_info_box(f"Diet Recommendation: {data['diet_plan']}")
                    custom_info_box2(f"Workout Recommendation: {data['workout_plan']}")
                    st.caption(f"Based on BMI: {data['bmi']}")
                else:
                    st.error(f"Error: {res.text}")
            except Exception as e:
                st.error(f"Connection failed: {e}")