import streamlit as st
import pandas as pd
import joblib

# Load model and encoder
model = joblib.load("salary_model.pkl")
le = joblib.load("label_encoder.pkl")

st.title("💼 Employee Salary Predictor")

# Input fields
experience = st.slider("Years of Experience", 0, 40, 5)
education = st.selectbox("Education Level", ["Bachelor", "Master", "PhD"])
job = st.selectbox("Job Title", ["Data Scientist", "Software Engineer", "Manager", "Analyst"])
location = st.selectbox("Location", ["Chennai", "Bangalore", "Mumbai", "Hyderabad"])

# Convert input to numbers
edu_num = le.transform([education])[0]
job_num = le.transform([job])[0]
loc_num = le.transform([location])[0]

input_data = pd.DataFrame([[experience, edu_num, job_num, loc_num]], columns=['Experience', 'Education_Level', 'Job_Title', 'Location'])

if st.button("Predict Salary"):
    result = model.predict(input_data)[0]
    st.success(f"💰 Predicted Salary: ₹{int(result):,}")
