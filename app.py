import streamlit as st
import joblib
import numpy as np

model = joblib.load("insurance_model.pkl")

st.title("Insurance Cost Prediction")

age = st.number_input("Age")
bmi = st.number_input("BMI")
children = st.number_input("Children")
smoker = st.selectbox("Smoker",["Yes","No"])

if smoker == "Yes":
    smoker = 1
else:
    smoker = 0

smoker_bmi = smoker * bmi

if st.button("Predict Cost"):
    
    data = np.array([[age,bmi,children,smoker_bmi]])
    
    prediction = model.predict(data)
    
    st.success(f"Estimated Insurance Cost: {prediction[0]:,.2f}")