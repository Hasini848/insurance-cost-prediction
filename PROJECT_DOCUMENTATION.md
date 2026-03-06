1. Project Title:Insurance Cost Prediction using Linear Regression
2. Problem Statement:Insurance companies need to estimate medical insurance costs for customers based on their personal and health information.The goal of this project is to build a Machine Learning model that predicts insurance charges using Linear Regression.
3. Objective
• Build a regression model to predict insurance cost
• Understand relationship between features and target variable
• Perform feature engineering and preprocessing
• Deploy the model using Streamlit
4. Dataset Information
Dataset used: Insurance Dataset
Source:https://www.kaggle.com/dataset/mirichoi0218/insurance
Dataset contains the following features:
Feature	           Description
age	               Age of customer
bmi	               Body Mass Index
children	       Number of children
smoker	           Whether customer smokes
charges	           Insurance cost (target variable)
5. Tools & Technologies
• Python
• Pandas
• NumPy
• Scikit-learn
• Streamlit
• GitHub
• VS Code
6. Data Preprocessing
Steps performed:
Loaded dataset using pandas
created BMI categories
Created age group categories
Created interaction feature smoker_bmi
Removed outliers using IQR method
Selected important features
Feature engineering performed:
smoker_bmi = smoker * bmi
Selected features:
age
bmi
children
smoker_bmi
Target variable:charges
7. Model Building
Algorithm used:Linear Regression
Example code:
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)
8. Model Evaluation
Metrics used:
• Mean Absolute Error (MAE)
• Root Mean Squared Error (RMSE)
• R² Score
Example:
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
9. Model Saving
Model saved using joblib (.pkl) format.
import joblib
joblib.dump(model,"insurance_model.pkl")
10. Deployment
The model was deployed using Streamlit.
Example:
import streamlit as st
import joblib
model = joblib.load("insurance_model.pkl")
Users input values such as age, BMI and children to predict insurance cost.
11. Results
The model successfully predicts insurance charges based on customer details.
Example input:
Age = 30
BMI = 25
Children = 2
Smoker = No
Predicted insurance cost:
Estimated Cost ≈ $8000
12. GitHub Repository
Project available at:https://github.com/Hasini848/insurance-cost-prediction
13. Streamlit Deployment
Live application:https://insurance-cost-prediction-kcxnvnbwtkbxs3phlcsg4d.streamlit.app/
14. Conclusion:This project demonstrates how Linear Regression can be used to predict medical insurance costs using customer information. The model was successfully trained, evaluated and deployed as a web application using Streamlit.
