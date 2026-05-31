import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv(r"C:\Users\Vansh\OneDrive\Documents\Html\Python\react.js\my-app.js\Diabetes prediction\diabetes_prediction_dataset - Copy.csv")

# Encode categorical columns properly
label_encoders = {}
for col in ['gender', 'smoking_history']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Split features and target
X = df.drop("diabetes", axis=1)
y = df["diabetes"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("diabetes_model.pkl", "wb"))

# Streamlit UI
st.title("Diabetes Prediction System")
st.write("Enter patient details below:")

# Input fields
gender = st.selectbox("Gender", label_encoders['gender'].classes_)
age = st.number_input("Age", min_value=1, max_value=120, step=1)
hypertension = st.selectbox("Hypertension", [0, 1])
heart_disease = st.selectbox("Heart Disease", [0, 1])
smoking_history = st.selectbox("Smoking History", label_encoders['smoking_history'].classes_)
bmi = st.number_input("BMI", min_value=10.0, max_value=70.0, step=0.1)
hba1c = st.number_input("HbA1c Level", min_value=3.0, max_value=15.0, step=0.1)
blood_glucose = st.number_input("Blood Glucose Level", min_value=50, max_value=400, step=1)

# Convert categorical inputs using same encoders
gender_encoded = label_encoders['gender'].transform([gender])[0]
smoking_encoded = label_encoders['smoking_history'].transform([smoking_history])[0]

# Prediction
if st.button("Predict"):
    input_data = np.array([[gender_encoded, age, hypertension, heart_disease,
                            smoking_encoded, bmi, hba1c, blood_glucose]])
    prediction = model.predict(input_data)
    result = "Diabetic" if prediction[0] == 1 else "Non-Diabetic"
    st.success(f"Prediction: {result}")
