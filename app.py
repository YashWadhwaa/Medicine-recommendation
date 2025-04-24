import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model
model_path = "svc.pkl"  # Ensure this file is uploaded to your repo
with open(model_path, "rb") as f:
    rf_model = pickle.load(f)

# Load supporting datasets
df = pd.read_csv('Training.csv.zip')
sym_des = pd.read_csv('Symptom-severity.csv')
precautions = pd.read_csv('precautions_df.csv')
workout = pd.read_csv('workout_df.csv')
description = pd.read_csv('description.csv')
medications = pd.read_csv('medications.csv')
diets = pd.read_csv('diets.csv')

# Prepare training data
X = df.drop('prognosis', axis=1)
y = df['prognosis']
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Dictionary for reverse lookup
inv_label_dict = {v: k for k, v in enumerate(label_encoder.classes_)}

# Prepare disease data dictionary
disease_data = {}
for i in range(len(medications)):
    disease = medications["Disease"][i]
    disease_data[disease] = [
        medications["Medication"][i],
        description["Description"][i],
        precautions["Precaution_1"][i],
        precautions["Precaution_2"][i],
        precautions["Precaution_3"][i],
        precautions["Precaution_4"][i],
        workout["workout"][i],
        diets["Diet"][i]
    ]

# Disease prediction function
def predict_disease(symptoms):
    symptom_feed = {col: 1 if col in symptoms else 0 for col in X.columns}
    feed_df = pd.DataFrame([symptom_feed])
    pred_encoded = rf_model.predict(feed_df)[0]
    disease_name = label_encoder.inverse_transform([pred_encoded])[0]
    return disease_name

# Recommendation function
def get_recommendations(disease):
    if disease in disease_data:
        data = disease_data[disease]
        return {
            "Medication": data[0],
            "Description": data[1],
            "Precautions": data[2:6],
            "Workout": data[6],
            "Diet": data[7]
        }
    else:
        return None

# Streamlit UI
st.set_page_config(page_title="Disease Predictor", layout="centered")
st.title("🩺 Disease Prediction and Recommendation App")
st.markdown("### Select your symptoms")

# Symptom selection
symptoms = st.multiselect("Choose from the list of symptoms", options=X.columns.tolist())

if st.button("Predict Disease"):
    if not symptoms:
        st.warning("Please select at least one symptom.")
    else:
        disease = predict_disease(symptoms)
        st.success(f"Predicted Disease: *{disease}*")
        rec = get_recommendations(disease)

        if rec:
            st.markdown("### 📝 Description")
            st.info(rec["Description"])

            st.markdown("### 💊 Medications")
            st.write(rec["Medication"])

            st.markdown("### ⚠ Precautions")
            for i, prec in enumerate(rec["Precautions"], start=1):
                st.write(f"{i}. {prec}")

            st.markdown("### 🏋 Workout Suggestion")
            st.write(rec["Workout"])

            st.markdown("### 🥗 Diet Plan")
            st.write(rec["Diet"])
        else:
            st.error("No recommendations found for the predicted disease.")