import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
import pickle

warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('Training.csv')
sym_des = pd.read_csv('Symptom-severity.csv')
precautions = pd.read_csv('precautions_df.csv')
workout = pd.read_csv('workout_df.csv')
description = pd.read_csv('description.csv')
medications = pd.read_csv('medications.csv')
diets = pd.read_csv('diets.csv')

# Prepare data
X = df.drop('prognosis', axis=1)
y = df['prognosis']
label_encoder = LabelEncoder()
label_encoder.fit(y)
y = label_encoder.transform(y)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=5, test_size=0.2, train_size=0.8)

# Train all models
models = {
    'SVC': SVC(kernel='linear'),
    'Random Forest': RandomForestClassifier(random_state=5, n_estimators=100),
    'KNeighbors': KNeighborsClassifier(n_neighbors=5),
    'Gradient Boosting': GradientBoostingClassifier(random_state=5, n_estimators=100),
    'MultinomialNB': MultinomialNB()
}

st.title("Disease Prediction System")
st.write("Training and evaluating all models...")

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    accuracy = accuracy_score(y_valid, preds)
    cm = confusion_matrix(y_valid, preds)
    st.subheader(f"{name} Accuracy: {accuracy:.4f}")
    st.text(f"Confusion Matrix:\n{cm}")
    st.markdown("---")

# Final model for prediction
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train, y_train)

# Save and load model
with open("svc.pkl", "wb") as f:
    pickle.dump(rf_model, f)

with open("svc.pkl", "rb") as f:
    rf_model = pickle.load(f)

# Disease mapping
dis_ld = medications["Disease"]
dis_lm = medications["Medication"]
dis_ldesc = description["Description"].to_list()
dis_lprec1 = precautions["Precaution_1"]
dis_lprec2 = precautions["Precaution_2"]
dis_lprec3 = precautions["Precaution_3"]
dis_lprec4 = precautions["Precaution_4"]
dis_lworkout = workout["workout"]
dis_ldiets = diets["Diet"]

disease_data = {}
for i in range(len(dis_ld)):
    disease_data[dis_ld[i]] = [dis_lm[i], dis_ldesc[i], dis_lprec1[i], dis_lprec2[i], dis_lprec3[i], dis_lprec4[i], dis_lworkout[i], dis_ldiets[i]]

yy = label_encoder.inverse_transform(y_valid)
disease_to_encoded = {}
for x in range(len(yy)):
    disease_to_encoded[yy[x]] = y_valid[x]

# Prediction + recommendation
def recomm(disease):
    if disease in disease_data:
        return disease_data[disease]
    return ["N/A"] * 8

def recomm_s(symptom_lst):
    symptom_feed = {b: 1 if b in symptom_lst else 0 for b in X_valid.columns}
    feed_df = pd.DataFrame([list(symptom_feed.values())], columns=X_valid.columns)
    prediction = rf_model.predict(feed_df)
    for key in disease_to_encoded:
        if disease_to_encoded[key] == prediction[0]:
            return key
    return "Unknown Disease"

# Streamlit input section
st.header("Symptom Input")
inputtable = list(X_valid.columns)
selected_symptoms = st.multiselect("Select Symptoms:", inputtable)

if st.button("Predict Disease"):
    if not selected_symptoms:
        st.warning("Please select at least one symptom.")
    else:
        disease = recomm_s(selected_symptoms)
        medlist, description, prec1, prec2, prec3, prec4, workout, diet = recomm(disease)

        st.success(f"Predicted Disease: {disease}")
        st.markdown(f"*Description:* {description}")
        st.markdown(f"*Medicines:* {medlist}")
        st.markdown("*Precautions:*")
        st.markdown(f"- {prec1}")
        st.markdown(f"- {prec2}")
        st.markdown(f"- {prec3}")
        st.markdown(f"- {prec4}")
        st.markdown(f"*Workout Recommendation:* {workout}")
        st.markdown(f"*Diet Recommendation:* {diet}")
