import streamlit as st
import sklearn.compose._column_transformer
class _RemainderColsList(list):
    pass
if not hasattr(sklearn.compose._column_transformer,'_RemainderColsList'):    
    sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList
import joblib
import pandas as pd

model = joblib.load('weather_final_pipeline.pkl')

st.title("Will it Rain Tomorrow?")
st.write("Bas basic details daalo")

# --- Sirf basic inputs lo user se ---
Location = st.selectbox("Location", ["Sydney", "Melbourne", "Brisbane", "Perth"])
MinTemp = st.slider("MinTemp", 0.0, 30.0, 15.0)
MaxTemp = st.slider("MaxTemp", 10.0, 40.0, 22.0)
Rainfall = st.number_input("Aaj kitni baarish hui (mm)", 0.0, 50.0, 0.0)
Humidity9am = st.slider("Humidity Subah", 0, 100, 70)
RainToday = st.selectbox("Aaj Baarish Hui?", ["No", "Yes"])

# --- Baaki ke columns ka default value ---
# Ye values tere training data ke median / mode se aayengi
default_values = {
    'Evaporation': 5.2,
    'Sunshine': 8.5,
    'WindGustDir': 'N',
    'WindGustSpeed': 40.0,
    'WindDir9am': 'NW',
    'WindDir3pm': 'SE',
    'WindSpeed9am': 10.0,
    'WindSpeed3pm': 15.0,
    'Humidity3pm': 60,
    'Pressure9am': 1015.0,
    'Pressure3pm': 1012.0,
    'Cloud9am': 3,
    'Cloud3pm': 4,
    'Temp9am': 18.0,
    'Temp3pm': 21.0
}

if st.button("Predict Karo"):
    # User wala data + default data mix karke pura row banao
    input_data = {
        'Location': Location,
        'MinTemp': MinTemp,
        'MaxTemp': MaxTemp,
        'Rainfall': Rainfall,
        'Humidity9am': Humidity9am,
        'RainToday': RainToday,
        **default_values
    }

    df_input = pd.DataFrame([input_data])
    pred = model.predict(df_input)[0]
    proba = model.predict_proba(df_input)[0][1]

    if pred == 1:
        st.error(f"Kal Baarish Hogi! {proba*100:.0f}% chance")
    else:
        st.success(f"Kal Baarish Nahi Hogi! {proba*100:.0f}% chance")
