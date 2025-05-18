import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from mlxtend.regressor import StackingRegressor

MODEL_PATH = "rainfall_model.pkl"
GOOGLE_DRIVE_MODEL_ID = "15NOorz6TZk-rZr9djbg_1bNRT6ZE2GaO"
GOOGLE_DRIVE_URL = f"https://drive.google.com/uc?export=download&id={GOOGLE_DRIVE_MODEL_ID}"

@st.cache_data
def load_model_and_encoders():
    # Download model from Google Drive if not exists
    if not os.path.exists(MODEL_PATH):
        st.write("Downloading model from Google Drive...")
        response = requests.get(GOOGLE_DRIVE_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)
        st.write("Model downloaded successfully.")

    # Load the model
    model = joblib.load(MODEL_PATH)

    # Load and prepare encoders from CSV
    df = pd.read_csv('rainfall in india 1901-2015.csv')
    df.dropna(inplace=True)
    months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
    df_melted = df.melt(id_vars=['SUBDIVISION', 'YEAR'], value_vars=months,
                        var_name='MONTH', value_name='RAINFALL')

    le_month = LabelEncoder()
    le_subdivision = LabelEncoder()
    le_month.fit(df_melted['MONTH'])
    le_subdivision.fit(df_melted['SUBDIVISION'])

    return model, le_month, le_subdivision

# Load model and encoders
model, le_month, le_subdivision = load_model_and_encoders()

# Streamlit UI
st.title("Rainfall Prediction App 🌧️")
st.markdown("Predict rainfall based on month, region, and year.")

month = st.selectbox("Select Month", list(le_month.classes_))
subdivision = st.selectbox("Select Subdivision", list(le_subdivision.classes_))
year = st.number_input("Enter Year", min_value=2025, max_value=2100, step=1)

if st.button("Predict Rainfall"):
    month_enc = le_month.transform([month])[0]
    sub_enc = le_subdivision.transform([subdivision])[0]
    input_vector = np.array([[month_enc, sub_enc, year]])
    prediction = model.predict(input_vector)[0]
    st.success(f"Expected rainfall in **{subdivision}** during **{month} {year}** is **{prediction:.2f} mm**.")
