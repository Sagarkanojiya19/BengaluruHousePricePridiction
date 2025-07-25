
import pickle
import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder,FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

class LogTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.Series):
            return np.log1p(X.values).reshape(-1, 1)  # Fix reshape issue
        return np.log1p(X) 

# Load the trained model
with open('model.pkl', 'rb') as file:
    model = joblib.load(file)

# Title of the app
st.title("Bengaluru House Price Prediction")

# Input fields
location = st.text_input("Location:")
size = st.text_input("Size (e.g., 1 BHK, 2 BHK):")
total_sqft = st.number_input("Total Square Feet:", min_value=300.0, max_value=10000.0, step=50.0)
bath = st.number_input("Number of Bathrooms:", min_value=1, max_value=10, step=1)

# Predict button
if st.button("Predict Price"):
    if location and size:
        # Create input dataframe
        input_data = pd.DataFrame({
            'location': [location],
            'size': [size],
            'total_sqft': [total_sqft],
            'bath': [bath]
        })

        # Predict
        prediction = model.predict(input_data)
        predicted_price = np.expm1(prediction[0])  # Reverse log transformation

        # Display result
        st.success(f"Estimated House Price: ₹{predicted_price:,.2f}lakhs")
    else:
        st.error("Please enter both location and size.")
