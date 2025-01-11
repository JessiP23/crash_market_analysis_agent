import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the saved model and scaler
model = joblib.load('market_stress_model.pkl')
scaler = joblib.load('feature_scaler.pkl')

# Set page title
st.title('Market Stress Predictor')

# Create input fields for features
st.header('Enter Market Indicators')

# Create input fields for each feature
features = {}
for feature, display_name in zip(
    ['MXWO Index', 'MXUS Index', 'GC1 Comdty', 'Cl1 Comdty', 'VIX Index', 'DXY Curncy'],
    ['MXWO', 'MXUS', 'GC1', 'Cl1', 'VIX', 'DXY']):
    col1, col2 = st.columns(2)
    with col1:
        features[f'{feature}_returns'] = st.number_input(f'{display_name} Returns', value=0.0)
    with col2:
        features[f'{feature}_volatility'] = st.number_input(f'{display_name} Volatility', value=0.0)

if st.button('Predict Market Stress'):
    # Prepare input data
    input_data = pd.DataFrame([features])
    
    # Scale the input data
    scaled_input = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0]
    
    # Show results
    st.header('Prediction Results')
    if prediction == 1:
        st.warning('Market Stress Detected!')
    else:
        st.success('Normal Market Conditions')
    
    st.write(f'Probability of Market Stress: {probability[1]:.2%}')