import streamlit as st
import pandas as pd
import joblib
import yfinance as yf
import numpy as np

# Load the pre-trained model and scaler
model = joblib.load('market_crash_model.pkl')
scaler = joblib.load('scaler.pkl')

# Function to prepare data
def prepare_data(df):
    # Print the columns for debugging
    print("Columns in the DataFrame:", df.columns.tolist())
    
    # Ensure only relevant columns are kept
    relevant_columns = ['MXWO Index', 'BDIY Index', 'CO1 Comdty', 'CRY Index', 'Cl1 Comdty', 'DU1 Comdty']  # Add all relevant features used during training
    
    # Check if all relevant columns are in the DataFrame
    missing_columns = [col for col in relevant_columns if col not in df.columns]
    if missing_columns:
        raise KeyError(f"Missing columns in DataFrame: {missing_columns}")
    
    df = df[relevant_columns]  # Keep only relevant columns

    # Calculate returns for MXWO Index (world market index)
    df['MXWO Index_returns'] = df['MXWO Index'].pct_change()
    
    # Define crash as a drop of more than 2% in world market
    df['crash'] = (df['MXWO Index_returns'] <= -0.02).astype(int)
    
    # Remove rows with NaN values
    df = df.dropna()
    
    # Exclude 'crash' and 'MXWO Index_returns' from features
    feature_columns = [col for col in df.columns if col not in ['crash', 'MXWO Index_returns']]
    
    return df, feature_columns

# Streamlit app
st.title("Market Crash Prediction App")

# User input for stock ticker
ticker = st.text_input("Enter Stock Ticker (e.g., ^GSPC for S&P 500):", "^GSPC")
start_date = st.date_input("Start Date", value=pd.to_datetime("2010-01-01"))
end_date = st.date_input("End Date", value=pd.to_datetime("today"))

if st.button("Fetch Data"):
    # Fetch data from yfinance
    df = yf.download(ticker, start=start_date, end=end_date)
    df.reset_index(inplace=True)
    
    # Prepare data
    df, feature_columns = prepare_data(df)

    # Scale features
    X = df[feature_columns]
    X_scaled = scaler.transform(X)  # Use the scaler to transform the features

    # Get latest prediction
    latest_features = X_scaled[-1]
    latest_prob = model.predict_proba([latest_features])[0][1]
    risk_level = "HIGH" if latest_prob > 0.5 else "LOW"

    # Display results
    st.write(f"Latest Crash Probability: {latest_prob:.2%}")
    st.write(f"Risk Level: {risk_level}")

if st.button("Clear Data"):
    st.experimental_rerun()