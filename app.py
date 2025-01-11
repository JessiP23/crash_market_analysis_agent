import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# Function to prepare data
def prepare_data(df):
    df['MXWO Index_returns'] = df['MXWO Index'].pct_change()
    df['crash'] = (df['MXWO Index_returns'] <= -0.02).astype(int)
    df = df.dropna()
    feature_columns = [col for col in df.columns if col not in ['crash', 'MXWO Index_returns']]
    return df, feature_columns

# Function to train the model
def train_model(df, feature_columns):
    X = df[feature_columns]
    y = df['crash']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = XGBClassifier(scale_pos_weight=10, random_state=42)
    model.fit(X_train_scaled, y_train)
    return model, scaler, X_test_scaled, y_test

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
    df['MXWO Index'] = df['Close']  # Assuming MXWO Index is the Close price for simplicity

    # Prepare data
    df, feature_columns = prepare_data(df)

    # Train model
    model, scaler, X_test_scaled, y_test = train_model(df, feature_columns)

    # Get latest prediction
    latest_features = X_test_scaled[-1]
    latest_prob = model.predict_proba([latest_features])[0][1]
    risk_level = "HIGH" if latest_prob > 0.5 else "LOW"

    # Display results
    st.write(f"Latest Crash Probability: {latest_prob:.2%}")
    st.write(f"Risk Level: {risk_level}")

    # Plot feature importance
    importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(10, 5))
    importance.head(10).plot(x='feature', y='importance', kind='bar')
    plt.title('Top 10 Most Important Features')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)

if st.button("Clear Data"):
    st.experimental_rerun()