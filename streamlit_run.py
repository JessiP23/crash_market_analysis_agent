import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
    
    # Visualize Market Stress Probability
    st.subheader('Market Stress Probability Gauge')
    fig_gauge = plt.figure(figsize=(10, 6))
    plt.pie([probability[1], 1-probability[1]], 
            colors=['red' if probability[1] > 0.5 else 'orange', 'lightgray'],
            labels=[f'Stress: {probability[1]:.1%}', f'Normal: {probability[0]:.1%}'],
            explode=[0.1, 0])
    plt.title('Market Stress Probability Distribution')
    st.pyplot(fig_gauge)
    
    # Visualize Market Indicators
    st.subheader('Current Market Indicators Analysis')
    
    # Prepare data for visualization
    returns_data = {k.split('_')[0]: v for k, v in features.items() if k.endswith('returns')}
    volatility_data = {k.split('_')[0]: v for k, v in features.items() if k.endswith('volatility')}
    
    # Create returns bar chart
    fig_returns = plt.figure(figsize=(10, 6))
    plt.bar(returns_data.keys(), returns_data.values())
    plt.title('Market Returns by Indicator')
    plt.xticks(rotation=45)
    plt.ylabel('Returns')
    plt.grid(True, alpha=0.3)
    st.pyplot(fig_returns)
    
    # Create volatility heat map
    st.subheader('Market Volatility Heatmap')
    volatility_df = pd.DataFrame([volatility_data])
    fig_heatmap = plt.figure(figsize=(10, 3))
    sns.heatmap(volatility_df, cmap='YlOrRd', annot=True, fmt='.2f')
    plt.title('Volatility Levels Across Indicators')
    st.pyplot(fig_heatmap)
    
    # Add market stress interpretation
    st.subheader('Market Analysis Summary')
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('**Key Indicators Status:**')
        for indicator, value in returns_data.items():
            if abs(value) > 0.02:  # Threshold for significant movement
                st.write(f"- {indicator}: {'📈' if value > 0 else '📉'} {value:.2%}")
    
    with col2:
        st.markdown('**Risk Assessment:**')
        risk_level = "High" if probability[1] > 0.7 else "Medium" if probability[1] > 0.3 else "Low"
        st.write(f"- Overall Risk Level: {risk_level}")
        st.write(f"- Highest Volatility: {max(volatility_data.items(), key=lambda x: x[1])[0]}")
        st.write(f"- Market Sentiment: {'Negative' if prediction == 1 else 'Positive'}")

# Add explanation of indicators
st.header('Understanding Market Indicators')
st.markdown("""
- **MXWO**: MSCI World Index - Tracks large and mid-cap equity performance across developed markets
- **MXUS**: MSCI USA Index - Measures the performance of the US equity market
- **GC1**: Gold Futures - Represents gold price movements
- **CL1**: Crude Oil Futures - Tracks oil price movements
- **VIX**: Volatility Index - Measures market's expectation of 30-day volatility
- **DXY**: US Dollar Index - Indicates the general value of the USD
""")

# Footer with disclaimer
st.markdown('---')
st.markdown("""
*Disclaimer: This tool provides predictions based on historical data and should not be used as the sole basis for investment decisions. 
Always conduct thorough research and consult with financial professionals before making investment choices.*
""")