import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# visualization
import seaborn as sns

# Load the saved model and scaler
model = joblib.load('market_stress_model.pkl')
scaler = joblib.load('feature_scaler.pkl')

# Set page configuration for streamlit
st.set_page_config(page_title="Market Stress Early Warning System", layout="wide")

# Main title and description
st.title('Market Stress Early Warning System')
st.markdown("""
This system uses machine learning to detect market anomalies and provide early warnings 
of potential market stress conditions. It analyzes multiple market indicators to assess risk 
and recommend mitigation strategies.
""")

# Create tabs for different sections
tab1, tab2, tab3 = st.tabs(["Market Analysis", "Risk Assessment", "Strategy Recommendations"])

with tab1:
    st.header('Enter Market Indicators')
    
    # Create input fields for market indicators
    features = {}
    for feature, display_name in zip(
        ['MXWO Index', 'MXUS Index', 'GC1 Comdty', 'Cl1 Comdty', 'VIX Index', 'DXY Curncy'],
        ['MXWO', 'MXUS', 'GC1', 'Cl1', 'VIX', 'DXY']):

        # columns for returns and volatility
        col1, col2 = st.columns(2)
        with col1:
            features[f'{feature}_returns'] = st.number_input(f'{display_name} Returns', value=0.0)
        with col2:
            features[f'{feature}_volatility'] = st.number_input(f'{display_name} Volatility', value=0.0)


    # Analyze market conditions
    if st.button('Analyze Market Conditions'):
        # Prepare input data
        input_data = pd.DataFrame([features])
        
        # Scale the input data
        scaled_input = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(scaled_input)[0]
        probability = model.predict_proba(scaled_input)[0]
        
        # Store results in session state for other tabs
        st.session_state['prediction'] = prediction
        st.session_state['probability'] = probability
        st.session_state['features'] = features
        
        # Show results with enhanced visualization
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader('Market Status')
            if prediction == 1:
                 # Display error message for market stress
                st.error('⚠️ MARKET STRESS DETECTED')
                st.warning(f'Stress Probability: {probability[1]:.1%}')
            else:
                 # normal conditions
                st.success('✅ NORMAL MARKET CONDITIONS')
                st.info(f'Stress Probability: {probability[1]:.1%}')

        with col2:
            # Stress probability gauge
            fig_gauge = plt.figure(figsize=(8, 4))
            plt.pie([probability[1], 1-probability[1]], 
                   colors=['red' if probability[1] > 0.5 else 'orange', 'lightgray'],
                   labels=[f'Stress: {probability[1]:.1%}', f'Normal: {probability[0]:.1%}'],
                   explode=[0.1, 0])
            plt.title('Market Stress Probability')
            st.pyplot(fig_gauge)

        # Anomaly Detection Section
        st.subheader('Anomaly Detection Analysis')
        
        # Calculate z-scores for returns and volatility
        returns_data = {k.split('_')[0]: v for k, v in features.items() if k.endswith('returns')}
        volatility_data = {k.split('_')[0]: v for k, v in features.items() if k.endswith('volatility')}
        
        # Create anomaly detection visualization
        fig_anomaly = plt.figure(figsize=(12, 6))
        plt.scatter(list(returns_data.values()), list(volatility_data.values()), 
                   c=['red' if abs(r)+v > 0.1 else 'blue' for r, v in zip(returns_data.values(), volatility_data.values())],
                   s=100)
        plt.xlabel('Returns')
        plt.ylabel('Volatility')
        for i, txt in enumerate(returns_data.keys()):
            plt.annotate(txt, (list(returns_data.values())[i], list(volatility_data.values())[i]))
        plt.title('Market Indicators Anomaly Detection')
        plt.grid(True)
        st.pyplot(fig_anomaly)

with tab2:
    if 'prediction' in st.session_state:
        st.header('Risk Assessment Dashboard')
        
        # Risk Metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Overall Risk Level",
                value=f"{st.session_state['probability'][1]:.1%}",
                delta=f"{'↑' if st.session_state['probability'][1] > 0.5 else '↓'} Risk"
            )
        
        with col2:
            volatility_score = np.mean(list(volatility_data.values()))
            st.metric(
                label="Average Volatility",
                value=f"{volatility_score:.2f}",
                delta=f"{'High' if volatility_score > 0.05 else 'Normal'}"
            )
        
        with col3:
            returns_score = np.mean(list(returns_data.values()))
            st.metric(
                label="Average Returns",
                value=f"{returns_score:.2%}",
                delta=f"{'Negative' if returns_score < 0 else 'Positive'}"
            )
        
        # Risk Factors Analysis
        st.subheader('Key Risk Factors')
        risk_factors = pd.DataFrame({
            'Indicator': list(features.keys()),
            'Value': list(features.values()),
            'Risk Level': ['High' if abs(v) > 0.05 else 'Medium' if abs(v) > 0.02 else 'Low' for v in features.values()]
        })
        
        st.dataframe(risk_factors.style.apply(lambda x: ['background: #ffcdd2' if v == 'High' 
                                                        else 'background: #fff9c4' if v == 'Medium'
                                                        else 'background: #c8e6c9' for v in x], 
                                            subset=['Risk Level']))

with tab3:
    if 'prediction' in st.session_state:
        st.header('Investment Strategy Recommendations')
        
        # Define risk level
        risk_level = ("High" if st.session_state['probability'][1] > 0.7 
                     else "Medium" if st.session_state['probability'][1] > 0.3 
                     else "Low")
        
        # Strategy recommendations based on risk level
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader('Asset Allocation Strategy')
            if risk_level == "High":
                st.warning("""
                ### Defensive Position Recommended
                - 🛡️ Increase allocation to defensive assets (40-50%)
                - 💰 Maintain higher cash reserves (20-30%)
                - 📊 Reduce exposure to high-beta stocks
                - 🏦 Focus on quality bonds and defensive sectors
                - 🌐 Consider safe-haven currencies
                """)
            elif risk_level == "Medium":
                st.info("""
                ### Balanced Approach Recommended
                - ⚖️ Maintain balanced portfolio allocation
                - 📈 Selective exposure to growth assets
                - 🛡️ Keep moderate defensive position
                - 💰 Standard cash reserves (10-15%)
                - 🔄 Regular rebalancing
                """)
            else:
                st.success("""
                ### Growth Opportunity
                - 📈 Increase exposure to growth assets
                - 🎯 Look for market opportunities
                - 💼 Standard diversification
                - 💰 Normal cash reserves (5-10%)
                - 📊 Regular monitoring
                """)
        
        with col2:
            st.subheader('Risk Mitigation Actions')
            st.markdown("""
            ### Immediate Actions:
            1. Review portfolio allocation
            2. Check stop-loss levels
            3. Assess hedging requirements
            4. Monitor key indicators daily
            5. Prepare contingency plans
            
            ### Medium-term Strategy:
            1. Diversification review
            2. Sector rotation analysis
            3. Risk exposure assessment
            4. Liquidity management
            5. Stress testing scenarios
            """)

# Footer with disclaimer and data update time
st.markdown('---')
st.markdown("""
*Disclaimer: This tool provides predictions based on historical data and should not be used as the sole basis for investment decisions. 
Always conduct thorough research and consult with financial professionals before making investment choices.*

Last data update: {}
""".format(pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')))