import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import streamlit as st
import joblib  
from xgboost import XGBClassifier
import torch

# %pip install streamlit yfinance scikit-learn tensorflow

# Load and prepare the data
df = pd.read_csv('dataset.csv')
df['Date'] = pd.to_datetime(df['Ticker'])

# Calculate returns for major indices and assets
for col in ['MXWO Index', 'MXUS Index', 'GC1 Comdty', 'Cl1 Comdty']:
    df[f'{col}_returns'] = df[col].pct_change()

# Define market crash conditions (more sophisticated approach)
lookback_period = 20
volatility_window = 60

# Calculate rolling volatility
df['MXWO_volatility'] = df['MXWO Index_returns'].rolling(window=volatility_window).std()
df['VIX_MA'] = df['VIX Index'].rolling(window=lookback_period).mean()

# Define crash conditions
crash_threshold = -0.02  # 2% daily drop
vix_threshold = 1.5     # VIX 50% above its moving average

df['crash'] = ((df['MXWO Index_returns'] <= crash_threshold) & 
               (df['VIX Index'] >= df['VIX_MA'] * vix_threshold)).astype(int)

# Feature engineering
df['yield_curve'] = df['USGG30YR Index'] - df['USGG2YR Index']
df['gold_oil_ratio'] = df['GC1 Comdty'] / df['Cl1 Comdty']
df['vix_change'] = df['VIX Index'].pct_change()

# Select features
feature_columns = [
    'XAU BGNL Curncy', 'BDIY Index', 'CRY Index', 'DXY Curncy',
    'VIX Index', 'USGG30YR Index', 'GT10 Govt', 'USGG2YR Index',
    'yield_curve', 'gold_oil_ratio', 'vix_change',
    'MXWO Index_returns', 'GC1 Comdty_returns', 'Cl1 Comdty_returns'
]

# Prepare clean dataset
df_clean = df[feature_columns + ['crash', 'Date']].copy()
df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
df_clean = df_clean.dropna()

# Split the data chronologically
train_size = int(len(df_clean) * 0.8)
df_train = df_clean.iloc[:train_size]
df_test = df_clean.iloc[train_size:]

X_train = df_train[feature_columns]
y_train = df_train['crash']
X_test = df_test[feature_columns]
y_test = df_test['crash']

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train multiple models
models = {
    'XGBoost': xgb.XGBClassifier(scale_pos_weight=10, random_state=42),
    'RandomForest': RandomForestClassifier(class_weight='balanced', random_state=42),
    'LogisticRegression': LogisticRegression(class_weight='balanced', random_state=42)
}

# Train and evaluate models
results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    results[name] = {
        'accuracy': model.score(X_test_scaled, y_test),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'report': classification_report(y_test, y_pred, zero_division=1)
    }

# Print results
for name, metrics in results.items():
    print(f"\
{name} Results:")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"ROC AUC: {metrics['roc_auc']:.3f}")
    print("Classification Report:")
    print(metrics['report'])

# Select best model (XGBoost in this case)
best_model = models['XGBoost']

# Feature importance for XGBoost
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': best_model.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

print("\
Top 10 Most Important Features:")
print(feature_importance.head(10))




#  Second milestone

# Milestone 2: Develop a data-driven investment strategy

# Define a simple investment strategy based on model predictions
def investment_strategy(predictions, returns, threshold=0.5):
    """
    Simulates an investment strategy based on model predictions.
    If the model predicts a crash (probability > threshold), move to cash (0% return).
    Otherwise, invest in the market (use actual returns).
    """
    strategy_returns = []
    for pred, ret in zip(predictions, returns):
        if pred > threshold:
            strategy_returns.append(0)  # Move to cash during predicted crashes
        else:
            strategy_returns.append(ret)  # Use actual market returns otherwise
    return np.array(strategy_returns)

# Use the best model (XGBoost) to predict probabilities
y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]

# Simulate the investment strategy
market_returns = df_test['MXWO Index_returns'].values
strategy_returns = investment_strategy(y_pred_proba, market_returns)

# Calculate cumulative returns for both strategies
cumulative_market_returns = np.cumprod(1 + market_returns) - 1
cumulative_strategy_returns = np.cumprod(1 + strategy_returns) - 1

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(df_test['Date'], cumulative_market_returns, label='Market Returns', color='blue')
plt.plot(df_test['Date'], cumulative_strategy_returns, label='Strategy Returns', color='green')
plt.title('Cumulative Returns: Market vs. Strategy')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.grid()
plt.show()

# Print final cumulative returns
print("Final Cumulative Market Returns:", cumulative_market_returns[-1])
print("Final Cumulative Strategy Returns:", cumulative_strategy_returns[-1])




# third milestone
# Milestone 3: AI-driven bot for explaining investment strategy

class InvestmentBot:
    def __init__(self, model, feature_importance, feature_names):
        self.model = model
        self.feature_importance = feature_importance
        self.feature_names = feature_names
        
    def explain_prediction(self, features, prediction_prob):
        """Explains the model's prediction in natural language."""
        risk_level = "high" if prediction_prob > 0.5 else "low"
        
        # Get top contributing features
        top_features = self.feature_importance.head(3)
        
        explanation = f"""
        Market Crash Risk Assessment:
        - Current risk level: {risk_level} (probability: {prediction_prob:.1%})
        
        Key factors influencing this prediction:
        """
        
        for _, row in top_features.iterrows():
            feature = row['feature']
            importance = row['importance']
            explanation += f"- {feature}: {importance:.1%} importance\
"
            
        recommendation = """
        Investment Recommendation:
        """
        if prediction_prob > 0.5:
            recommendation += "- Consider moving to defensive positions or cash"
        else:
            recommendation += "- Market conditions appear favorable for maintaining positions"
            
        return explanation + recommendation

# Demonstrate the bot's functionality with the latest data point
latest_features = X_test_scaled[-1]
latest_prob = best_model.predict_proba(latest_features.reshape(1, -1))[0, 1]

# Create bot instance
investment_bot = InvestmentBot(best_model, feature_importance, feature_columns)

# Get and print explanation
explanation = investment_bot.explain_prediction(latest_features, latest_prob)
print(explanation)

# improve the bot
'''
# Enhancing the chatbot with testing logic and detailed explanations

class InvestmentBot:
    def __init__(self, model, feature_importance, feature_names):
        self.model = model
        self.feature_importance = feature_importance
        self.feature_names = feature_names
        
    def explain_prediction(self, features, prediction_prob):
        """Explains the model's prediction in natural language."""
        risk_level = "high" if prediction_prob > 0.5 else "low"
        
        # Get top contributing features
        top_features = self.feature_importance.head(3)
        
        explanation = f"""
        Market Crash Risk Assessment:
        - Current risk level: {risk_level} (probability: {prediction_prob:.1%})
        
        Key factors influencing this prediction:
        """
        
        for _, row in top_features.iterrows():
            feature = row['feature']
            importance = row['importance']
            explanation += f"- {feature}: {importance:.1%} importance\
"
            
        recommendation = """
        Investment Recommendation:
        """
        if prediction_prob > 0.5:
            recommendation += "- Consider moving to defensive positions or cash"
        else:
            recommendation += "- Market conditions appear favorable for maintaining positions"
            
        return explanation + recommendation

    def test_bot(self, test_data, test_labels):
        """Tests the bot's performance on a test dataset."""
        predictions = self.model.predict(test_data)
        probabilities = self.model.predict_proba(test_data)[:, 1]
        
        # Evaluate performance
        report = classification_report(test_labels, predictions)
        auc = roc_auc_score(test_labels, probabilities)
        
        print("Bot Test Results:")
        print("Classification Report:")
        print(report)
        print("ROC AUC Score:", auc)

# Detailed explanations for each milestone
milestone_details = {
    "Milestone 1": "Developed an anomaly detection model using XGBoost, Random Forest, and Logistic Regression. XGBoost was chosen as the best model due to its superior performance in handling imbalanced datasets, high accuracy, and interpretability through feature importance.",
    "Milestone 2": "Designed a data-driven investment strategy that uses the model's predictions to minimize losses during market crashes by moving to cash and maximizing returns during stable periods.",
    "Milestone 3": "Integrated an AI-driven bot to explain the investment strategy to end users, providing actionable insights and recommendations based on model predictions."
}

# Why XGBoost was the best option
xgboost_reasoning = "XGBoost was selected as the best model because it handles imbalanced datasets effectively using scale_pos_weight, provides high accuracy and ROC AUC scores, and offers feature importance metrics for interpretability. Its ability to capture non-linear relationships and interactions between features makes it ideal for financial anomaly detection."

# Create bot instance
investment_bot = InvestmentBot(best_model, feature_importance, feature_columns)

# Test the bot
investment_bot.test_bot(X_test_scaled, y_test)

# Print milestone details and reasoning
print("Milestone Details:")
for milestone, details in milestone_details.items():
    print(milestone + ": " + details)

print("\
Why XGBoost was the Best Option:")
print(xgboost_reasoning)
'''



import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib

# Function to prepare data
def prepare_data(df):
    # Set the first column as the index and convert it to datetime
    df.index = pd.to_datetime(df.iloc[:, 0], errors='coerce')  # Use the first column as the date index
    df = df.iloc[:, 1:]  # Drop the first column after setting it as index
    
    # Calculate returns for MXWO Index (world market index)
    df['MXWO Index_returns'] = df['MXWO Index'].pct_change()
    
    # Define crash as a drop of more than 2% in world market
    df['crash'] = (df['MXWO Index_returns'] <= -0.02).astype(int)
    
    # Remove rows with NaN values
    df = df.dropna()
    
    # Exclude 'crash' and 'MXWO Index_returns' from features
    feature_columns = [col for col in df.columns if col not in ['crash', 'MXWO Index_returns']]
    
    return df, feature_columns

# Function to train the model
def train_model(df, feature_columns):
    X = df[feature_columns]
    y = df['crash']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = XGBClassifier(scale_pos_weight=10, random_state=42)
    model.fit(X_train_scaled, y_train)
    return model, scaler

# Load your dataset
df = pd.read_csv('dataset.csv', header=0)  # Load the dataset without parsing dates initially
df, feature_columns = prepare_data(df)

# Train the model
model, scaler = train_model(df, feature_columns)

# Save the model and scaler as .pkl files
joblib.dump(model, 'market_crash_model.pkl')  # Save the model
joblib.dump(scaler, 'scaler.pkl')              # Save the scaler

print("Model and scaler saved successfully as .pkl files.")