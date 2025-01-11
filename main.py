import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import matplotlib.pyplot as plt
import joblib  
from xgboost import XGBClassifier

# Load and prepare the data from csv
df = pd.read_csv('dataset.csv')

# convert date to datetime
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Calculate returns and volatility for major indices and assets
feature_columns = ['MXWO Index', 'MXUS Index', 'GC1 Comdty', 'Cl1 Comdty', 'VIX Index', 'DXY Curncy']
for col in feature_columns:
     # Calculate daily returns for each feature
    df[f'{col}_returns'] = df[col].pct_change()
    # Calculate rolling volatility 20 days
    df[f'{col}_volatility'] = df[col].rolling(window=20).std()  


# Remove rows with NaN values
df = df.dropna()  #

# Define market crash conditions using VIX as an indicator
df['market_stress'] = (df['VIX Index'] > df['VIX Index'].mean() + df['VIX Index'].std()).astype(int)

# Prepare features and target variable
X = df[[col for col in df.columns if '_returns' in col or '_volatility' in col]]

# Variable for market stress
y = df['market_stress']

# Scale the features
scaler = StandardScaler()

# Transform the features
X_scaled = scaler.fit_transform(X)

# Train the model using all data
# Select best model 
# XGBoost outperforms accuracy and speed with large datasets.
# Overfitting L1 (Lasso) and L2 (Ridge) regularization
# Parallel processing
model = XGBClassifier(random_state=42)
model.fit(X_scaled, y)

# Save both the model and scaler
joblib.dump(model, 'market_stress_model.pkl')
joblib.dump(scaler, 'feature_scaler.pkl')

# Print feature names for future reference
feature_names = X.columns.tolist()
print("Model and scaler saved successfully!")
print("Feature names for reference:")
print(feature_names)

# Print basic model evaluation metrics on the training data
y_pred = model.predict(X_scaled)

# predictions on training data
accuracy = (y == y_pred).mean()
print(f"Model accuracy on training data: {accuracy:.2%}")

# Define more sophisticated market crash conditions
lookback_period = 20
volatility_window = 60

# Calculate rolling volatility and moving average for VIX
df['MXWO_volatility'] = df['MXWO Index_returns'].rolling(window=volatility_window).std()
df['VIX_MA'] = df['VIX Index'].rolling(window=lookback_period).mean()

# Define crash conditions
crash_threshold = -0.02

# VIX 50% above its moving average
vix_threshold = 1.5  

# Identify crashes based on defined conditions
df['crash'] = ((df['MXWO Index_returns'] <= crash_threshold) & (df['VIX Index'] >= df['VIX_MA'] * vix_threshold)).astype(int)

# Feature engineering
df['yield_curve'] = df['USGG30YR Index'] - df['USGG2YR Index']
df['gold_oil_ratio'] = df['GC1 Comdty'] / df['Cl1 Comdty']
df['vix_change'] = df['VIX Index'].pct_change()

# Select features for the clean dataset
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

# Prepare training and testing data
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

# Select best model 
# XGBoost outperforms accuracy and speed with large datasets.
# Overfitting L1 (Lasso) and L2 (Ridge) regularization
# Parallel processing
best_model = models['XGBoost']

# Feature importance for XGBoost
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    # Get feature importance from the model
    'importance': best_model.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

print("\
Top 10 Most Important Features:")

# Print the top 10 most important feature
print(feature_importance.head(10))




#  Second milestone

# Define a simple investment strategy based on model predictions
def investment_strategy(predictions, returns, threshold=0.5):
    """
    Simulates an investment strategy based on model predictions.
    If the model predicts a crash (probability > threshold), move to cash (0% return).
    Otherwise, invest in the market (use actual returns).
    """

    # list to store strategy returns
    strategy_returns = []


    for pred, ret in zip(predictions, returns):
        if pred > threshold:
            # Move to cash during predicted crashes
            strategy_returns.append(0) 
        else:
            # Use actual market returns otherwise
            strategy_returns.append(ret)  
    return np.array(strategy_returns)

# XGBoost to predict probabilities
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

        #store the model, feature importance, and feature names
        self.model = model
        self.feature_importance = feature_importance
        self.feature_names = feature_names
        
    def explain_prediction(self, features, prediction_prob):
        """Explains the model's prediction in natural language."""

        # Determine risk level based on prediction probability
        risk_level = "high" if prediction_prob > 0.5 else "low"
        
        # Get top contributing features
        top_features = self.feature_importance.head(3)
        
        explanation = f""" Market Crash Risk Assessment:
        - Current risk level: {risk_level} (probability: {prediction_prob:.1%})
        
        Key factors influencing this prediction:
        """
        
        
        for _, row in top_features.iterrows():
            # Get feature name and importance
            feature = row['feature']
            importance = row['importance']

            # add feature explanation
            explanation += f"- {feature}: {importance:.1%} importance\
"
            
        recommendation = """
        Investment Recommendation:
        """
        if prediction_prob > 0.5:
            # Recommendation for high risk
            recommendation += "- Consider moving to defensive positions or cash"
        else:
            # Recommendation for low risk
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

# Save the model and scaler as .pkl files
joblib.dump(best_model, 'market_crash_model.pkl')  # Save the model
joblib.dump(scaler, 'scaler.pkl')              # Save the scaler

print("Model and scaler saved successfully as .pkl files.")
