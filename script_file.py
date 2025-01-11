import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# 1. Load and prepare data
def prepare_data(file_path='dataset.csv'):
    # Load the data
    df = pd.read_csv(file_path)
    
    # Calculate returns for MXWO Index (world market index)
    df['MXWO Index_returns'] = df['MXWO Index'].pct_change()
    
    # Define crash as a drop of more than 2% in world market
    df['crash'] = (df['MXWO Index_returns'] <= -0.02).astype(int)
    
    # Remove first row (NaN from pct_change) and any other NaN values
    df = df.dropna()
    
    # All columns except 'crash' and returns will be features
    feature_columns = [col for col in df.columns if col not in ['crash', 'MXWO Index_returns', 'Ticker']]
    
    return df, feature_columns

# 2. Train model
def train_model(df, feature_columns):
    # Prepare features and target
    X = df[feature_columns]
    y = df['crash']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = XGBClassifier(scale_pos_weight=10, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Calculate accuracy
    accuracy = model.score(X_test_scaled, y_test)
    print(f"\nModel Accuracy: {accuracy:.2%}")
    
    # Get feature importance
    importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return model, scaler, X_test_scaled, y_test, importance

# 3. Investment Bot
class InvestmentBot:
    def __init__(self, model, importance):
        self.model = model
        self.importance = importance
    
    def analyze_market(self, features, prediction_prob):
        risk_level = "HIGH" if prediction_prob > 0.5 else "LOW"
        
        analysis = f"""
Market Crash Risk Analysis:
--------------------------
Risk Level: {risk_level}
Crash Probability: {prediction_prob:.1%}

Top 5 Important Factors:
"""
        
        for _, row in self.importance.head(5).iterrows():
            analysis += f"- {row['feature']}: {row['importance']:.1%}\n"
            
        analysis += f"\nRecommendation: {'MOVE TO CASH' if prediction_prob > 0.5 else 'STAY INVESTED'}"
        
        return analysis

def main():
    # Load and prepare data
    print("Loading data...")
    df, feature_columns = prepare_data()
    print(f"Number of features being used: {len(feature_columns)}")
    
    # Train model
    print("Training model...")
    model, scaler, X_test_scaled, y_test, importance = train_model(df, feature_columns)
    
    # Create bot
    bot = InvestmentBot(model, importance)
    
    # Get latest market prediction
    latest_features = X_test_scaled[-1]
    latest_prob = model.predict_proba([latest_features])[0][1]
    
    # Get and print analysis
    analysis = bot.analyze_market(latest_features, latest_prob)
    print("\nCurrent Market Analysis:")
    print(analysis)
    
    # Plot feature importance
    plt.figure(figsize=(15, 8))
    importance.head(10).plot(x='feature', y='importance', kind='bar')
    plt.title('Top 10 Most Important Features')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()