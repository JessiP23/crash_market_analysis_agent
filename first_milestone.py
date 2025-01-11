import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

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
        'report': classification_report(y_test, y_pred)
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