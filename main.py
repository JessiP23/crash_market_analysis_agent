import pandas as pd
import numpy as np
from datetime import datetime
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# Load and prepare the data
df = pd.read_csv('./dataset.csv')

# Convert Ticker column to datetime
df['Date'] = pd.to_datetime(df['Ticker'])

# Calculate returns for major market indices
df['MXWO_returns'] = df['MXWO Index'].pct_change()
df['MXUS_returns'] = df['MXUS Index'].pct_change()
df['VIX_change'] = df['VIX Index'].pct_change()

# Define market crash conditions (example: significant drop in world index or US index with VIX spike)
crash_threshold = -0.03  # 3% daily drop
vix_threshold = 0.15    # 15% VIX increase

# Create crash label
df['crash'] = ((df['MXWO_returns'] <= crash_threshold) | 
               (df['MXUS_returns'] <= crash_threshold) & 
               (df['VIX_change'] >= vix_threshold)).astype(int)

# Select features for the model
feature_columns = ['XAU BGNL Curncy', 'BDIY Index', 'CRY Index', 'DXY Curncy', 
                  'VIX Index', 'USGG30YR Index', 'GT10 Govt', 'USGG2YR Index',
                  'MXWO Index', 'MXUS Index', 'GC1 Comdty', 'Cl1 Comdty']

# Clean the data
df_clean = df[feature_columns + ['crash']].copy()
df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
df_clean = df_clean.dropna()

# Display basic statistics about crashes
crash_count = df_clean['crash'].sum()
total_samples = len(df_clean)
crash_ratio = crash_count / total_samples

print("Dataset Overview:")
print(f"Total number of samples: {total_samples}")
print(f"Number of crash events: {crash_count}")
print(f"Crash ratio: {crash_ratio:.2%}")

# Display first few rows of prepared dataset
print("\
Prepared Dataset Preview:")
print(df_clean.head())



# Prepare features and target
X = df_clean.drop('crash', axis=1)
y = df_clean['crash']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train XGBoost model
model = xgb.XGBClassifier(
    learning_rate=0.01,
    n_estimators=200,
    max_depth=4,
    min_child_weight=6,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    scale_pos_weight=10,  # Handle class imbalance
    random_state=42
)

model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

print("\
Top 10 Most Important Features:")
print(feature_importance.head(10))