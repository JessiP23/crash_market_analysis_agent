import pandas as pd
import numpy as np
from datetime import datetime

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