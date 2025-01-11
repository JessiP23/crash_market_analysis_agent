import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

def prepare_data(file_path):
    # Read CSV file, skipping the metadata rows
    df = pd.read_csv(file_path, skiprows=6)
    
    # Drop empty columns and the first date-related columns
    df = df.dropna(axis=1, how='all')
    df = df.iloc[:, 2:]  # Skip the first two columns which are dates/metadata
    
    # Clean column names
    df.columns = [f'col_{i}' for i in range(df.shape[1])]
    
    # Convert to numeric, handling errors
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Forward fill missing values
    df = df.ffill()
    
    # Select the main index column (first column) for crash detection
    main_index = df.iloc[:, 0]
    
    # Calculate features
    returns = df.pct_change(fill_method=None)
    volatility = returns.rolling(window=20).std()
    ma_50 = df.rolling(window=50).mean()
    ma_200 = df.rolling(window=200).mean()
    
    # Create features DataFrame
    features = pd.concat([
        returns,
        volatility,
        ma_50 / df,  # Normalized moving averages
        ma_200 / df,
        (ma_50 - ma_200) / ma_200  # MACD-like indicator
    ], axis=1)
    
    # Define crash conditions
    returns_main = returns.iloc[:, 0]
    volatility_main = volatility.iloc[:, 0]
    
    crash_conditions = (
        (returns_main < -0.05) |  # Single day 5% drop
        (returns_main.rolling(5).sum() < -0.10) |  # 5-day 10% drop
        (volatility_main > volatility_main.quantile(0.95))  # High volatility
    )
    
    # Create target variable
    target = crash_conditions.astype(int)
    
    # Remove NaN values
    features = features.dropna()
    target = target[features.index]
    
    # Verify we have data
    print(f"Dataset shape: {features.shape}")
    print(f"Number of crash events: {target.sum()}")
    print(f"Crash rate: {target.mean():.2%}")
    
    return features, target

def train_and_evaluate_model(model, X, y, model_name):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Print classification report
    print(f"\nClassification Report for {model_name}:")
    print(classification_report(y_test, y_pred))
    
    # Calculate and print ROC AUC
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"ROC AUC Score: {roc_auc:.3f}")
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    return model, scaler

def compare_models(X, y):
    models = {
        'KNN': KNeighborsClassifier(n_neighbors=5, weights='distance', metric='euclidean'),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
        results[name] = scores
        print(f"{name} - Mean ROC AUC: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
    
    # Visualize results
    plt.figure(figsize=(12, 6))
    plt.boxplot(results.values(), labels=results.keys())
    plt.title('Model Comparison - ROC AUC Scores')
    plt.ylabel('ROC AUC Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    return results

def main():
    # Prepare data
    print("Loading and preparing data...")
    X, y = prepare_data('dataset.csv')
    
    if len(y) == 0:
        print("Error: No valid data after preprocessing!")
        return
    
    # Print class distribution
    print("\nClass distribution:")
    print(y.value_counts(normalize=True))
    
    # Compare models
    print("\nComparing models...")
    model_results = compare_models(X, y)
    
    # Train and evaluate the best performing model (assuming Random Forest for this example)
    best_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    print("\nTraining and evaluating the best model (Random Forest)...")
    rf_model, rf_scaler = train_and_evaluate_model(best_model, X, y, "Random Forest")
    
    # Feature importance analysis for Random Forest
    importances = rf_model.feature_importances_
    feature_imp = pd.DataFrame(sorted(zip(importances, X.columns), reverse=True), columns=['Importance', 'Feature'])
    print("\nTop 10 most important features:")
    print(feature_imp.head(10))
    
    # Visualize feature importance
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_imp.head(20))
    plt.title('Top 20 Feature Importance - Random Forest')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()