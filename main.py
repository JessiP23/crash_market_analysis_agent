import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

def prepare_data(file_path):
    # Read CSV file
    df = pd.read_csv(file_path, skiprows=6)  # Skip header rows
    
    # Clean column names
    df.columns = [f'col_{i}' for i in range(len(df.columns))]
    
    # Convert all columns to numeric, replacing errors with NaN
    numeric_df = df.apply(pd.to_numeric, errors='coerce')
    
    # Forward fill missing values
    numeric_df = numeric_df.ffill()
    
    # Calculate returns for all columns
    returns = numeric_df.pct_change()
    
    # Create rolling volatility features
    volatility = returns.rolling(window=20).std()
    
    # Create moving averages
    ma_50 = numeric_df.rolling(window=50).mean()
    ma_200 = numeric_df.rolling(window=200).mean()
    
    # Create features DataFrame
    features = pd.concat([
        returns,
        volatility,
        ma_50,
        ma_200,
        (ma_50 - ma_200) / ma_200  # Moving average convergence/divergence
    ], axis=1)
    
    # Define crash conditions (multiple criteria)
    crash_conditions = (
        (returns.iloc[:, 0] < -0.10) |  # Large single-day drop
        (returns.iloc[:, 0].rolling(5).sum() < -0.15) |  # Sustained decline
        (volatility.iloc[:, 0] > volatility.iloc[:, 0].quantile(0.95))  # High volatility
    )
    
    # Create target variable
    target = crash_conditions.astype(int)
    
    # Remove NaN values
    features = features.dropna()
    target = target[features.index]
    
    return features, target

def train_model(X, y):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize and train the model with optimized parameters
    model = XGBClassifier(
        learning_rate=0.01,
        n_estimators=200,
        max_depth=4,
        min_child_weight=6,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        scale_pos_weight=1,  # Adjust based on class imbalance
        random_state=42
    )
    
    # Perform cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    print("\nCross-validation scores:", cv_scores)
    print("Average CV score:", cv_scores.mean())
    
    # Train the model
    model.fit(
        X_train_scaled, 
        y_train,
        eval_set=[(X_test_scaled, y_test)],
        eval_metric='auc',
        early_stopping_rounds=50,
        verbose=100
    )
    
    return model, scaler, X_test_scaled, y_test

def evaluate_model(model, X_test, y_test):
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Calculate and print ROC AUC score
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\nROC AUC Score: {roc_auc:.4f}")
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

def plot_feature_importance(model, feature_names):
    # Get feature importance
    importance = model.feature_importances_
    
    # Create DataFrame of features and their importance scores
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    # Plot top 20 most important features
    plt.figure(figsize=(12, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(20))
    plt.title('Top 20 Most Important Features')
    plt.show()
    
    return feature_importance

def main():
    # Prepare data
    print("Loading and preparing data...")
    X, y = prepare_data('dataset.csv')
    
    # Print class distribution
    print("\nClass distribution:")
    print(y.value_counts(normalize=True))
    
    # Train model
    print("\nTraining model...")
    model, scaler, X_test_scaled, y_test = train_model(X, y)
    
    # Evaluate model
    print("\nEvaluating model...")
    evaluate_model(model, X_test_scaled, y_test)
    
    # Plot feature importance
    print("\nAnalyzing feature importance...")
    feature_importance = plot_feature_importance(model, X.columns)
    
    # Print top 10 most important features
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))

if __name__ == "__main__":
    main()