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