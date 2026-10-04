from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
def evaluate_predictions(y_true, predictions):
    """
    Evaluate model predictions using MAE, RMSE and R².
    """
    mae = mean_absolute_error(y_true, predictions)
    rmse = mean_squared_error(
        y_true,
        predictions
    ) ** 0.5
    r2 = r2_score(
        y_true,
        predictions
    )
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    }