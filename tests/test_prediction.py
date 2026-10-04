from ml.prediction.predictor import predict_sales


features = {
    "day": 15,
    "month": 5,
    "quarter": 2,
    "day_of_week": 4,
    "week_of_year": 20,
    "is_weekend": 0,
    "lag_1": 120,
    "lag_7": 118,
    "lag_14": 121,
    "lag_30": 119,
    "rolling_mean_7": 119.5,
    "rolling_mean_14": 120.0,
    "rolling_mean_30": 119.2,
    "price": 500,
    "discount": 5,
    "promotion": 1,
    "inventory": 400,
}


prediction = predict_sales(features)

print("Predicted Sales:", prediction)