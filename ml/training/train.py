from pathlib import Path
from ml.data.loader import load_dataset
from ml.feature_engineering.features import create_features
from ml.models.linear_regression import create_linear_regression
from ml.models.random_forest import create_random_forest
from ml.models.xg_boost import create_xgboost
from ml.evaluation.evaluation import evaluate_predictions
from ml.models.model_saver import save_model

FEATURE_COLUMNS=["day","month","quarter","day_of_week","week_of_year","is_weekend","lag_1","lag_7","lag_14","lag_30","rolling_mean_7","rolling_mean_14","rolling_mean_30","price","discount","promotion","inventory"]

def prepare_training_data(file_path:str):
    df=load_dataset(file_path)
    for column in ("price","discount","promotion","inventory"):
        if column not in df: df[column]=0
    df=create_features(df).sort_values("date",kind="stable").dropna(subset=FEATURE_COLUMNS+["sales"])
    return df[FEATURE_COLUMNS],df["sales"],df["date"]

def train_models(X_train,X_test,y_train,y_test):
    models={"Linear Regression":create_linear_regression(),"Random Forest":create_random_forest(),"XGBoost":create_xgboost()}
    results={}
    for name,model in models.items():
        model.fit(X_train,y_train); predictions=model.predict(X_test)
        metrics=evaluate_predictions(y_test,predictions)
        results[name]={"model":model,"mae":metrics["mae"],"rmse":metrics["rmse"],"r2":metrics["r2"]}
    return results

if __name__=="__main__":
    root=Path(__file__).resolve().parents[2]
    X,y,dates=prepare_training_data(str(root/"data"/"sample"/"sales.csv"))
    cutoff=dates.sort_values().unique()[max(1,int(dates.nunique()*.8))]
    mask=dates<cutoff
    X_train,X_test,y_train,y_test=X.loc[mask],X.loc[~mask],y.loc[mask],y.loc[~mask]
    print("Training samples:",len(X_train));print("Testing samples:",len(X_test))
    results=train_models(X_train,X_test,y_train,y_test)
    for name,result in results.items():
        print(name,"MAE:",round(result["mae"],2),"RMSE:",round(result["rmse"],2),"R²:",round(result["r2"],4))
    best=min(results,key=lambda name:results[name]["rmse"])
    print("Best model:",best)
    model_path=save_model(results[best]["model"],company_id="company_1",version="model_v1")
    print("Model saved at:",model_path)
