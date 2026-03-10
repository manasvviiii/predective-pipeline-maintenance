import os
import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from dotenv import load_dotenv

load_dotenv()
DB_URL = os.getenv('DB_URL')

def train_maintenance_model():
    # 1. Load data from PostgreSQL
    engine = create_engine(DB_URL)
    df = pd.read_sql('SELECT * FROM train_labeled', engine)
    
    # 2. Prepare Features (X) and Target (y)
    # We drop 'unit_id' and 'RUL' from features
    X = df.drop(['unit_id', 'RUL'], axis=1)
    y = df['RUL']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Start MLflow Experiment
    mlflow.set_experiment("Predictive_Maintenance_RUL")
    
    with mlflow.start_run():
        # Set Model Parameters
        params = {
            "objective": "reg:squarederror",
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 5
        }
        
        # Train Model
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)
        
        # Predict and Evaluate
        predictions = model.predict(X_test)
       # Remove the 'squared=False' part entirely:
        rmse = root_mean_squared_error(y_test, predictions)

        
        # 4. Log to MLflow
        mlflow.log_params(params)
        mlflow.log_metric("rmse", rmse)
        mlflow.xgboost.log_model(model, "model")
        
        print(f"✅ Model Trained. RMSE: {rmse:.2f}")
        print("🚀 Results logged to MLflow Dashboard!")

if __name__ == "__main__":
    train_maintenance_model()
