import mlflow.xgboost
import pandas as pd
from fastapi import FastAPI
import uvicorn

app = FastAPI()

# Pointing to the specific Run ID you found
RUN_ID = "459ae6d8a88644fba1f56bb3771aaa07"
model = mlflow.xgboost.load_model(f"runs:/{RUN_ID}/model")

@app.get("/")
def home():
    return {"status": "Maintenance API is Online", "run_id": RUN_ID}

@app.post("/predict")
def predict(data: dict):
    # Convert input JSON to DataFrame for the model
    input_df = pd.DataFrame([data])
    prediction = model.predict(input_df)
    return {"RUL_Prediction": float(prediction[0])}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
